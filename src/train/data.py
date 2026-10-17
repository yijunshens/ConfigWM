"""Training dataset that mixes map-style and iterable adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
import logging
import math
from pathlib import Path
import random
from typing import (
    Any,
    Generic,
    TypeVar,
    Union,
)
import warnings

from src.data.adapters import create_adapter, resolve_dataset_spec
from src.data.config import load_data_config as _load_data_config
from src.data.preprocessor import Preprocessor, create_preprocessor
from src.data.schema import PreprocessedSample, Sample
from src.data.seeding import seed_encoder_data
from src.data.sharding import get_shard_info
from src.data.specs import ResolvedDatasetSpec
from src.train.sample_transforms import DatasetSampleTransform
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)

T = TypeVar("T")
DatasetConfig = Union[str, Path, Mapping[str, Any]]
ChildDataset = Dataset[Sample] | IterableDataset[Sample]
IteratorFactory = Callable[[int], Iterator[Sample]]
REPEAT_WARNING_INTERVAL = 10


@dataclass
class DatasetEntry:
    resolved_spec: ResolvedDatasetSpec
    dataset: ChildDataset
    sample_transform: DatasetSampleTransform
    raw_length: int
    effective_length: int


def rank_shard_length(total_items: int, world_size: int) -> int:
    """Number of samples each rank should see with drop-last semantics."""
    if total_items < 0:
        raise ValueError("total_items must be >= 0")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    return total_items // world_size


def worker_length_from_rank_length(
    rank_length: int,
    worker_id: int,
    num_workers: int,
) -> int:
    """Split a rank-local length across workers."""
    if rank_length < 0:
        raise ValueError("rank_length must be >= 0")
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    if not 0 <= worker_id < num_workers:
        raise ValueError("worker_id must be in [0, num_workers)")
    return len(range(worker_id, rank_length, num_workers))


def worker_target_count(total_items: int) -> int:
    """Nominal worker-local sample count for a globally sized dataset."""
    info = get_shard_info()
    rank_length = rank_shard_length(total_items, info.world_size)
    return worker_length_from_rank_length(
        rank_length,
        info.worker_id,
        info.num_workers,
    )


def effective_dataset_length(raw_length: int, weight: float) -> int:
    """Scale a raw dataset length by its configured sampling weight."""
    if raw_length < 0:
        raise ValueError("raw_length must be >= 0")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and >= 0")
    return math.floor(raw_length * weight)


def required_overdraw_passes(target_count: int, pass_count: int) -> int:
    """Replay passes needed after the first iterator pass."""
    if target_count < 0:
        raise ValueError("target_count must be >= 0")
    if pass_count < 0:
        raise ValueError("pass_count must be >= 0")
    if target_count <= 0 or pass_count <= 0:
        return 0
    return max(0, math.ceil(target_count / pass_count) - 1)


class ShardedMapDatasetView(IterableDataset[T], Generic[T]):
    """Iterable worker-local view over a map-style dataset."""

    def __init__(
        self,
        dataset: Dataset[T],
        *,
        seed: int = 0,
        epoch: int = 0,
        shuffle: bool = True,
        split_inbatch_groups_across_ranks: bool = False,
    ) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = epoch
        self.shuffle = shuffle
        self.split_inbatch_groups_across_ranks = (
            split_inbatch_groups_across_ranks
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def rank_length(self, *, world_size: int | None = None) -> int:
        world_size = (
            get_shard_info().world_size if world_size is None else world_size
        )
        return rank_shard_length(len(self.dataset), world_size)

    def worker_length(
        self,
        *,
        worker_id: int | None = None,
        num_workers: int | None = None,
    ) -> int:
        info = get_shard_info()
        worker_id = info.worker_id if worker_id is None else worker_id
        num_workers = info.num_workers if num_workers is None else num_workers
        rank_length = self.rank_length(world_size=info.world_size)
        return worker_length_from_rank_length(
            rank_length, worker_id, num_workers
        )

    def _sharded_indices(self, *, world_size: int) -> list[int]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)

        total_kept = self.rank_length(world_size=world_size) * world_size
        return indices[:total_kept]

    def worker_indices(
        self,
        *,
        rank: int | None = None,
        world_size: int | None = None,
        worker_id: int | None = None,
        num_workers: int | None = None,
    ) -> list[int]:
        info = get_shard_info()
        rank = info.rank if rank is None else rank
        world_size = info.world_size if world_size is None else world_size
        worker_id = info.worker_id if worker_id is None else worker_id
        num_workers = info.num_workers if num_workers is None else num_workers

        group_size = int(getattr(self.dataset, "inbatch_group_size", 1))
        base_length = int(getattr(self.dataset, "base_length", 0))
        if group_size > 1 and base_length > 0:
            groups = list(range(base_length))
            if self.shuffle:
                random.Random(self.seed + self.epoch).shuffle(groups)
            if self.split_inbatch_groups_across_ranks and world_size > 1:
                if group_size % world_size != 0:
                    raise ValueError(
                        "inbatch_group_size must be divisible by world_size "
                        "when splitting groups across ranks: "
                        f"group_size={group_size}, world_size={world_size}"
                    )
                worker_groups = groups[worker_id::num_workers]
                return [
                    group * group_size + slot
                    for group in worker_groups
                    for slot in range(rank, group_size, world_size)
                ]
            total_kept = rank_shard_length(base_length, world_size) * world_size
            groups = groups[:total_kept]
            rank_groups = groups[rank : len(groups) : world_size]
            worker_groups = rank_groups[worker_id::num_workers]
            return [
                group * group_size + slot
                for group in worker_groups
                for slot in range(group_size)
            ]

        indices = self._sharded_indices(world_size=world_size)
        rank_indices = indices[rank : len(indices) : world_size]
        return rank_indices[worker_id::num_workers]

    def __iter__(self) -> Iterator[T]:
        for index in self.worker_indices():
            yield self.dataset[index]


class WorkerDatasetSource:
    """Worker-local source wrapper used by the mixed training iterator."""

    def __init__(
        self,
        iterator_factory: IteratorFactory,
        target_count: int,
        *,
        dataset_name: str,
        epoch: int,
        max_repeat_passes: int = 0,
        repeat_indefinitely: bool = False,
        sample_transform: DatasetSampleTransform | None = None,
        transform_seed: int | None = None,
        atomic_chunk_size: int = 1,
    ) -> None:
        if max_repeat_passes < 0:
            raise ValueError("max_repeat_passes must be >= 0")
        if repeat_indefinitely and max_repeat_passes:
            raise ValueError(
                "repeat_indefinitely cannot be combined with max_repeat_passes"
            )

        self._iterator_factory = iterator_factory
        self._base_epoch = epoch
        self._iterator = iterator_factory(epoch)
        self.remaining = target_count
        self.target_count = target_count
        self.dataset_name = dataset_name
        self.max_repeat_passes = max_repeat_passes
        self.repeat_indefinitely = repeat_indefinitely
        self.produced_count = 0
        self.repeat_count = 0
        self._current_pass_produced_count = 0
        self.atomic_chunk_size = atomic_chunk_size
        self.sample_transform = sample_transform
        self._transform_rng = (
            random.Random(transform_seed)
            if (
                sample_transform is not None
                and sample_transform.enabled
                and transform_seed is not None
            )
            else None
        )

    def __bool__(self) -> bool:
        return self.remaining > 0

    def _restart_iterator(self, next_epoch: int) -> None:
        self._iterator = self._iterator_factory(next_epoch)
        self._current_pass_produced_count = 0

    def _raise_iterable_repeat_stall(
        self, last_pass_produced_count: int
    ) -> None:
        info = get_shard_info()
        next_epoch = self._base_epoch + self.repeat_count + 1
        raise RuntimeError(
            "Iterable dataset repeat made no progress: "
            f"dataset_name={self.dataset_name} "
            f"rank={info.rank} worker_id={info.worker_id} slot_id={info.slot_id} "
            f"base_epoch={self._base_epoch} next_epoch={next_epoch} "
            f"repeat_count={self.repeat_count} target_count={self.target_count} "
            f"produced_count={self.produced_count} "
            f"last_pass_produced_count={last_pass_produced_count}."
        ) from None

    def _log_repeat_warning(self, *, last_pass_produced_count: int) -> None:
        if self.repeat_count % REPEAT_WARNING_INTERVAL != 0:
            return

        info = get_shard_info()
        next_epoch = self._base_epoch + self.repeat_count
        logger.warning(
            "Dataset repeat warning: dataset_name=%s rank=%d worker_id=%d slot_id=%d "
            "base_epoch=%d next_epoch=%d repeat_count=%d target_count=%d "
            "produced_count=%d last_pass_produced_count=%d",
            self.dataset_name,
            info.rank,
            info.worker_id,
            info.slot_id,
            self._base_epoch,
            next_epoch,
            self.repeat_count,
            self.target_count,
            self.produced_count,
            last_pass_produced_count,
        )

    def next_sample(self) -> Sample:
        if self.remaining <= 0:
            raise StopIteration

        while True:
            try:
                sample = next(self._iterator)
                if (
                    self.sample_transform is not None
                    and self._transform_rng is not None
                ):
                    sample = self.sample_transform.apply(
                        sample, rng=self._transform_rng
                    )
                self.remaining -= 1
                self.produced_count += 1
                self._current_pass_produced_count += 1
                return sample
            except StopIteration:
                last_pass_produced_count = self._current_pass_produced_count

                if self.repeat_indefinitely:
                    if last_pass_produced_count <= 0:
                        self._raise_iterable_repeat_stall(
                            last_pass_produced_count
                        )
                    self.repeat_count += 1
                    self._log_repeat_warning(
                        last_pass_produced_count=last_pass_produced_count,
                    )
                    self._restart_iterator(self._base_epoch + self.repeat_count)
                    continue

                if self.repeat_count < self.max_repeat_passes:
                    self.repeat_count += 1
                    self._restart_iterator(self._base_epoch + self.repeat_count)
                    continue

                info = get_shard_info()
                raise RuntimeError(
                    "Dataset "
                    f"'{self.dataset_name}' exhausted before reaching its worker-local target count "
                    f"on rank={info.rank}, worker_id={info.worker_id}, slot_id={info.slot_id}. "
                    f"Expected {self.target_count} samples, produced {self.produced_count}, "
                    f"repeat_count={self.repeat_count}/{self.max_repeat_passes}."
                ) from None


class TrainDataset(IterableDataset):
    """Finite per-epoch training mixer across multiple adapters."""

    def __init__(
        self,
        config: DatasetConfig,
        preprocessor: Preprocessor | None = None,
        datasets_key: str = "train_datasets",
        seed: int | None = None,
        shuffle_dataset: bool | None = None,
        shuffle_buffer_size: int | None = None,
    ) -> None:
        config_dict = self._load_config(config)
        run_seed = seed if seed is not None else config_dict.get("seed")
        config_dict = seed_encoder_data(config_dict, run_seed)
        self.config = self._require_datasets_key(config_dict, datasets_key)
        self.num_frames: int = self.config.get("num_frames", 0)
        self.seed = run_seed
        self.shuffle_dataset = (
            self.config.get("shuffle_dataset", True)
            if shuffle_dataset is None
            else shuffle_dataset
        )
        self.shuffle_buffer_size = (
            self.config.get("shuffle_buffer_size")
            if shuffle_buffer_size is None
            else shuffle_buffer_size
        )
        self.mix_inbatch_groups_atomically = bool(
            self.config.get("mix_inbatch_groups_atomically", False)
        )
        self._next_epoch = 0

        self.preprocessor = self._build_preprocessor(preprocessor)

        dataset_specs = self.config[datasets_key]
        self.entries = self._load_datasets(dataset_specs)

        self._raw_length = sum(entry.raw_length for entry in self.entries)
        self._effective_length = sum(
            entry.effective_length for entry in self.entries
        )
        world_size = self._world_size()
        self._length = sum(
            rank_shard_length(entry.effective_length, world_size) * world_size
            for entry in self.entries
        )
        logger.info(
            "TrainDataset: %s subsets, raw %s samples, weighted %s samples, after drop_last %s samples",
            len(self.entries),
            f"{self._raw_length:,}",
            f"{self._effective_length:,}",
            f"{self._length:,}",
        )

    @staticmethod
    def _load_config(config: DatasetConfig) -> dict[str, Any]:
        return _load_data_config(config)

    @staticmethod
    def _require_datasets_key(
        config: dict[str, Any],
        datasets_key: str,
    ) -> dict[str, Any]:
        if datasets_key not in config:
            raise ValueError(
                f"data config must define top-level '{datasets_key}'."
            )
        return config

    def _build_preprocessor(
        self,
        preprocessor: Preprocessor | None,
    ) -> Preprocessor | None:
        config_preprocessor = self.config.get("preprocessor")
        if preprocessor is not None:
            if config_preprocessor:
                warnings.warn(
                    "Both preprocessor argument and config['preprocessor'] are provided. "
                    "Using the provided preprocessor argument and ignoring config.",
                    UserWarning,
                    stacklevel=2,
                )
            return preprocessor

        if config_preprocessor:
            logger.info(
                "Creating preprocessor from config: %s", config_preprocessor
            )
            return create_preprocessor(config_preprocessor)

        warnings.warn(
            "No preprocessor provided: either pass preprocessor argument or include "
            "'preprocessor' in config",
            UserWarning,
            stacklevel=2,
        )
        return None

    def _load_datasets(
        self, dataset_specs: list[dict[str, Any]]
    ) -> list[DatasetEntry]:
        paired_dataset_ids = {
            "tworoom_paired_hdf5",
            "tworoom_lang_select_hdf5",
            "tworoom_native_gaussian_multitask_hdf5",
        }
        selected_ids = {
            str(spec.get("id", "")).strip() for spec in dataset_specs
        }
        if selected_ids & paired_dataset_ids and not selected_ids <= paired_dataset_ids:
            raise ValueError(
                "Paired vision datasets currently must be used as the only training dataset "
                "type in a run; mixed paired/non-paired batches are not supported."
            )

        global_data_config = self.config
        if self.shuffle_buffer_size is not None:
            global_data_config = dict(self.config)
            global_data_config["shuffle_buffer_size"] = self.shuffle_buffer_size
        entries: list[DatasetEntry] = []
        for index, spec in enumerate(dataset_specs):
            resolved_spec = resolve_dataset_spec(spec, global_data_config)
            sample_transform = DatasetSampleTransform.from_spec(
                resolved_spec.spec
            )
            dataset = create_adapter(resolved_spec)
            if hasattr(dataset, "shuffle"):
                dataset.shuffle = self.shuffle_dataset
            raw_length = self._resolve_dataset_length(
                dataset, resolved_spec.dataset_id
            )
            weighted_length = effective_dataset_length(
                raw_length, resolved_spec.weight
            )
            entry = DatasetEntry(
                resolved_spec=resolved_spec,
                dataset=dataset,
                sample_transform=sample_transform,
                raw_length=raw_length,
                effective_length=weighted_length,
            )
            entries.append(entry)

            logger.info(
                "Loaded subset %s/%s: '%s' (raw=%s, weight=%s, effective=%s)",
                index + 1,
                len(dataset_specs),
                resolved_spec.dataset_id,
                f"{raw_length:,}",
                resolved_spec.weight,
                f"{weighted_length:,}",
            )
            if sample_transform.enabled:
                logger.info(
                    "Dataset '%s': enabled sample transforms (p_sample_sentence_chunk=%.3f)",
                    resolved_spec.dataset_id,
                    sample_transform.p_sample_sentence_chunk,
                )
            if weighted_length <= 0:
                logger.info(
                    "Dataset '%s' has effective length 0 after weighting and will be skipped during mixing",
                    resolved_spec.dataset_id,
                )
        return entries

    @staticmethod
    def _resolve_dataset_length(dataset: ChildDataset, dataset_id: str) -> int:
        try:
            return len(dataset)
        except TypeError:
            length = int(getattr(dataset, "DEFAULT_LENGTH", 1_000_000))
            logger.warning(
                "Dataset '%s' has no __len__, using default length=%s",
                dataset_id,
                length,
            )
            return length

    @staticmethod
    def _world_size() -> int:
        return dist.get_world_size() if dist.is_initialized() else 1

    def _rank_length(self) -> int:
        world_size = self._world_size()
        return sum(
            rank_shard_length(entry.effective_length, world_size)
            for entry in self.entries
        )

    def _set_child_epoch(self, dataset: ChildDataset, epoch: int) -> None:
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        else:
            dataset._epoch = epoch

    def _child_iterator_factory(
        self,
        entry: DatasetEntry,
        *,
        dataset_index: int,
    ) -> IteratorFactory:
        dataset = entry.dataset
        if isinstance(dataset, IterableDataset):

            def iterator_factory(source_epoch: int) -> Iterator[Sample]:
                self._set_child_epoch(dataset, source_epoch)
                return iter(dataset)

            return iterator_factory

        def iterator_factory(source_epoch: int) -> Iterator[Sample]:
            view = ShardedMapDatasetView(
                dataset,
                seed=self.seed + dataset_index * 10_007,
                epoch=source_epoch,
                shuffle=self.shuffle_dataset,
                split_inbatch_groups_across_ranks=bool(
                    self.config.get(
                        "split_inbatch_groups_across_ranks",
                        False,
                    )
                ),
            )
            return iter(view)

        return iterator_factory

    def _build_source(
        self,
        entry: DatasetEntry,
        *,
        dataset_index: int,
        epoch: int,
    ) -> WorkerDatasetSource | None:
        info = get_shard_info()
        group_size = int(getattr(entry.dataset, "inbatch_group_size", 1))
        base_length = int(getattr(entry.dataset, "base_length", 0))
        split_groups = bool(
            self.config.get("split_inbatch_groups_across_ranks", False)
        )
        if (
            split_groups
            and info.world_size > 1
            and group_size > 1
            and base_length > 0
            and entry.resolved_spec.weight == 1.0
        ):
            if group_size % info.world_size != 0:
                raise ValueError(
                    "inbatch_group_size must be divisible by world_size "
                    "when splitting groups across ranks: "
                    f"group_size={group_size}, world_size={info.world_size}"
                )
            worker_groups = len(
                range(info.worker_id, base_length, info.num_workers)
            )
            rank_slots = len(range(info.rank, group_size, info.world_size))
            target_count = worker_groups * rank_slots
        elif group_size > 1 and base_length > 0 and entry.resolved_spec.weight == 1.0:
            rank_groups = rank_shard_length(base_length, info.world_size)
            target_count = (
                len(range(info.worker_id, rank_groups, info.num_workers))
                * group_size
            )
        else:
            target_count = worker_target_count(entry.effective_length)
        if target_count <= 0:
            return None

        atomic_chunk_size = (
            len(range(info.rank, group_size, info.world_size))
            if self.mix_inbatch_groups_atomically else 1
        )

        if group_size > 1 and base_length > 0 and entry.resolved_spec.weight == 1.0:
            pass_count = target_count
        else:
            pass_count = worker_target_count(entry.raw_length)
        if pass_count <= 0:
            raise RuntimeError(
                "Dataset "
                f"'{entry.resolved_spec.dataset_id}' has worker-local pass size 0 "
                f"but weighted target count {target_count} on rank={info.rank}, "
                f"worker_id={info.worker_id}, slot_id={info.slot_id}. "
                "Current drop_last sharding cannot realize this weighted mix."
            )

        iterator_factory = self._child_iterator_factory(
            entry,
            dataset_index=dataset_index,
        )
        repeat_indefinitely = isinstance(entry.dataset, IterableDataset)
        max_repeat_passes = 0
        if not repeat_indefinitely:
            max_repeat_passes = required_overdraw_passes(
                target_count, pass_count
            )
        return WorkerDatasetSource(
            iterator_factory,
            target_count,
            dataset_name=entry.resolved_spec.dataset_id,
            epoch=epoch,
            max_repeat_passes=max_repeat_passes,
            repeat_indefinitely=repeat_indefinitely,
            sample_transform=entry.sample_transform,
            transform_seed=(
                self.seed
                + epoch * 1_000_003
                + dataset_index * 10_007
                + info.slot_id * 1_000_000_007
            ),
            atomic_chunk_size=atomic_chunk_size,
        )

    def _build_sources(self, epoch: int) -> list[WorkerDatasetSource]:
        sources: list[WorkerDatasetSource] = []
        for index, entry in enumerate(self.entries):
            source = self._build_source(entry, dataset_index=index, epoch=epoch)
            if source is not None:
                sources.append(source)
        return sources

    def _choose_source_index(
        self,
        sources: list[WorkerDatasetSource],
        mix_rng: random.Random,
    ) -> int:
        if not sources:
            return -1
        if not self.shuffle_dataset:
            return 0

        total_remaining = sum(source.remaining for source in sources)
        if total_remaining <= 0:
            return -1

        chosen = mix_rng.random() * total_remaining
        cumulative = 0.0
        for index, source in enumerate(sources):
            cumulative += source.remaining
            if chosen < cumulative:
                return index
        return len(sources) - 1

    def _ensure_num_frames(self, sample: Sample) -> Sample:
        if len(sample.frames) <= self.num_frames:
            return sample

        indices = torch.linspace(
            0,
            len(sample.frames) - 1,
            self.num_frames,
        ).long()
        return Sample(
            frames=[sample.frames[i] for i in indices],
            label_map=sample.label_map[indices],
            targets=sample.targets,
            meta=sample.meta,
            target_frames=(
                [sample.target_frames[i] for i in indices]
                if sample.target_frames is not None
                else None
            ),
            target_label_map=(
                sample.target_label_map[indices]
                if sample.target_label_map is not None
                else None
            ),
            query=sample.query,
            text_input=sample.text_input,
        )

    def _process_sample(self, sample: Sample) -> PreprocessedSample:
        if sample.text_input is not None:
            return PreprocessedSample(
                vision_input=None,
                targets=sample.targets,
                query=[sample.query or ""],
                text_input=sample.text_input,
                meta=sample.meta,
            )

        sample = self._ensure_num_frames(sample)
        if self.preprocessor is None:
            return sample
        return self.preprocessor.process_sample(sample)

    def __len__(self) -> int:
        rank_length = self._rank_length()
        self._length = rank_length * self._world_size()
        return rank_length

    @property
    def pixel_shape(self) -> tuple[int, int, int, int]:
        height = width = self.config["preprocessor"]["resolution"]
        return (self.num_frames, 3, height, width)

    def __iter__(self) -> Iterator[PreprocessedSample]:
        epoch = self._next_epoch
        self._next_epoch += 1

        info = get_shard_info()
        mix_seed = self.seed + epoch * 1_000_003
        if not self.mix_inbatch_groups_atomically:
            mix_seed += info.slot_id
        mix_rng = random.Random(mix_seed)
        sources = self._build_sources(epoch)

        while sources:
            source_index = self._choose_source_index(sources, mix_rng)
            if source_index < 0:
                break

            source = sources[source_index]
            for _ in range(source.atomic_chunk_size):
                try:
                    sample = source.next_sample()
                except StopIteration:
                    sources.pop(source_index)
                    break
                yield self._process_sample(sample)
            if source.remaining <= 0:
                sources.pop(source_index)


__all__ = [
    "DatasetEntry",
    "ShardedMapDatasetView",
    "TrainDataset",
    "WorkerDatasetSource",
    "effective_dataset_length",
    "rank_shard_length",
    "required_overdraw_passes",
    "worker_length_from_rank_length",
    "worker_target_count",
]
