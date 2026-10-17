"""Object-centric augmentation utilities for stable-worldmodel."""

from .collect_augment import CollectAugmentConfig, collect_and_augment_dataset
from .hdf5_static_copy import (
    FanoutObjectCopyConfig,
    ObjectCopyConfig,
    augment_hdf5_dataset,
    augment_hdf5_dataset_fanout,
    repair_hdf5_dataset_fanout_failures,
)
from .registry import get_object_spec
from .tworoom_dynamic_toy import (
    TwoRoomDynamicToyConfig,
    run_tworoom_dynamic_donor_toy,
)
from .tworoom_dynamic_from_cache import (
    TwoRoomDynamicAugmentConfig,
    TwoRoomDynamicFanoutConfig,
    augment_tworoom_dynamic_dataset_fanout,
    build_dynamic_preview_set,
    build_d0_d3_preview_set,
    compose_episode_with_dynamic_donors,
    load_successful_donor_entries,
    select_donor_entries_for_target,
)

__all__ = [
    'CollectAugmentConfig',
    'FanoutObjectCopyConfig',
    'ObjectCopyConfig',
    'collect_and_augment_dataset',
    'augment_hdf5_dataset',
    'augment_hdf5_dataset_fanout',
    'repair_hdf5_dataset_fanout_failures',
    'get_object_spec',
    'TwoRoomDynamicToyConfig',
    'run_tworoom_dynamic_donor_toy',
    'TwoRoomDynamicAugmentConfig',
    'TwoRoomDynamicFanoutConfig',
    'augment_tworoom_dynamic_dataset_fanout',
    'build_dynamic_preview_set',
    'build_d0_d3_preview_set',
    'compose_episode_with_dynamic_donors',
    'load_successful_donor_entries',
    'select_donor_entries_for_target',
]
