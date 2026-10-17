# Training Configurable Perception with different backbones and your own datasets

Configurable Encoder training supports DINOv2-S, DINOv2-Giant, DINOv3-7B, LingBot-Vision-Giant, and V-JEPA2.1-Gigantic. Choose the backbone independently of the dataset with `--backbone`.

## Use the released data

Download the complete [real-world collection](https://huggingface.co/datasets/YijunShen/ConfigWM/tree/main/realworld) and train DINOv2-Giant on eight GPUs:

```bash
hf download YijunShen/ConfigWM --repo-type dataset \
  --include "realworld/**" --local-dir ../configwm-data

bash launch/train_configurable_encoder.sh \
  --env realworld --gpus 0,1,2,3,4,5,6,7 \
  --data-root ../configwm-data/realworld \
  --backbone facebook/dinov2-giant \
  --predictor-model alpindale/Llama-3.2-1B \
  --output-dir ./outputs/configurable_encoder/realworld/dinov2-giant
```

## Prepare image pairs

Organize your data as follows (see the [example dataset](../examples/realworld/)):

```text
my-data/
  manifest.jsonl
  images/
    scene01_source.png
    scene01_target.png
```

Each line in `manifest.jsonl` describes one pair:

```json
{
  "source": "images/scene01_source.png",
  "target": "images/scene01_target.png",
  "query": "Focus on the red cup.",
  "case_id": "scene01",
  "split": "train"
}
```

- `source`: observed image, including distractors.
- `target`: corresponding image retaining the task-relevant content.
- `query`: task description supplied to the encoder.
- `case_id` (optional): identifies related frames.
- `split` (optional): `train`, `val`, or `test`; defaults to `train`.

Image paths are relative to the data folder.

## Try the format example

Run one training step with the provided image pairs:

```bash
bash launch/train_configurable_encoder.sh \
  --env realworld --gpus 0 \
  --data-root ./examples/realworld \
  --backbone dinov2-small \
  --predictor-model alpindale/Llama-3.2-1B \
  --group-size 6 --steps 1 --output-dir ./outputs/configurable_encoder/example
```

Replace `--data-root` with your own folder and omit `--steps 1` for training. Adjust its settings in `configs/configurable_encoder/trainer_realworld.yaml`.

## Choose a backbone

| `--backbone` | Default input | Spatial patches | Feature dimension |
| --- | --- | --- | --- |
| `dinov2-small` | 224 × 224 | 256 | 384 |
| `dinov2-giant` | 896 × 896 | 4096 | 1536 |
| `dinov3-7b` | 1024 × 1024 | 4096 | 4096 |
| `lingbot-giant` | 1024 × 1024 | 4096 | 1536 |
| `vjepa2.1-gigantic` | 1024 × 1024 | 4096 | 1664 |

For example, train DINOv2-Giant on eight GPUs:

```bash
bash launch/train_configurable_encoder.sh \
  --env realworld --gpus 0,1,2,3,4,5,6,7 \
  --data-root ../my-data --backbone facebook/dinov2-giant \
  --predictor-model alpindale/Llama-3.2-1B \
  --output-dir ./outputs/configurable_encoder/realworld/dinov2-giant
```

LingBot and V-JEPA use their upstream source implementations. Install their extra
dependencies in the same environment, then clone the required source:

```bash
python -m pip install -r requirements/backbones.txt
git clone https://github.com/robbyant/lingbot-vision.git ./third_party/lingbot-vision
git clone https://github.com/facebookresearch/vjepa2.git ./third_party/vjepa2
```

Then select `--backbone lingbot-giant` or `--backbone vjepa2.1-gigantic`.
