# Building Segmentation

Binary building footprint extraction from high-resolution aerial imagery.
Trained on the [Inria Aerial Image Labeling](https://project.inria.fr/aerialimagelabeling/) dataset.

## EDA

**Dataset composition:** 180 tiles, 5 cities, 36 tiles each.

**Building density and average number of buildings per tile for each city**:

| City    | Tiles | Building px | Avg buildings/tile |
|---------|-------|-------------|--------------------|
| Vienna  | 36    | 28.8%       | 845                |
| Chicago | 36    | 24.3%       | 2233               |
| Austin  | 36    | 15.0%       | 1444               |
| Tyrol-W | 36    | 5.8%        | 465                |
| Kitsap  | 36    | 4.9%        | 666                |

This is why train/val are split by whole image, not by patches - otherwise crops from
the same image could end up in both sets, leaking information between
train and val.

**Class balance.** Across the full dataset, building pixels make up
only ~16% (vs ~84% background). This motivated the *Dice + Focal +
boundary loss* combination over plain BCE, which would be dominated by
the majority class.

**Building size.** Median building area is ~1130 px², with the largest
legitimate structures reaching 500k px². The `min_area=40` filter in postprocessing
sits well below the median, removing noise without dropping real small
structures.

## Results

Five architectures were compared under identical training conditions (same loss, optimizer, spatial split).

| Model                 | Params    | Dataset IoU | F1        | Precision | Recall    |
|-----------------------|-----------|-------------|-----------|-----------|-----------|
| DeepLabV3+ ResNet-34  | 22.4M     | 0.801       | 0.890     | 0.877     | 0.902     |
| UNet++ ResNet-34      | 26.1M     | 0.804       | 0.891     | 0.879     | 0.903     |
| SegFormer MiT-B1      | 13.7M     | 0.812       | 0.896     | 0.885     | 0.908     |
| SegFormer MiT-B3      | 44.6M     | 0.832       | 0.908     | 0.904     | 0.913     |
| **SegFormer MiT-B5**  | **82.0M** | **0.835**   | **0.910** | **0.908** | **0.913** |

All metrics are dataset-level (micro) IoU computed on a held-out spatial validation split (20% of training tiles, grouped by source image to prevent leakage). MiT-B3 offers the best accuracy/size trade-off — 99.6% of MiT-B5 quality at 54% of its parameter count.

## Project structure

```
├── train.py                      # trains a model 
├── configs/                      # YAML configs
├── src/
│   ├── dataset.py                # loads tiles, cuts them into 512x512 patches
│   ├── model.py                  # wraps smp architectures into a single class
│   ├── losses.py                 # dice + focal + boundary loss
│   ├── postprocess.py            # cleans up masks and extracts building polygons
│   ├── inference.py              # runs model on full 5000x5000 tiles seamlessly
│   └── utils.py                  # seed, IoU metric, spatial train/val split
├── scripts/
│   ├── evaluate.py               # measures IoU/F1 on the validation split
│   ├── visualize.py              # saves prediction images for inspection
│   └── vectorize_to_geojson.py   # converts predicted masks to GeoJSON polygons
├── tools/
│   ├── check_models.py           # verifies all configs load
│   └── extract_pth.py            # reads metrics saved inside a checkpoint
├── Dockerfile                    # GPU image for training and inference on Linux
├── Dockerfile.smoke              # lightweight CPU image to verify the build
├── docker-compose.yml
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Download the [Inria dataset](https://project.inria.fr/aerialimagelabeling/download/) and place it at:
```
data/inria/train/images/
data/inria/train/gt/
data/inria/test/images/
```

## Usage

**Training**
```bash
python train.py --config configs/...
python train.py --config configs/... --resume checkpoints/...
```

**Evaluation**
```bash
python -m scripts.evaluate --config configs/... --checkpoint checkpoints/...
```

**Visualization**
```bash
python -m scripts.visualize --mode predict --config configs/... --checkpoint checkpoints/...
python -m scripts.visualize --mode pipeline --config configs/... --checkpoint checkpoints/...
python -m scripts.visualize --mode fulltile --config configs/... --checkpoint checkpoints/... --tile data/inria/train/images/... --gt data/inria/train/gt/...
```

**GeoJSON export**
```bash
python -m scripts.vectorize_to_geojson --mask predictions/... --source data/inria/train/images/... --out vectors/...
```

## Docker

```bash
# smoke test (CPU)
docker build -f Dockerfile.smoke -t building-seg-smoke .
docker run --rm building-seg-smoke

# full GPU image (Linux + nvidia-container-toolkit)
docker compose build
docker compose run --rm building-seg python train.py --config configs/...
```

## Streamlit demo

```bash
streamlit run app.py
```
