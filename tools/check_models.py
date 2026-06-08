"""
tools/check_models.py

Instantiates each model from its config
and prints parameter counts and top-level structure.

Usage:
    python -m tools.check_models
"""
import yaml
from src.model import BuildingSegmentationModel


def check_config(config_path):
    print(f"\n{config_path}")
    print("-" * 50)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    arch = cfg["model"]["architecture"]
    encoder = cfg["model"]["encoder_name"]

    model = BuildingSegmentationModel(
        architecture=arch,
        encoder_name=encoder,
        encoder_weights=None,
    )

    total = sum(p.numel() for p in model.parameters())
    modules = list(model.named_children())

    print(f"  architecture: {arch}")
    print(f"  encoder: {encoder}")
    print(f"  params: {total:,}")
    print(f"  first block: {modules[0][0]}  ({type(modules[0][1]).__name__})")
    print(f"  last block: {modules[-1][0]}  ({type(modules[-1][1]).__name__})")


if __name__ == "__main__":
    configs = [
        "configs/phase1_segformer.yaml",
        "configs/phase1_unetplusplus.yaml",
        "configs/phase1_deeplabv3plus.yaml",
        "configs/phase2_segformer_mitb3.yaml",
        "configs/phase2_segformer_mitb5.yaml",
    ]
    for path in configs:
        try:
            check_config(path)
        except Exception as e:
            print(f"error: {e}")
