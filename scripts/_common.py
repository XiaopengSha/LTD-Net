"""Shared release configuration, dataset and checkpoint utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# Always use the bundled modified anomalib, including before editable installation.
sys.path.insert(0, str(ROOT / "src"))


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", type=Path, default=ROOT / "configs/ltd_net.yaml")
    result.add_argument("--data-root", type=Path)
    result.add_argument("--category")
    result.add_argument("--output", type=Path)
    result.add_argument("--accelerator", choices=["auto", "cpu", "gpu"])
    return result


def read_config(args) -> dict:
    import yaml

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"seed", "data", "model", "trainer"}:
        raise ValueError("Config must contain seed, data, model and trainer")
    for key in ("data", "model", "trainer"):
        if not isinstance(config[key], dict):
            raise ValueError(f"{key} must be a mapping")
    for arg, section, key in [
        ("data_root", "data", "root"), ("category", "data", "category"),
        ("output", "trainer", "output"), ("accelerator", "trainer", "accelerator"),
        ("epochs", "trainer", "max_epochs"),
    ]:
        value = getattr(args, arg, None)
        if value is not None:
            config[section][key] = str(value) if isinstance(value, Path) else value
    if config["trainer"]["check_val_every_n_epoch"] > config["trainer"]["max_epochs"]:
        raise ValueError("max_epochs must allow at least one validation pass")
    return config


def make_data(config):
    from anomalib.data import MVTecAD

    options = dict(config["data"])
    category = Path(options["root"]) / options["category"]
    for subdir in ("train/good", "test/good", "ground_truth"):
        if not (category / subdir).is_dir():
            raise FileNotFoundError(f"Missing {category / subdir}; prepare MVTec AD locally first")
    return MVTecAD(**options, seed=config["seed"])


def make_evaluator():
    from anomalib.metrics import AUPRO, AUROC, Evaluator, F1Score

    def auc_metrics():
        return [
            AUROC(fields=["pred_score", "gt_label"], prefix="image_"),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
        ]
    # Distinct metric instances prevent validation/test state sharing.
    return Evaluator(
        val_metrics=auc_metrics(),
        test_metrics=auc_metrics() + [
            AUPRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_", fpr_limit=0.3),
            F1Score(fields=["pred_label", "gt_label"], prefix="image_"),
            F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_"),
        ],
    )


def load_checkpoint(path, *, evaluator=None):
    from anomalib.models import Stfpm

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    # All teacher parameters come from the checkpoint; no ImageNet download needed.
    # Lightning checkpoints contain saved preprocessing/evaluator Python objects.
    # Accept only trusted checkpoints. Preserve all modules during strict loading,
    # including postprocessing statistics even when raw inference only uses .model.
    model = Stfpm.load_from_checkpoint(
        str(path), map_location="cpu", strict=True, teacher_pretrained=False,
        weights_only=False, visualizer=False,
    )
    if evaluator is not None:
        model.evaluator = evaluator
    return model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def image_tensor(path, size):
    """RGB -> float [0,1] -> antialiased resize -> ImageNet normalization."""
    from PIL import Image
    import torch
    from torchvision.transforms.v2 import Compose, Normalize, Resize, ToDtype, ToImage

    transform = Compose([
        ToImage(), ToDtype(torch.float32, scale=True), Resize(size, antialias=True),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    with Image.open(path) as image:
        return transform(image.convert("RGB")).unsqueeze(0)
