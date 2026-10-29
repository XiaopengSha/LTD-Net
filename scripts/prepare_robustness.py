"""Generate independent, reproducible MVTec AD test corruptions (no training).

Requires numpy and Pillow. See robustness_usage.md for the evaluation protocol.
Images must be 8-bit RGB or grayscale, masks binary. Source data is never edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
GEOMETRY = {
    "rotate90": Image.Transpose.ROTATE_90,
    "rotate180": Image.Transpose.ROTATE_180,
    "rotate270": Image.Transpose.ROTATE_270,
    "hflip": Image.Transpose.FLIP_LEFT_RIGHT,
    "vflip": Image.Transpose.FLIP_TOP_BOTTOM,
}
DEFAULT_VARIANTS = ["clean", "noise_0.02", "noise_0.05", "noise_0.10",
                    "brightness_0.6", "brightness_0.8", "brightness_1.2",
                    "brightness_1.4", "rotate90", "rotate180", "rotate270",
                    "hflip", "vflip"]


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None
    relative: Path
    mask_relative: Path | None


def variant_spec(name):
    if name == "clean" or name in GEOMETRY:
        return name, None
    for prefix in ("noise", "brightness"):
        if name.startswith(prefix + "_"):
            value = float(name[len(prefix) + 1:])
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Perturbation strength must be finite and positive")
            if prefix == "noise" and value > 1:
                raise ValueError("Noise sigma uses [0, 1] intensity units, not [0, 255]")
            # Reject names that could form paths or nonportable Windows filenames.
            if any(c not in "0123456789.eE+-" for c in name[len(prefix) + 1:]):
                raise ValueError(f"Invalid variant name: {name}")
            return prefix, value
    raise ValueError(f"Unknown variant: {name}")


def file_seed(seed, relative):
    """Stable across Python sessions and ordering; paired noise across severities."""
    key = f"{seed}:{relative.as_posix()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def transform(image, mask, variant, seed):
    family, value = variant_spec(variant)
    if family in GEOMETRY:
        op = GEOMETRY[family]
        return image.transpose(op), None if mask is None else mask.transpose(op)
    if family == "clean":
        return image.copy(), None if mask is None else mask.copy()
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    if family == "noise":
        noise = np.random.default_rng(seed).standard_normal(pixels.shape).astype(np.float32)
        pixels = pixels + value * noise
    else:
        pixels = pixels * value
    result = np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(result), None if mask is None else mask.copy()


def inspect_dataset(root, categories=None):
    if not root.is_dir():
        raise ValueError(f"Dataset root not found: {root}")
    available = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "test").is_dir())
    selected = available if categories is None else list(dict.fromkeys(categories))
    if not selected:
        raise ValueError("No <category>/test folders. --root must be the MVTec AD dataset root")
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"Unknown categories: {sorted(unknown)}")
    samples, destinations = [], set()
    for category in selected:
        files = sorted(p for p in (root / category / "test").rglob("*")
                       if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"Empty category: {category}")
        for path in files:
            relative = path.relative_to(root)
            if len(relative.parts) != 4:
                raise ValueError(f"Expected category/test/defect/image: {relative}")
            defect = relative.parts[2]
            output_rel = relative.with_suffix(".png")
            key = output_rel.as_posix().lower()
            if key in destinations:
                raise ValueError(f"Output filename collision: {output_rel}")
            destinations.add(key)
            mask_rel = (Path(category) / "ground_truth" / defect / f"{path.stem}_mask.png"
                        if defect != "good" else None)
            mask_path = None if mask_rel is None else root / mask_rel
            with Image.open(path) as image:
                image.load()
                if image.mode not in {"RGB", "L"}:
                    raise ValueError(f"Only 8-bit RGB/L images supported: {path} ({image.mode})")
                size = image.size
            if mask_path is not None:
                if not mask_path.is_file():
                    raise ValueError(f"Missing ground-truth mask: {mask_path}")
                with Image.open(mask_path) as mask:
                    if mask.size != size:
                        raise ValueError(f"Image/mask size mismatch: {path}")
                    values = set(np.unique(np.asarray(mask.convert("L"))).tolist())
                    if not values.issubset({0, 1, 255}) or not any(values):
                        raise ValueError(f"Expected nonempty binary anomaly mask: {mask_path}")
            samples.append(Sample(path, mask_path, output_rel, mask_rel))
    return samples, selected


def build(root, output, variants, categories=None, seed=42, copy_train=False, dry_run=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Source/output must be separate, non-nested directories")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be new or empty (no overwrite): {output}")
    variants = list(dict.fromkeys(variants))
    if not variants:
        raise ValueError("Select at least one variant")
    for variant in variants:
        variant_spec(variant)
    samples, categories = inspect_dataset(root, categories)
    if copy_train:
        for category in categories:
            if not (root / category / "train").is_dir():
                raise ValueError(f"Missing train directory: {root / category / 'train'}")
    summary = {"source": str(root), "output": str(output), "seed": seed,
               "categories": categories, "variants": variants,
               "test_images_per_variant": len(samples),
               "anomaly_masks_per_variant": sum(s.mask is not None for s in samples),
               "copy_train": copy_train,
               "settings": {v: {"family": variant_spec(v)[0], "value": variant_spec(v)[1]}
                            for v in variants},
               "protocol": "Independent transforms on original-resolution test images; no composition.",
               "noise": "Gaussian iid, sigma in [0,1] units, clipped; same base noise across severities.",
               "geometry": "Lossless counterclockwise 90-degree multiples; no crop, pad or resize.",
               "brightness": "Multiply stored RGB/grayscale intensity, then clip and round to uint8.",
               "labels": "Inherited; geometry requires category-specific label-invariance review.",
               "status": "planned"}
    print(f"Validated: {len(categories)} categories, {len(samples)} images x {len(variants)} variants", flush=True)
    if dry_run:
        return summary
    output.mkdir(parents=True, exist_ok=True)
    summary["status"] = "incomplete"
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # Read each original once; all transforms start from this same original.
    for index, sample in enumerate(samples, 1):
        with Image.open(sample.image) as handle:
            image = handle.copy()
        mask = None
        if sample.mask is not None:
            with Image.open(sample.mask) as handle:
                mask = Image.fromarray((np.asarray(handle.convert("L")) > 0).astype(np.uint8) * 255)
        for variant in variants:
            result, result_mask = transform(image, mask, variant, file_seed(seed, sample.relative))
            target = output / variant / sample.relative
            target.parent.mkdir(parents=True, exist_ok=True)
            result.save(target)
            if result_mask is not None:
                target_mask = output / variant / sample.mask_relative
                target_mask.parent.mkdir(parents=True, exist_ok=True)
                result_mask.save(target_mask)
        if index % 100 == 0 or index == len(samples):
            print(f"Generated {index}/{len(samples)} source images", flush=True)
    if copy_train:
        for variant in variants:
            for category in categories:
                shutil.copytree(root / category / "train", output / variant / category / "train")
    summary["status"] = "complete"
    manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done: {output}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Root containing category folders")
    parser.add_argument("--output", type=Path, required=True, help="New/empty, non-nested output directory")
    parser.add_argument("--categories", nargs="+", help="Default: all categories")
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS,
                        help="clean, rotate90/180/270, hflip, vflip, noise_SIGMA, brightness_FACTOR")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-train", action="store_true", help="Also copy unchanged train folders (uses extra disk)")
    parser.add_argument("--dry-run", action="store_true", help="Validate source data without writing output")
    args = parser.parse_args()
    try:
        build(args.root, args.output, args.variants, args.categories, args.seed, args.copy_train, args.dry_run)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
