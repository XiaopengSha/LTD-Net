"""Run checkpoint or ONNX inference, saving raw maps and display-only heatmaps."""

import argparse
from pathlib import Path
from scripts._common import image_tensor, load_checkpoint, write_json


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    backend = cli.add_mutually_exclusive_group(required=True)
    backend.add_argument("--checkpoint", type=Path)
    backend.add_argument("--onnx", type=Path)
    cli.add_argument("--input", required=True, type=Path, help="Image or directory (recursive)")
    cli.add_argument("--output", type=Path, default=Path("outputs/inference"))
    cli.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Checkpoint backend only")
    cli.add_argument("--save-input", action="store_true", help="Save preprocessed NCHW inputs for export verification")
    args = cli.parse_args()
    import numpy as np
    import torch
    from PIL import Image

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if args.input.is_file():
        files, base = [args.input], args.input.parent
    elif args.input.is_dir():
        files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in extensions and p.is_file())
        base = args.input
    else:
        cli.error(f"Input does not exist: {args.input}")
    if not files:
        cli.error("No images found")
    if args.output.exists() and any(args.output.iterdir()):
        cli.error("Output directory is not empty; choose a new --output")

    if args.checkpoint:
        model = load_checkpoint(args.checkpoint).eval().to(args.device)
        size = tuple(model.image_size)
        def predict(tensor):
            with torch.inference_mode():
                return model.model(tensor.to(args.device)).anomaly_map.cpu().numpy()
    else:
        if args.device != "cpu":
            cli.error("This ONNX entry uses CPUExecutionProvider; omit --device cuda")
        import onnxruntime as ort
        session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
        shape = session.get_inputs()[0].shape
        if len(shape) != 4 or shape[:2] != [1, 3] or not all(isinstance(x, int) and x > 0 for x in shape):
            raise ValueError("Expected a fixed-shape float32 [1,3,H,W] ONNX model")
        size = tuple(shape[2:])
        def predict(tensor):
            return session.run(["anomaly_map"], {session.get_inputs()[0].name: tensor.numpy()})[0]

    records = []
    for path in files:
        tensor = image_tensor(path, size)
        result = predict(tensor)
        if result.shape != (1, 1, *size) or not np.isfinite(result).all():
            raise ValueError(f"Invalid anomaly map for {path}")
        raw = result[0, 0].astype(np.float32)
        relative = path.relative_to(base)
        target = args.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(target) + ".anomaly.npy", raw, allow_pickle=False)
        # Display-only scaling never changes the raw anomaly map or score.
        scaled = (raw - raw.min()) / max(float(np.ptp(raw)), 1e-12)
        Image.fromarray((scaled * 255).astype(np.uint8)).save(str(target) + ".heatmap.png")
        if args.save_input:
            np.save(str(target) + ".input.npy", tensor.numpy(), allow_pickle=False)
        records.append({"image": relative.as_posix(), "raw_score": float(raw.max())})
    write_json(args.output / "scores.json", records)
    print(f"Saved {len(records)} raw anomaly maps. No calibrated pass/fail threshold applied.")


if __name__ == "__main__":
    main()
