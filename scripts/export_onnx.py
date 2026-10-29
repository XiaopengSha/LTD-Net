# SPDX-License-Identifier: Apache-2.0
# Adapted from export_onnx2.py: release paths and offline checkpoint loading.
"""Export SPA-STFPM with unchanged inference semantics and static input shape.

Input: preprocessed float32 [1, 3, H, W]. Output: raw anomaly_map [1, 1, H, W].
Image loading, resize, normalization and thresholding remain outside the graph.
"""

import argparse
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SPAExportWrapper(nn.Module):
    """Get features in eval mode, bypassing SPA.forward's broken ONNX branch."""

    def __init__(self, trained_model: nn.Module):
        super().__init__()
        if getattr(trained_model, "tiler", None) is not None:
            raise ValueError("Tiled inference is not supported by this static exporter.")
        if not trained_model.layers or not set(trained_model.layers) <= {"layer1", "layer2", "layer3"}:
            raise ValueError("This exporter supports SPA layer1/layer2/layer3 only.")
        self.model = trained_model.cpu().float().eval()
        self.model.requires_grad_(False)
        self.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        teacher_features = self.model.teacher_model(images)
        adapters = {
            "layer1": self.model.t_adapter1,
            "layer2": self.model.t_adapter2,
            "layer3": self.model.t_adapter3,
        }
        adapted_features = {
            name: adapters[name](feature) for name, feature in teacher_features.items()
        }
        student_features = self.model.student_model(images)
        # Original formula: 0.5 * squared L2 distance, bilinear resize,
        # then MULTIPLY across scales. Do not reimplement it as a sum.
        return self.model.anomaly_map_generator(
            teacher_features=adapted_features,
            student_features=student_features,
            image_size=images.shape[-2:],
        )


def make_check_inputs(image_size, input_npy_paths=()):
    """Add optional preprocessed real images to deterministic smoke-test inputs."""
    shape = (1, 3, *image_size)
    generator = torch.Generator().manual_seed(42)
    samples = [
        ("random_seed_42", torch.randn(shape, generator=generator)),
        ("zeros", torch.zeros(shape)),
    ]
    for path in input_npy_paths:
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != shape:
            raise ValueError(f"{path}: expected float32 {shape}, got {array.dtype} {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{path}: input contains NaN or Inf")
        samples.append((str(path), torch.from_numpy(np.ascontiguousarray(array))))
    return samples


def compare_outputs(reference, actual, label):
    """Reject shape changes, non-finite results and excessive numerical error."""
    if reference.shape != actual.shape:
        raise AssertionError(f"{label}: shape mismatch: {reference.shape} vs {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise AssertionError(f"{label}: output contains NaN or Inf")
    error = np.abs(reference - actual)
    # Smoke-test tolerances, NOT acceptance criteria for detection accuracy.
    np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-6, err_msg=label)
    score_error = np.abs(reference.max(axis=(-2, -1)) - actual.max(axis=(-2, -1))).max()
    print(
        f"[PASS] {label}: max_abs={error.max():.3e}, "
        f"mean_abs={error.mean():.3e}, score_abs={score_error:.3e}"
    )


def export_checked_model(model, output_path, samples, overwrite=False):
    """Compare against original eval inference, export, then verify with ORT."""
    import onnx
    import onnxruntime as ort

    output_path = Path(output_path).resolve()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Use a new name or --overwrite.")
    if not samples:
        raise ValueError("At least one verification input is required.")
    input_shape = tuple(samples[0][1].shape)
    if len(input_shape) != 4 or input_shape[:2] != (1, 3):
        raise ValueError("Input must have shape [1, 3, height, width].")
    for name, sample in samples:
        if tuple(sample.shape) != input_shape or sample.dtype != torch.float32:
            raise ValueError(f"{name}: all inputs must have the same float32 shape.")

    wrapper = SPAExportWrapper(model)
    reference_outputs = []
    with torch.no_grad():
        for name, sample in samples:
            sample = sample.cpu()
            # Outside ONNX export: this is the original model's eval path.
            reference = model(sample).anomaly_map.cpu().numpy()
            actual = wrapper(sample).cpu().numpy()
            if reference.shape != (1, 1, *input_shape[-2:]):
                raise AssertionError(f"Unexpected anomaly_map shape: {reference.shape}")
            compare_outputs(reference, actual, f"PyTorch vs wrapper / {name}")
            reference_outputs.append(reference)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a verified file; failure leaves any previous model untouched.
    with tempfile.TemporaryDirectory(prefix="onnx_check_", dir=output_path.parent) as temp_dir:
        temporary_model = Path(temp_dir) / output_path.name
        export_options = {}
        # Pin the legacy exporter where available; its default changed in
        # newer PyTorch. Earlier versions have no dynamo parameter.
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_options["dynamo"] = False
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                samples[0][1].cpu(),
                str(temporary_model),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["anomaly_map"],
                # No dynamic_axes: batch and spatial dimensions stay fixed.
                **export_options,
            )

        # Inspect before loading external tensors to detect multi-file models.
        graph = onnx.load(str(temporary_model), load_external_data=False)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
            raise ValueError("External ONNX tensor data needs a separate packaging workflow.")
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(temporary_model), providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != list(input_shape):
            raise AssertionError(f"ONNX input is not fixed: {session.get_inputs()[0].shape}")
        for (name, sample), reference in zip(samples, reference_outputs):
            actual = session.run(["anomaly_map"], {"input": sample.cpu().numpy()})[0]
            compare_outputs(reference, actual, f"PyTorch vs ONNX Runtime / {name}")
        temporary_model.replace(output_path)

    print(f"[SUCCESS] Verified ONNX: {output_path}")
    print(f"[INFO] input: float32 {input_shape}; output: anomaly_map [1, 1, H, W]")
    print("[INFO] Preprocessing/thresholding are external. RKNN compatibility is not yet verified.")
    return output_path


def export_to_onnx(checkpoint_path, output_onnx_path, image_size=(256, 256), *, input_npy_paths=(), overwrite=False):
    if sys.version_info < (3, 10):
        raise RuntimeError("This anomalib repository requires Python >= 3.10. Use the original training environment.")
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if len(image_size) != 2 or any(size <= 0 or size % 16 for size in image_size):
        raise ValueError("For this FPN baseline, height/width must be positive multiples of 16.")
    # Use this repository's SPA implementation, rather than an installed
    # upstream anomalib model that happens to share the Stfpm class name.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from anomalib.models import Stfpm

    samples = make_check_inputs(image_size, input_npy_paths)
    print(f"[INFO] Loading checkpoint with the CURRENT repository architecture: {checkpoint_path}")
    # Lightning metadata can contain Python modules: load only trusted checkpoints.
    lightning_model = Stfpm.load_from_checkpoint(
        str(checkpoint_path), map_location="cpu", strict=True,
        teacher_pretrained=False, weights_only=False,
    )
    if tuple(lightning_model.image_size) != tuple(image_size):
        raise ValueError(f"Checkpoint image_size={lightning_model.image_size} differs from export size={image_size}")
    result = export_checked_model(lightning_model.model, output_onnx_path, samples, overwrite=overwrite)
    if not input_npy_paths:
        print("[NOTE] Synthetic inputs only. Check real normal/defective images before paper evaluation.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint matching the current SPA architecture")
    parser.add_argument("--output", default="outputs/export/ltd_net.onnx", help="Verified static ONNX output")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--input-npy", action="append", default=[], help="Preprocessed float32 [1,3,H,W] .npy; repeatable")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output after verification")
    args = parser.parse_args()
    export_to_onnx(
        args.checkpoint,
        args.output,
        image_size=(args.height, args.width),
        input_npy_paths=args.input_npy,
        overwrite=args.overwrite,
    )
