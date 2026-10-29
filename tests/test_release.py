"""Offline checks of the release model, loss, metrics and export contract."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts._common import image_tensor, load_checkpoint, make_evaluator, read_config
from scripts.export_onnx import SPAExportWrapper, export_checked_model, make_check_inputs
from anomalib.models import Stfpm
from anomalib.metrics.aupro import _AUPRO
from anomalib.models.image.stfpm.loss_gra_spa import STFPMLoss

torch.set_num_threads(1)


def new_model():
    torch.manual_seed(42)
    return Stfpm(teacher_pretrained=False, image_size=(32, 32), evaluator=False,
                 visualizer=False, post_processor=False)


def test_teacher_frozen_and_student_adapter_gradients():
    model = new_model().train()
    original, adapted, student = model.model(torch.randn(2, 3, 32, 32))
    loss = model.loss(original, adapted, student)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None for p in model.model.teacher_model.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.student_model.parameters())
    assert model.model.t_adapter1.alpha.grad is not None
    # alpha starts at zero: its residual branch becomes trainable through the gate.
    assert torch.isfinite(model.model.t_adapter1.alpha.grad).all()


def test_loss_zero_on_identical_features_and_validates_weights():
    feature = {"layer1": torch.randn(2, 64, 8, 8)}
    assert STFPMLoss()(feature, feature, feature).item() == 0
    with pytest.raises(ValueError):
        STFPMLoss(mining_ratio=0)


def test_eval_wrapper_matches_and_unsupported_backbone_rejected():
    model = new_model().eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        reference = model.model(x)
        wrapped = SPAExportWrapper(model.model)(x)
    torch.testing.assert_close(reference.anomaly_map, wrapped, rtol=0, atol=0)
    torch.testing.assert_close(reference.pred_score, wrapped.amax(dim=(-2, -1)))
    with pytest.raises(ValueError):
        Stfpm(backbone="resnet50", teacher_pretrained=False, evaluator=False, visualizer=False)


def test_legacy_bn_warmup_accepts_image_batch_without_changing_stats():
    model = new_model()
    teacher = model.model.teacher_model
    before = {key: value.clone() for key, value in teacher.state_dict().items() if "running_" in key}
    model.model.update_teacher_bn_stats([SimpleNamespace(image=torch.randn(2, 3, 32, 32))], "cpu", 1)
    for key, value in before.items():
        torch.testing.assert_close(teacher.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("perfect", [True, False])
def test_aupro_perfect_and_reversed_ranking(perfect):
    mask = torch.zeros(2, 8, 8, dtype=torch.int)
    mask[0, 1:3, 1:3] = 1
    mask[1, 2:5, 2:5] = 1
    prediction = mask.float() if perfect else 1 - mask.float()
    result = _AUPRO(fpr_limit=0.3)(prediction, mask)
    assert float(result) == pytest.approx(1.0 if perfect else 0.0, abs=1e-6)


def test_preprocessing_rgb_and_range(tmp_path):
    from PIL import Image
    path = tmp_path / "white.png"
    Image.new("L", (11, 17), 255).save(path)
    tensor = image_tensor(path, (32, 32))
    assert tensor.shape == (1, 3, 32, 32)
    expected = (torch.ones(3) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
    torch.testing.assert_close(tensor[0, :, 0, 0], expected)


def test_default_config_matches_preserved_sgd():
    path = Path(__file__).resolve().parents[1] / "configs/ltd_net.yaml"
    config = read_config(SimpleNamespace(config=path))
    assert config["model"]["student_lr"] == 0.4
    assert config["model"]["adapter_lr"] == 0.05
    assert config["data"]["val_split_mode"] == "same_as_test"


def test_mvtec_split_is_not_empty(tmp_path):
    from PIL import Image
    from anomalib.data import MVTecAD
    for subdir in ["train/good", "test/good", "test/crack", "ground_truth/crack"]:
        folder = tmp_path / "metal_nut" / subdir
        folder.mkdir(parents=True)
        name = "000_mask.png" if subdir.startswith("ground_truth") else "000.png"
        Image.new("RGB", (32, 32), "white").save(folder / name)
    data = MVTecAD(root=tmp_path, category="metal_nut", val_split_mode="same_as_test", num_workers=0)
    data.setup()
    assert len(data.train_data) == 1
    assert len(data.val_data) == len(data.test_data) == 2


def test_checkpoint_restores_postprocessor_and_strict_model_state(tmp_path):
    import lightning
    model = Stfpm(teacher_pretrained=False, image_size=(32, 32), evaluator=make_evaluator(), visualizer=False)
    path = tmp_path / "model.ckpt"
    checkpoint = {"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                  "pytorch-lightning_version": lightning.__version__}
    torch.save(checkpoint, path)
    restored = load_checkpoint(path)
    assert set(restored.state_dict()) == set(model.state_dict())
    assert restored.post_processor is not None
    assert restored.image_size == (32, 32)
    # Strict loading must refuse a missing network weight.
    del checkpoint["state_dict"]["model.t_adapter1.alpha"]
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_checkpoint(path)


def test_real_onnx_export_and_no_overwrite(tmp_path):
    model = new_model().eval().model
    output = tmp_path / "model.onnx"
    samples = make_check_inputs((32, 32))
    export_checked_model(model, output, samples)
    assert output.is_file()
    with pytest.raises(FileExistsError):
        export_checked_model(model, output, samples)
    # Invalid numpy shape must be rejected before inference.
    invalid = tmp_path / "invalid.npy"
    np.save(invalid, np.zeros((3, 32, 32), dtype=np.float32))
    with pytest.raises(ValueError):
        make_check_inputs((32, 32), [invalid])
