# 本仓库的 Stfpm 是 LTD-Net

运行入口：`lightning_model.py` → `SPA.py` + `loss_gra_spa.py`。
这里保留 `Stfpm` 类名以维持现有 checkpoint 的兼容性。

- `SPA.py`：ResNet18 教师、门控适配器、ResNet10 学生、FPN 和方向门控注意力。
- `loss_gra_spa.py`：Top-K、Sobel 与保留损失。
- `anomaly_map.py`：官方多尺度特征差异图计算，逐层相乘。
- `torch_model.py`：保留的官方 STFPM 架构参考，不是活动训练入口。

训练和复现说明请阅读 [项目 README](../../../../../README.md)。
原有 `loss.py` 含未接入当前入口的 FFT 实验，未纳入发布版。
