# 依赖与环境约定

项目以根目录 `pyproject.toml` 为依赖来源。先安装匹配的 PyTorch / torchvision，
再执行 `python -m pip install -e ".[ltd,export,verify]"`。
`ltd` 固定 PyTorch 2.8.0 / torchvision 0.23.0，约束 Lightning 2.6.x 和 torchmetrics 1.8.x。
`export` 提供 ONNX 与 ONNX Runtime，`verify` 提供 pytest。

完整安装命令见根目录 README。不要在同一个环境再安装 PyPI 上的另一份 anomalib。
保留的官方可选模型有各自额外依赖；本项目验证范围是 LTD-Net 主流程。
验证环境的实测版本及局限见 `docs/VALIDATION.md`；这些版本不代表原论文训练环境。

基础依赖约束 `pandas>=2.2,<3`：实际训练检查发现 pandas 3 的字符串类型与本地 Split 枚举比较不兼容，会产生空数据集。此处固定兼容范围，不改写上游数据管线。
