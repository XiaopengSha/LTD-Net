# LTD-Net 工业表面异常检测

LTD-Net 面向仅使用正常样本训练的工业图像异常检测与定位。项目在
[anomalib](https://github.com/open-edge-platform/anomalib) 的 STFPM 实现上进行修改，
保留其数据加载、训练引擎、评估和预处理组件。

本发布目录提供训练、评估、PyTorch 推理和固定尺寸 ONNX 导出。
模型仍通过 `anomalib.models.Stfpm` 导入，该名称在本仓库中指向修改后的 LTD-Net，
不能把这里的 `Stfpm` 当作未修改的官方 STPM 基线。

**当前状态：代码整理与验证版本。** 不附带训练权重、数据集或论文原稿。
未在本次整理中复现论文精度、FPS 或 Jetson 实验；测试范围见 [验证记录](docs/VALIDATION.md)。
论文训练参数与代码差异见 [复现说明](docs/REPRODUCIBILITY.md)。

## 主要改动

- 教师采用 ImageNet 预训练 ResNet18 的前三个阶段，骨干参数冻结。
- 三个可训练门控适配器对教师特征进行领域适配；代码名为 SPA，论文称 LFA。
- 学生采用裁去末段和分类头的 ResNet10 结构，通过跨层特征融合及方向门控注意力增强多尺度特征。
- 使用三元训练目标：Top-20% 困难像素损失、Sobel 梯度损失、教师特征保留损失。
- 推理对每层归一化特征计算差异并上采样，**逐层相乘**得到异常图；最大值为图像异常分数。

推理同时需要教师、适配器和学生。`EMA` 类保留工作副本实现，不能据类名认定它包含完整论文 EMA 的所有分支。

## 目录

```text
.
├── configs/ltd_net.yaml          # 当前 SGD 配置
├── scripts/
│   ├── train.py                  # 训练、保存最优权重、可选测试
│   ├── evaluate.py               # I-AUROC / P-AUROC / P-AUPRO / F1
│   ├── infer.py                  # checkpoint 或 ONNX 的单图/目录推理
│   ├── export_onnx.py            # 导出并对照 PyTorch、包装器、ONNX Runtime
│   ├── prepare_robustness.py     # 生成测试扰动，保持掩码对应关系
│   └── _common.py                # 共用配置、预处理和严格权重加载
├── src/anomalib/                 # 保留的 anomalib 包与 LTD-Net 改动
├── tests/                       # 发布流程回归检查
├── deployment/README.md          # 通用部署输入输出约定
├── datasets/README.md            # 数据准备说明；实际数据不提交
├── weights/README.md             # 权重说明；实际权重不提交
├── requirements/                 # 依赖说明与验证环境版本
├── docs/                        # 来源、迁移、复现和验证记录
├── pyproject.toml
├── LICENSE
└── third-party-programs.txt
```

`src/anomalib` 内保留其他基础模块，是为了兼容包初始化时的跨模型导入。
这些模块不等于本项目已经重新验证全部 anomalib 功能。

## 安装

使用独立的 Python 3.12 环境。以下命令均在**本 README 所在目录**运行，路径以当前工作目录为基准。

Windows PowerShell（无需激活脚本）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[ltd,export,verify]"
```

Linux：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[ltd,export,verify]'
```

NVIDIA GPU 环境：上面的 PyTorch 安装步骤可改为下面这一条，再安装本项目。
安装前按 [PyTorch 官方版本说明](https://pytorch.org/get-started/previous-versions/) 确认驱动与平台兼容。

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

以下示例中的 `python` 指已安装本项目的环境解释器。
PowerShell 若未激活环境，将 `python` 替换为 `.\.venv\Scripts\python.exe`。
检查导入位置应在本仓库 `src/anomalib` 下：

```bash
python -c "import anomalib; print(anomalib.__file__)"
python -m pip check
```

## 数据准备

从 [MVTec AD 官方页面](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) 按其条款获取数据，解压后组织为：

```text
datasets/MVTecAD/metal_nut/
├── train/good/*.png
├── test/good/*.png
├── test/<缺陷类型>/*.png
└── ground_truth/<缺陷类型>/*_mask.png
```

数据、权重与生成文件分别约定放在 `datasets/`、`weights/`、`outputs/`，均已在 `.gitignore` 中排除。
数据可以放在其他磁盘，用 `--data-root` 指定，无需修改源码。
脚本会检查本地数据目录，不自动下载整个数据集。

## 训练

```bash
python -m scripts.train --config configs/ltd_net.yaml --data-root datasets/MVTecAD --category metal_nut --output outputs/train/metal_nut
```

默认批量 16、200 epochs、学生/适配器学习率 0.4/0.05、SGD、动量 0.9、权重衰减 0.001，
采用余弦退火；每 2 个 epoch 验证一次。按 `image_AUROC` 保存最优权重，按 `train_loss` 早停。
线程数默认 0，便于 Windows 使用；可在配置中增大 `num_workers`。

主要输出：

```text
outputs/train/metal_nut/
├── run_config.json
├── weights/best.ckpt
├── weights/last.ckpt
├── logs/
└── test_metrics.json
```

可以用 `--accelerator cpu`、`--epochs 2` 进行小规模环境排查，用 `--skip-test` 跳过训练后的测试。
输出目录必须为空，避免混入旧实验。
首次正式训练会由 timm 下载教师预训练参数；离线训练需事先准备相应缓存。
`teacher_pretrained: false` 仅适合流程测试，不代表论文设定。

**数据划分须留意：**配置中的 `val_split_mode: same_as_test` 保留原工作副本行为，
验证和测试使用相同样本，按验证 AUROC 选择权重会使用测试标签信息。
若需要独立验证/测试，先将其改成 `from_test`，设置 `val_split_ratio`，重新训练，
评估时保持相同的配置和 seed。该选项会从官方测试集划出验证部分，最终测试不是完整官方测试集。
不要用不同划分直接比较论文表格。

## 评估

```bash
python -m scripts.evaluate --config configs/ltd_net.yaml --checkpoint outputs/train/metal_nut/weights/best.ckpt --data-root datasets/MVTecAD --category metal_nut --output outputs/evaluation/metal_nut
```

评估使用本地 anomalib 的数据管线、checkpoint 中的后处理统计和阈值，输出图像/像素 AUROC、
像素 AUPRO（FPR 上限 0.3）及图像/像素 F1。分数范围为 0–1，不自动乘 100。
结果在 `metrics.json`，实际配置在 `evaluation_config.json`。
该入口不采用旧单图脚本的 Otsu 前景掩码，不使用逐图显示归一化计算指标。
大类别的像素级评估需要缓存预测和掩码，内存不足时应减少并发或逐类别运行。

## 推理

```bash
python -m scripts.infer --checkpoint outputs/train/metal_nut/weights/best.ckpt --input datasets/MVTecAD/metal_nut/test --output outputs/inference/metal_nut
```

`--input` 也可传单张图片。每张图片输出原始浮点 `*.anomaly.npy` 和显示用 `*.heatmap.png`，
`scores.json` 保存原始图像异常分数。热力图的逐图缩放只用于显示；不同图片颜色深浅不能直接比较。
通用推理入口不输出“合格/不合格”，如需判定，应在独立验证集上校准阈值并固定归一化方案。

## ONNX 导出与部署

```bash
python -m scripts.export_onnx --checkpoint outputs/train/metal_nut/weights/best.ckpt --output outputs/export/metal_nut.onnx
python -m scripts.infer --onnx outputs/export/metal_nut.onnx --input datasets/MVTecAD/metal_nut/test --output outputs/onnx/metal_nut
```

输入固定为 float32 `[1,3,256,256]`，输出为原始异常图 `[1,1,256,256]`，ONNX opset 为 14。
RGB 解码、缩放、ImageNet 归一化和阈值判定均在图外。
导出脚本先比较原模型与包装器，再通过 ONNX checker 和 ONNX Runtime 进行数值对照；
只有检查通过才保存最终 ONNX。

默认检查随机输入及零输入。要加入真实图片，先用 `scripts.infer --save-input` 保存输入张量，
再把生成的 `*.input.npy` 传给导出脚本的 `--input-npy`（可以重复传入多个文件）。
静态导出不支持 tiled inference。随机张量数值一致不能替代整套测试集精度验证。
当前范围不包含 Jetson/TensorRT 或 RKNN 板端工程，见 [部署说明](deployment/README.md)。

## 鲁棒性数据准备与验证

```bash
python -m scripts.prepare_robustness --root datasets/MVTecAD --output outputs/robustness --categories metal_nut --dry-run
python -m pytest tests -q
```

去掉 `--dry-run` 才会生成扰动数据。该工具使用独立、非嵌套输出目录，不覆盖源数据。

## 来源、许可与复现记录

- 本地基础包版本字符串为 `2.3.0dev`；原始 Git 历史缺失，无法确认最初使用的官方 commit。
- [来源与改动](docs/PROVENANCE.md) 说明官方代码、项目改动和本次整理修改的边界。
- [迁移与排除清单](docs/MIGRATION.md) 解释哪些内容保留或暂不发布。
- [复现说明](docs/REPRODUCIBILITY.md) 列出论文/代码差异及发布前待确认项。
- 保留 Apache-2.0 [LICENSE](LICENSE)、源文件版权头和 [第三方声明](third-party-programs.txt)。
  子模块附带许可证继续适用；上游引用信息在 [docs/upstream](docs/upstream/README.md)。

论文标题暂沿用“LTD-Net：一种用于无监督工业产品异常检测的轻量级三元蒸馏网络”。
作者署名、论文发表链接及训练权重下载地址尚未补充，不提供推测链接。
