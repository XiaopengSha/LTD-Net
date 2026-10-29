# 发布版验证记录

验证日期：2026-10-29。范围为 Windows / Python 3.12.10 / CPU，
验证环境位于发布目录之外。没有使用真实工业数据、正式训练 checkpoint 或板端设备。

## 结果

| 检查 | 结果与边界 |
| --- | --- |
| 安装 | 从发布目录执行 editable 安装成功；pip check 无冲突 |
| 导入 | anomalib 来自发布目录的 src/anomalib，而非另一份官方安装 |
| CLI | train、evaluate、infer、export_onnx、prepare_robustness 的 --help 全部成功 |
| 回归测试 | `python -m pytest tests -q`：11 passed |
| 训练/评估链路 | 合成 MVTec 风格数据，8 张正常训练图，4 张正常+4 张异常测试图，32×32，2 epochs；六个流程步骤全部成功 |
| checkpoint | 完整保存并严格恢复；缺失网络参数时拒绝加载；后处理状态保留 |
| 评估一致性 | 训练后测试结果与独立 evaluate 入口的全部指标一致 |
| 推理一致性 | 8 张合成测试图片，checkpoint 与 ONNX 原始异常图最大绝对误差 2.3842e-7 |
| 默认尺寸 ONNX | 256×256、随机初始化教师、固定随机输入+零输入，通过 checker 和 ORT；随机输入最大绝对误差 3.5763e-7 |
| 模型语义对照 | 使用原始工作副本的真实模型/损失代码，在内存中执行，不写入原目录；相同参数下训练特征、默认损失、异常图与分数完全一致，state_dict 参数键一致 |
| 鲁棒性工具 | 实际生成亮度和水平翻转数据；确认翻转后的缺陷掩码与原掩码对应 |
| Git 忽略规则 | 8 类产物/私密文件被忽略，7 类必要源码/文档保持可提交；包括 src 内的 data/datasets 不被误忽略 |
| 原始文件 | 对准备前记录的 742 个原始文件进行 SHA-256 复核，未发现内容改变 |
| 静态检查 | 所有发布 Python 文件可解析；扫描本机路径、常见凭据模式、二进制模型、缓存及超大文件，未发现命中 |

合成数据的指标只用于证明流程连通，不能代表 MVTec AD、论文精度或泛化能力。
32×32 用于缩短集成检查时间，论文/默认训练尺寸仍为 256×256。
原模型语义对照为了避免下载，双方均关闭教师预训练并加载同一组参数；
这项测试验证整理没有改变计算，并不验证预训练模型的精度。

## 已定位并处理的问题

1. pandas 3 的字符串列与该版本的 Split 枚举比较产生空数据集。依赖现约束为 pandas 2.x，未改动基础数据筛选实现。
2. 原 warm-up 访问 `batch['image']` 与当前 ImageBatch 不兼容，已支持 `.image`。
3. Lightning checkpoint 包含序列化的预处理器/评估器；显式采用 `weights_only=False` 恢复可信 checkpoint，网络参数依然 `strict=True`。
4. 原模型 ONNX 分支在定义 pred_score 前使用它，已修复；导出依然通过包装器与原 eval 结果比对。
5. Engine 自动注入 ModelCheckpoint；评估入口不再传与它冲突的 `enable_checkpointing=False`。
6. 原批量推理缺失 inference_metrics 依赖，已由新的评估和推理入口替代。

测试期间 Windows 执行沙箱对临时目录访问造成权限错误；在获得工具执行许可后，同一代码在本地审查目录完成验证。
这不是通过放宽数值误差阈值或跳过检查解决的。

## 仍存在的提示与未验证项

- ONNX 使用 legacy exporter；保留固定尺寸 tracing 提示，不承诺动态输入或任意 batch。
- Lightning 提示预处理器/评估器已作为模块保存；本次仍保留其 checkpoint 兼容行为。
- 本地 AUPRO 有参数弃用提示；基础完美排序/反序测试通过，不等于全套官方指标回归。
- 未做完整 MVTec AD 训练、GPU 测试、Linux 实机测试、正式 checkpoint 兼容验证、TensorRT/RKNN 转换或板端性能测试。
- 没有确认论文表格和已报告 FPS 对应的最终权重及代码版本，未将这些数字当成验证结果写入项目简介。
- 凭据检查是针对发布内容的模式扫描与文件选择审查，不是第三方安全审计。

## 环境版本

| 包 | 实测版本 |
| --- | --- |
| torch / torchvision | 2.8.0+cpu / 0.23.0+cpu |
| lightning | 2.6.6 |
| torchmetrics | 1.8.2 |
| timm | 1.0.30 |
| pandas | 2.3.3 |
| numpy | 2.5.4 |
| onnx / onnxruntime | 1.23.2 / 1.31.1 |
| PyYAML | 6.0.3 |

完整去除本机路径的版本清单见 [verified-windows-cpu.txt](../requirements/verified-windows-cpu.txt)。
它是本次 CPU 验证环境记录，不是原论文的环境锁定文件，不应直接作为 GPU 环境约束。
迁移清单见 [migration.csv](migration.csv)，来源比对见 [PROVENANCE.md](PROVENANCE.md)。
