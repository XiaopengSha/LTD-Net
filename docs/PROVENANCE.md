# 来源与改动

本项目基于 [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib)，
工作副本的版本字符串是 `2.3.0dev`。原始目录没有 `.git`，所以无法从本地历史确定基准提交或逐个提交归属。

本次只读比对下载了官方 main 源码归档，其归档注释记录的 commit 为
`0cc8e6331a2a667fdb286b981d335393dc846383`。
对工作副本 `src` 下 428 个非缓存文件进行文本换行归一化比对：282 个相同，131 个不同，15 个不存在于对照快照。
**这是来源分类的辅助证据，不是原项目基准版本；不同文件不能全部归为作者修改。**
详细列表见 `upstream-comparison.json`。

## 保留的基础代码

保留 `src/anomalib/` 的公共组件、数据模块、训练引擎、指标和包初始化依赖。
它们存在 eager import 和跨模型调用，因此未为了缩小文件数量裁成仅剩 stfpm 的不完整包。
`stfpm/torch_model.py` 与对照快照相同，作为官方架构参考保留；活动 Lightning 入口不调用它。
其他模块沿用工作副本，不自动替换成最新上游代码。

## 确定的项目扩展与活动路径

```text
scripts/train.py
  -> anomalib.models.Stfpm
     -> models/image/stfpm/lightning_model.py
        -> SPA.py                    # LTD-Net 模型
        -> loss_gra_spa.py           # 三元损失
        -> anomaly_map.py           # 多层差异图乘积
```

原 `train.py`、`export_onnx2.py`、`augment_robustness.py` 和推理实验脚本提供了发布入口的依据。
`metrics/aupro.py` 包含全局排序/区域加权计算的实现，与对照快照不同，本版原样保留。
`loss.py` 含 FFT 实验，即使带有官方版权头，也不把它误标为未修改的官方基线。

## 本次发布整理产生的修改

本次仅在发布副本中新增/整理命令行入口、配置、中文说明、测试和忽略规则；
对活动模型增加参数入口、输入约束及明确运行错误修复。
具体差异见 `REPRODUCIBILITY.md` 和迁移清单。旧模型参数名称和实际前向计算予以保留。
`pyproject.toml` 保留原包名 `anomalib` 和基础依赖，增加 ltd/export/verify 可选组，
补充 wheel 的 src 包位置及发布目录的 sdist 文件列表。
只供从源码使用和独立 GitHub 仓库发布，不将该 fork 上传为官方 PyPI anomalib 包。

## 许可证与引用

保留根目录 Apache-2.0 `LICENSE`、`third-party-programs.txt`、源代码版权头以及各子模块 LICENSE。
新增的发布工具采用 Apache-2.0；不得把保留的上游作者列表解读为 LTD-Net 作者。
上游 `CITATION.cff` 放在 `docs/upstream`，避免根目录引用元数据把用户引向错误的论文。
最终 LTD-Net 署名与论文链接由作者确认后补充。
