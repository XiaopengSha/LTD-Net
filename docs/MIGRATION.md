# 文件迁移与排除清单

所有“排除”仅指未放入发布副本，原始文件没有删除或覆盖。
复制使用明确的 `src` 和根文件清单，不扫描发布目录作为输入，因此不会递归包含自身。
`migration.csv` 提供逐文件迁移记录与原文件 SHA-256；整目录排除使用聚合记录。

## 主要迁移

| 原位置 | 发布位置 | 处理依据 |
| --- | --- | --- |
| src/anomalib | src/anomalib | 保留包导入闭包和版权；下列未接入变体除外 |
| train.py | scripts/train.py + configs/ltd_net.yaml | 提取真实入口中的 SGD、指标、回调和训练设置；替换写死的数据/输出路径 |
| export_onnx2.py | scripts/export_onnx.py | 保留原模型/包装器/ORT 三方数值对照，修正新目录导入和输出位置 |
| test.py、test-wenjianjia.py | scripts/infer.py、scripts/evaluate.py | 参考输入输出需求，重建可运行统一入口，未原样搬运缺失依赖及物体专用后处理 |
| augment_robustness.py | scripts/prepare_robustness.py | 已有参数化输入、非嵌套输出检查和掩码处理，可独立使用 |
| README.md | README.md | 重写为 LTD-Net 中文安装与操作说明 |
| CITATION.cff | docs/upstream/CITATION.cff | 保留上游引用资料，避免误当成 LTD-Net 引用 |
| LICENSE、third-party-programs.txt | 同名文件 | 原样保留 |
| .gitignore、pyproject.toml | 同名文件 | 补充发布约定、依赖和打包配置 |

## 模型目录中未发布的实验

| 文件 | 内容检查后的原因 |
| --- | --- |
| torch_model_res10_FPN.py | 独立 ResNet10/FPN 学生试验，活动入口并未导入 |
| torch_model_res10_FPN_ada.py | 独立领域适配变体，未接入当前 SPA 入口 |
| orin_PDN.py | 另一种部署/学生结构，非当前训练调用链 |
| loss_gra.py | 两路特征的梯度损失，当前入口使用三路 loss_gra_spa |
| loss.py | 包含 FrequencyDomainLoss 和多段注释实验，活动入口不使用；不能视作纯官方 loss |
| 五个“副本”txt | 不是 Python 导入目标；部分与现有文件相同，部分含历史适配器/训练差异，全部保留在工作副本供追溯 |

活动 `SPA.py` 内仍保留未实例化的 ResNet18WithFPN 类，以及 EMA 中未参与 forward 的状态成员，
避免在缺少最终消融版本信息时改变 checkpoint 结构；这不表示它们属于当前执行图。

## 其他排除项

| 原位置 | 检查结论/原因 |
| --- | --- |
| stfpm.yaml、stfpm2.yaml | 参数与活动 train.py 不一致；后一份使用 Folder/Real-IAD 布局，不能直接作为 MVTecAD 主配置 |
| rd4ad.py、dinomaly.py | 有实际基线训练代码，但类别/输出硬编码，未证实对应最终论文实验；暂不发布专用入口，包内基线实现保留 |
| light.py | 只调用教师/学生测速，未包含适配器与异常图，不是 LTD-Net 完整部署链路 |
| biaoge-fps.py、biaoge-la.py | 使用写在脚本中的实验数字绘图，不能从模型重算结论 |
| clean.py | 根据文件对应关系执行删除的个人数据清理工具，不属于训练/评估必要步骤 |
| test-wenjianjia.py | 引用缺失的 inference_metrics 模块；迁移需求而非复制损坏脚本 |
| test.py | 有 Otsu 前景过滤、逐图归一化及 strict=False 加载，不作为通用评估实现 |
| datasets/ | 实际含 imagenette 等数据；大型输入，不随源码发布 |
| output/、test_image/、light_metrics/ | 扰动图片、可视化/输出及历史性能表；不作为可复现实验依据附带 |
| wood.onnx | 已导出模型；没有确认对应 checkpoint、训练配置与公开用途 |
| artifacts/ | 检查了 README、脚本导入与 C++ 入口；含板端交付、调试/诊断、构建缓存、虚拟环境、PPT 素材及 node_modules；按用户选择本次不发布 |
| tools/ | 上游通用推理、benchmark、tiled-ensemble 入口，发布主流程不依赖；其中旧 torch_inference 有过时 API 引用 |
| docs/、examples/、tests/（原） | 上游全库文档、示例与测试，大部分超出 LTD-Net 主流程；发布版另建针对性文档和测试，子模块许可仍保留 |
| .github/ | 上游 CI、发布、机器人与治理设置，不能直接视为新独立仓库的自动化 |
| GOVERNANCE、MAINTAINERS、CODE_OF_CONDUCT、CONTRIBUTING、SECURITY、CHANGELOG 等 | 官方项目治理、联系人和全库历史，未作为新仓库自身声明复制；来源指向官方仓库 |
| .vscode、.cursor、缓存、临时与编辑器配置 | 本机开发状态，不属于可安装源码 |
| EMBEDDED_EDGE_AI_PROJECT_MEMORY.md | 包含本机路径、个人背景和未来计划，非公开发布文档 |
| 论文原稿 | 仅用于理解项目，不拷贝到发布目录 |
| github-release-review/ | 本次本地校验环境、原文件哈希、验证输出；位于发布目录外，不上传 |

不能确认用途的重要实验已在 `REPRODUCIBILITY.md` 集中列出。没有依据文件名直接删除任何原始内容。
