# 权重

本次不附带 `.ckpt`、`.onnx` 或板端模型，也未确认公开下载链接。
训练输出在 `outputs/train/<类别>/weights/best.ckpt`，可自行复制到本目录长期保管。

权重必须与当前三层 GatedAdapter（带 alpha 和末尾 BatchNorm）、ResNet10/FPN 结构一致。
发布脚本使用 strict=True；旧的无门控适配器、ResNet18 学生或其他消融权重不能静默混用。
加载 checkpoint 时不下载教师参数，直接严格恢复 checkpoint 中保存的教师状态。
仅加载自己生成或信任来源的 Lightning checkpoint。此类文件含预处理器/评估器等 Python 对象，
加载入口显式使用 weights_only=False；模型参数仍 strict=True，二者不是同一个设置。
