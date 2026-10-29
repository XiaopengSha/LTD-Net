# 数据

实际数据不提交 Git。MVTec AD 从其官方页面获取：
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

解压后的每个类别需包含 `train/good`、`test/good`、`test/<缺陷>`、
`ground_truth/<缺陷>/*_mask.png`。默认根目录为 `datasets/MVTecAD`，
可用 `--data-root` 指向其他位置。具体使用范围遵守数据集自身条款。

工作副本中的 imagenette、扰动数据和实验图片不属于本次发布数据。
没有确认可公开的自有样例，因此发布目录不附带真实工业图片。
