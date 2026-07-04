# 数据目录说明

`raw` 用于放原始图片数据，`processed` 用于放可选的中间处理结果。项目训练脚本直接支持 `torchvision.datasets.ImageFolder` 目录结构。

真实数据建议放在：

```text
data/raw/plantvillage/<类别名>/<图片文件>
```

合成演示数据可通过以下命令生成：

```powershell
python scripts/create_demo_dataset.py --output data/raw/demo_leaf
```
