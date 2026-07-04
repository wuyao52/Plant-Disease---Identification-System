# 自定义数据集训练说明

本项目支持别人使用自己的植物病害图片重新训练模型。推荐先把图片整理成下面两种格式之一。

## 格式一：普通 ImageFolder

适合只有一份原始图片数据的情况。每个类别一个文件夹：

```text
my_dataset
├── Apple_scab
│   ├── 001.jpg
│   └── 002.jpg
├── Apple_healthy
└── Tomato_late_blight
```

训练前先校验：

```powershell
cd D:\xt
python scripts/validate_dataset.py D:\my_dataset --output outputs\my_dataset_validation.json
```

然后自动划分训练集、验证集、测试集并训练：

```powershell
cd D:\xt
python train_custom.py --source D:\my_dataset --name my_leaf_model --epochs 15 --batch-size 16 --force
```

生成文件：

```text
data/processed/custom_my_leaf_model/train
data/processed/custom_my_leaf_model/val
data/processed/custom_my_leaf_model/test
models/my_leaf_model.pt
outputs/runs/my_leaf_model/metrics.json
outputs/runs/my_leaf_model/confusion_matrix.csv
outputs/runs/my_leaf_model/leakage_report.json
```

## 格式二：已经分好的 train/val/test

如果用户已经自己划分好数据：

```text
my_dataset
├── train
│   ├── class_a
│   └── class_b
├── val
│   ├── class_a
│   └── class_b
└── test
    ├── class_a
    └── class_b
```

可以直接训练，不会重新划分：

```powershell
cd D:\xt
python train_custom.py --source D:\my_dataset --name my_split_model --epochs 15
```

## 常用参数

- `--source`：自定义数据集根目录，必填。
- `--name`：训练任务名，用于生成模型和报告文件名。
- `--epochs`：训练轮数。
- `--batch-size`：批大小，CPU 推荐 8 或 16。
- `--architecture`：可选 `mobilenet_v3_small`、`resnet18`、`efficientnet_b0`。
- `--max-per-class`：每类最多使用多少张图片，适合先做小规模测试。
- `--force`：覆盖已经生成过的自定义 split 目录。
- `--no-pretrained`：不使用 ImageNet 预训练权重。
- `--freeze-backbone`：只训练分类头，适合数据很少时快速实验。

## 建议

每个类别建议至少 50 张图片。若图片来自真实环境，最好覆盖不同光照、背景、拍摄距离和病害阶段。训练结果不要只看准确率，也要看混淆矩阵、每类召回率和泄漏检查报告。
