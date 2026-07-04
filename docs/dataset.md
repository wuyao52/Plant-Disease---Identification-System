# 数据集准备说明

## 推荐数据格式

项目使用 `ImageFolder` 目录结构。每个子文件夹代表一个类别，文件夹名称会作为类别名写入模型权重。

```text
data/raw/plantvillage
|-- Apple___Apple_scab
|   |-- 0001.jpg
|   `-- 0002.jpg
|-- Apple___healthy
|   |-- 0001.jpg
|   `-- 0002.jpg
`-- Tomato___Late_blight
    |-- 0001.jpg
    `-- 0002.jpg
```

也可以使用已经分好的训练集、验证集和测试集：

```text
my_dataset
|-- train
|   `-- class_a
|-- val
|   `-- class_a
`-- test
    `-- class_a
```

## 获取示例数据

可以使用项目脚本下载公开 PlantVillage 子集：

```powershell
python scripts/download_plantvillage_subset.py --output data/raw/plantvillage --images-per-class 8
```

首次运行默认训练时，如果 `data/raw/plantvillage` 不存在或为空，训练脚本会尝试自动下载一个小型公开子集。

## 数据划分

如果输入数据尚未分为 `train`、`val`、`test`，项目会按配置比例随机划分：

- 验证集：15%
- 测试集：10%
- 训练集：75%

比例可在 `config.yaml` 中修改：

```yaml
data:
  val_split: 0.15
  test_split: 0.10
```

也可以手动生成固定划分：

```powershell
python scripts\prepare_split_dataset.py --source data\raw\plantvillage --output data\processed\plantvillage_split --force
```

## 数据检查

训练前建议先校验数据集：

```powershell
python scripts\validate_dataset.py data\raw\plantvillage
```

对于已经分好的数据集，可以检查跨训练集、验证集、测试集的重复或近重复图片：

```powershell
python scripts\check_split_leakage.py --root data\processed\plantvillage_split
```

## 本地运行产物

`data/raw`、`data/processed`、`models` 和 `outputs` 下生成的数据、权重和报告默认不会上传到 GitHub。
