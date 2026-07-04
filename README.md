# 基于深度学习的植物病害识别系统

这是一个面向课程项目和二次开发的植物病害识别系统，包含模型训练、单图/多图识别、批量结果导出、Grad-CAM 可视化、自定义数据集训练、数据集划分和数据泄漏检查等功能。

> 仓库只保存源码、脚本、测试和通用文档，不包含本地训练生成的模型、数据集和报告。首次使用预测功能前，需要先在本地训练或指定自己的模型权重。

## 获取与安装

```powershell
git clone https://github.com/wuyao52/Plant-Disease---Identification-System.git
cd Plant-Disease---Identification-System
python -m pip install -r requirements.txt
```

## 启动网页

```powershell
streamlit run app.py
```

浏览器打开：

```text
http://127.0.0.1:8501
```

页面包含两个主要功能页：

- `图片识别`：支持一张或多张图片同时识别，并可下载批量预测 CSV。
- `自定义训练`：填写自己的数据集路径，在网页中完成数据校验、训练集/验证集/测试集划分和模型训练。

## 首次生成可用模型

仓库不上传模型文件，所以刚克隆后直接预测会提示缺少模型。可以先运行一次训练：

```powershell
python train.py --config config.yaml
```

默认配置读取：

```text
data/raw/plantvillage
```

如果该目录不存在或为空，训练脚本会尝试自动下载一个小型 PlantVillage 公开子集用于首次训练。训练结束后会生成：

```text
models/plant_disease_mobilenet_v3_small.pt
outputs/
assets/class_names.json
```

这些都是本地运行产物，已被 `.gitignore` 排除，不会随代码上传。

如果只想快速验证流程，也可以生成合成演示数据：

```powershell
python scripts/create_demo_dataset.py --output data/raw/demo_leaf --images-per-class 24
python train.py --data-dir data/raw/demo_leaf --epochs 1 --checkpoint models/demo_synthetic.pt
```

再次启动网页后，系统会自动优先加载 `models/demo_synthetic.pt`。

## 使用自己的数据训练

网页方式：打开 `自定义训练` 页面，填写数据集路径，然后依次点击：

```text
校验数据集 -> 生成训练集/测试集 -> 开始训练
```

命令行方式：

```powershell
python scripts/validate_dataset.py D:\my_dataset
python train_custom.py --source D:\my_dataset --name my_leaf_model --epochs 15 --batch-size 16 --force
```

支持普通 ImageFolder 格式：

```text
my_dataset/class_a/*.jpg
my_dataset/class_b/*.jpg
```

也支持已经分好的数据集：

```text
my_dataset/train/class_a/*.jpg
my_dataset/val/class_a/*.jpg
my_dataset/test/class_a/*.jpg
```

详细说明见：

```text
docs/custom_training.md
docs/dataset.md
```

## 项目结构

```text
app.py                         # Web 界面
train.py                       # 默认训练入口
train_custom.py                # 自定义数据集训练入口
predict.py                     # 命令行单图预测
config.yaml                    # 默认配置
src/plant_disease              # 核心代码
scripts                        # 数据下载、校验、划分和查重脚本
docs                           # 项目文档
data                           # 本地数据目录
models                         # 本地模型权重目录
outputs                        # 本地训练输出目录
```

## 测试

```powershell
python -m pytest
```
