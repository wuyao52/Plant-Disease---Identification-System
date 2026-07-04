# 基于深度学习的植物病害识别系统

这是一个面向课程项目和二次开发的植物病害识别系统，包含模型训练、单图/多图识别、批量结果导出、Grad-CAM 可视化、自定义数据集训练、数据集划分和数据泄漏检查等功能。

## 快速运行

```powershell
cd D:\xt
streamlit run app.py
```

浏览器打开：

```text
http://127.0.0.1:8501
```

页面包含两个主要功能页：

- `图片识别`：支持一张或多张图片同时识别，并可下载批量预测 CSV。
- `自定义训练`：填写自己的数据集路径，在网页中完成数据校验、训练集/验证集/测试集划分和模型训练。

## 使用自己的数据训练

网页方式：打开 `自定义训练` 页面，填写数据集路径，然后依次点击：

```text
校验数据集 -> 生成训练集/测试集 -> 开始训练
```

命令行方式：

```powershell
cd D:\xt
python scripts\validate_dataset.py D:\my_dataset
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

## 默认训练

默认配置读取：

```text
data/raw/plantvillage
```

运行：

```powershell
cd D:\xt
python train.py --config config.yaml
```

训练生成的模型、数据集副本和输出报告属于本地运行产物，已经在 `.gitignore` 中排除，不会随代码上传。

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
cd D:\xt
python -m pytest
```
