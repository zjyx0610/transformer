# 项目名称
北京交通大学计算机学院2025年大模型基础与应用期中课程作业

## 项目所有者信息

北京交通大学，计算机学院，25120376，张姜雨辛

## 硬件要求

### 最低要求
- **CPU**: 支持运行，但速度较慢（约1-2分钟/batch）
- **内存**: 8GB以上

### 推荐配置
- **GPU**: NVIDIA GPU (CUDA支持)
- **显存**: 8GB以上
- **内存**: 16GB以上

## 环境配置

### 方式一：自动环境配置（推荐新手）
使用提供的脚本自动创建conda环境并安装依赖：
```bash
# 创建环境并安装依赖，输出日志到result.log
nohup bash run.sh > result.log 2>&1 &
```

### 方式二：手动环境配置
1. 创建conda环境：
```bash
conda create -n myenv python=3.8
conda activate myenv
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行命令

### 基础训练（默认配置）
```bash
python train.py
```

### 后台运行训练
如果已经配置好环境，使用以下命令在后台运行：
```bash
nohup bash run_hand.sh > result.log 2>&1 &
```

### 重现实验的精确命令
为确保实验结果可重现，请使用以下包含随机种子的命令：

```bash
#举例： 使用默认配置文件，设置随机种子为42
python train.py --seed 42 --config configs/base.yaml
```

## 超参数配置

### 方式一：修改配置文件
编辑 `configs/base.yaml` 文件来修改训练参数：
```yaml
# 示例配置
learning_rate: 0.001
batch_size: 32
num_epochs: 100
seed: 42
```

### 方式二：命令行参数
在所有 `python train.py` 命令后直接指定参数：
```bash
python train.py --config configs/base.yaml --lr 0.001 --batch_size 32 --epochs 100 --seed 42
```

## 输出文件

- **训练日志**: `result.log`
- **模型检查点**: 保存在 `checkpoints/` 目录
- **训练曲线**: 保存在 `checkpoints/` 目录

## 注意事项

1. 训练前需提前下载所需数据集（在data中包含）
2. 确保有足够的磁盘空间存储模型和日志
3. 使用相同的随机种子可以确保实验结果可重现
4. GPU训练会显著加快训练速度，推荐使用
=======

# 项目名称
北京交通大学计算机学院2025年大模型基础与应用期中课程作业

## 项目所有者信息

北京交通大学，计算机学院，25120376，张姜雨辛

## 硬件要求

### 最低要求
- **CPU**: 支持运行，但速度较慢（约1-2分钟/batch）
- **内存**: 8GB以上

### 推荐配置
- **GPU**: NVIDIA GPU (CUDA支持)
- **显存**: 8GB以上
- **内存**: 16GB以上

## 环境配置

### 方式一：自动环境配置（推荐新手）
使用提供的脚本自动创建conda环境并安装依赖：
```bash
# 创建环境并安装依赖，输出日志到result.log
nohup bash run.sh > result.log 2>&1 &
```

### 方式二：手动环境配置
1. 创建conda环境：
```bash
conda create -n myenv python=3.8
conda activate myenv
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 运行命令

### 基础训练（默认配置）
```bash
python train.py
```

### 后台运行训练
如果已经配置好环境，使用以下命令在后台运行：
```bash
nohup bash run_hand.sh > result.log 2>&1 &
```

### 重现实验的精确命令
为确保实验结果可重现，请使用以下包含随机种子的命令：

```bash
#举例： 使用默认配置文件，设置随机种子为42
python train.py --seed 42 --config configs/base.yaml
```

## 超参数配置

### 方式一：修改配置文件
编辑 `configs/base.yaml` 文件来修改训练参数：
```yaml
# 示例配置
learning_rate: 0.001
batch_size: 32
num_epochs: 100
seed: 42
```

### 方式二：命令行参数
在所有 `python train.py` 命令后直接指定参数：
```bash
python train.py --config configs/base.yaml --lr 0.001 --batch_size 32 --epochs 100 --seed 42
```

## 输出文件

- **训练日志**: `result.log`
- **模型检查点**: 保存在 `checkpoints/` 目录
- **训练曲线**: 保存在 `checkpoints/` 目录

## 注意事项

1. 训练前需提前下载所需数据集（在data中包含）
2. 确保有足够的磁盘空间存储模型和日志
3. 使用相同的随机种子可以确保实验结果可重现
4. GPU训练会显著加快训练速度，推荐使用
>>>>>>> ea7271bb0dd6cc6e23ef1476823578defda08b95
