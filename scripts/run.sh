#!/bin/bash

# 设置环境
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 创建必要的目录
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

# 安装依赖
pip install -r requirements.txt

# 训练模型
python train.py --config configs/base.yaml

echo "训练完成!"