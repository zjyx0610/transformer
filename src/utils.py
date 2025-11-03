<<<<<<< HEAD
import torch
import numpy as np
import random
import yaml
import logging
import os
from typing import Dict, Any


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """保存配置文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """生成正弦位置编码"""
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pos_encoding = torch.zeros(max_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)

    return pos_encoding


def create_mask(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int) -> tuple:
    """创建注意力掩码"""
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask

=======
import torch
import numpy as np
import random
import yaml
import logging
import os
from typing import Dict, Any


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log'), encoding='utf-8'),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """保存配置文件"""
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """生成正弦位置编码"""
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pos_encoding = torch.zeros(max_len, d_model)
    pos_encoding[:, 0::2] = torch.sin(position * div_term)
    pos_encoding[:, 1::2] = torch.cos(position * div_term)

    return pos_encoding


def create_mask(src: torch.Tensor, tgt: torch.Tensor, pad_idx: int) -> tuple:
    """创建注意力掩码"""
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)

    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(3)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    tgt_mask = tgt_mask & nopeak_mask

>>>>>>> ea7271bb0dd6cc6e23ef1476823578defda08b95
    return src_mask, tgt_mask