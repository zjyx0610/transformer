<<<<<<< HEAD
#!/usr/bin/env python3
import torch
import logging
import argparse
from src.model import Transformer
from src.data_loader import DataProcessor
from src.trainer import TransformerTrainer
from src.utils import set_seed, setup_logging, load_config


def main():
    parser = argparse.ArgumentParser(description='Transformer Training')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置随机种子
    set_seed(config['system']['seed'])

    # 设置日志
    setup_logging(config['system']['log_dir'])

    # 设置设备
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')

    # 数据预处理
    logging.info('加载数据...')
    data_processor = DataProcessor(config)
    train_loader, val_loader, test_loader = data_processor.get_data_loaders()
    src_vocab_size, tgt_vocab_size = data_processor.get_vocab_sizes()

    logging.info(f'源语言词汇表大小: {src_vocab_size}')
    logging.info(f'目标语言词汇表大小: {tgt_vocab_size}')

    # 创建模型
    logging.info('创建模型...')
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        max_seq_length=config['data']['max_length'] + 2
    ).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'总参数数: {total_params:,}')
    logging.info(f'可训练参数数: {trainable_params:,}')

    # 恢复训练（如果指定）
    if args.resume:
        logging.info(f'恢复训练从: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 创建训练器
    trainer = TransformerTrainer(model, config, train_loader, val_loader, device)

    # 开始训练
    trainer.train()

    logging.info('训练完成!')


if __name__ == '__main__':
=======
#!/usr/bin/env python3
import torch
import logging
import argparse
from src.model import Transformer
from src.data_loader import DataProcessor
from src.trainer import TransformerTrainer
from src.utils import set_seed, setup_logging, load_config


def main():
    parser = argparse.ArgumentParser(description='Transformer Training')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='配置文件路径')
    parser.add_argument('--resume', type=str, help='恢复训练的检查点路径')
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置随机种子
    set_seed(config['system']['seed'])

    # 设置日志
    setup_logging(config['system']['log_dir'])

    # 设置设备
    device = torch.device(config['system']['device'] if torch.cuda.is_available() else 'cpu')
    logging.info(f'使用设备: {device}')

    # 数据预处理
    logging.info('加载数据...')
    data_processor = DataProcessor(config)
    train_loader, val_loader, test_loader = data_processor.get_data_loaders()
    src_vocab_size, tgt_vocab_size = data_processor.get_vocab_sizes()

    logging.info(f'源语言词汇表大小: {src_vocab_size}')
    logging.info(f'目标语言词汇表大小: {tgt_vocab_size}')

    # 创建模型
    logging.info('创建模型...')
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        num_decoder_layers=config['model']['num_decoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        max_seq_length=config['data']['max_length'] + 2
    ).to(device)

    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'总参数数: {total_params:,}')
    logging.info(f'可训练参数数: {trainable_params:,}')

    # 恢复训练（如果指定）
    if args.resume:
        logging.info(f'恢复训练从: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # 创建训练器
    trainer = TransformerTrainer(model, config, train_loader, val_loader, device)

    # 开始训练
    trainer.train()

    logging.info('训练完成!')


if __name__ == '__main__':
>>>>>>> ea7271bb0dd6cc6e23ef1476823578defda08b95
    main()