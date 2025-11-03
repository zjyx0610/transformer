<<<<<<< HEAD
#!/usr/bin/env python3
import torch
import logging
import argparse
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from src.model import Transformer
from src.data_loader import DataProcessor
from src.utils import set_seed, setup_logging, load_config


class ModelValidator:
    def __init__(self, model, config, data_processor, device):
        self.model = model
        self.config = config
        self.data_processor = data_processor
        self.device = device
        self.model.eval()

    def calculate_loss_and_accuracy(self, data_loader):
        """计算数据集的平均损失和准确率"""
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0

        with torch.no_grad():
            for src, tgt in tqdm(data_loader, desc="计算损失和准确率"):
                src, tgt = src.to(self.device), tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
                tgt_mask = self._create_decoder_mask(tgt_input)

                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                # 计算准确率
                predictions = torch.argmax(output, dim=-1)
                mask = (tgt_output != 0)
                correct = (predictions == tgt_output) & mask

                batch_tokens = mask.sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                correct_tokens += correct.sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        perplexity = np.exp(avg_loss)

        return avg_loss, accuracy, perplexity

    def calculate_exact_match(self, data_loader, num_batches=5):
        """计算精确匹配率"""
        exact_matches = 0
        total_samples = 0

        # 获取特殊token的ID
        pad_id = 0
        bos_id = self.data_processor.src_tokenizer.token_to_id("[BOS]")
        eos_id = self.data_processor.src_tokenizer.token_to_id("[EOS]")
        special_tokens = {pad_id, bos_id, eos_id}

        with torch.no_grad():
            batch_count = 0
            for src, tgt in data_loader:
                if batch_count >= num_batches:
                    break

                src, tgt = src.to(self.device), tgt.to(self.device)

                # 使用teacher forcing计算输出
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
                tgt_mask = self._create_decoder_mask(tgt_input)

                output = self.model(src, tgt_input, src_mask, tgt_mask)
                predictions = torch.argmax(output, dim=-1)

                # 检查精确匹配
                for i in range(predictions.size(0)):
                    pred_seq = predictions[i].cpu().numpy()
                    target_seq = tgt_output[i].cpu().numpy()

                    # 移除padding和特殊token
                    pred_tokens = [token for token in pred_seq if token not in special_tokens]
                    target_tokens = [token for token in target_seq if token not in special_tokens]

                    if len(pred_tokens) == len(target_tokens) and all(
                            p == t for p, t in zip(pred_tokens, target_tokens)):
                        exact_matches += 1
                    total_samples += 1

                batch_count += 1

        exact_match_rate = (exact_matches / total_samples) * 100 if total_samples > 0 else 0
        return exact_match_rate

    def _create_decoder_mask(self, tgt):
        """创建解码器掩码"""
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device=self.device), diagonal=1)).bool()
        tgt_mask = tgt_pad_mask & nopeak_mask
        return tgt_mask


def load_trained_model(config, model_path, data_processor, device):
    """加载训练好的模型"""
    logging.info(f"加载模型: {model_path}")

    # 获取词汇表大小
    src_vocab_size, tgt_vocab_size = data_processor.get_vocab_sizes()

    # 创建模型结构
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

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("从检查点加载模型状态")
    else:
        model.load_state_dict(checkpoint)
        logging.info("直接加载模型权重")

    logging.info("模型加载成功")
    return model


def validate_model(config, model_path, data_processor, device):
    """完整的模型验证流程"""
    # 加载模型
    model = load_trained_model(config, model_path, data_processor, device)

    # 创建验证器
    validator = ModelValidator(
        model=model,
        config=config,
        data_processor=data_processor,
        device=device
    )

    # 获取数据加载器
    train_loader, val_loader, test_loader = data_processor.get_data_loaders()

    results = {
        'model_path': model_path,
        'config': {
            'model': config['model'],
            'data': config['data']
        }
    }

    # # 计算训练集指标
    # logging.info("计算训练集指标...")
    # train_loss, train_accuracy, train_perplexity = validator.calculate_loss_and_accuracy(train_loader)
    # results['train'] = {
    #     'loss': train_loss,
    #     'accuracy': train_accuracy,
    #     'perplexity': train_perplexity
    # }

    # 计算验证集指标
    logging.info("计算验证集指标...")
    val_loss, val_accuracy, val_perplexity = validator.calculate_loss_and_accuracy(val_loader)
    results['validation'] = {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'perplexity': val_perplexity
    }

    # 计算测试集指标
    logging.info("计算测试集指标...")
    test_loss, test_accuracy, test_perplexity = validator.calculate_loss_and_accuracy(test_loader)
    results['test'] = {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'perplexity': test_perplexity
    }

    # 计算精确匹配率（只在测试集上计算，减少计算量）
    logging.info("计算精确匹配率...")
    exact_match_rate = validator.calculate_exact_match(test_loader, num_batches=3)
    results['exact_match_rate'] = exact_match_rate

    return results


def print_results(results):
    """打印验证结果"""
    print("=" * 60)
    print("模型验证结果")
    print("=" * 60)
    print(f"模型路径: {results['model_path']}")

    print(f"\n模型配置:")
    print(f"  d_model: {results['config']['model']['d_model']}")
    print(f"  nhead: {results['config']['model']['nhead']}")
    print(f"  编码器层数: {results['config']['model']['num_encoder_layers']}")
    print(f"  解码器层数: {results['config']['model']['num_decoder_layers']}")

    print(f"\n性能指标:")
    print("          |   损失   |   准确率  |   困惑度  ")
    print("-" * 50)
    # print(
        # f"训练集   | {results['train']['loss']:8.4f} | {results['train']['accuracy']:7.2%} | {results['train']['perplexity']:8.2f}")
    print(
        f"验证集   | {results['validation']['loss']:8.4f} | {results['validation']['accuracy']:7.2%} | {results['validation']['perplexity']:8.2f}")
    print(
        f"测试集   | {results['test']['loss']:8.4f} | {results['test']['accuracy']:7.2%} | {results['test']['perplexity']:8.2f}")

    print(f"\n其他指标:")
    print(f"精确匹配率: {results['exact_match_rate']:.2f}%")


def save_results(results, save_dir="results"):
    """保存验证结果到文件"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # 保存JSON格式的完整结果
    json_path = save_path / "validation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存简洁的文本报告
    txt_path = save_path / "validation_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("模型验证报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"模型路径: {results['model_path']}\n\n")

        f.write("模型配置:\n")
        f.write(f"  d_model: {results['config']['model']['d_model']}\n")
        f.write(f"  nhead: {results['config']['model']['nhead']}\n")
        f.write(f"  编码器层数: {results['config']['model']['num_encoder_layers']}\n")
        f.write(f"  解码器层数: {results['config']['model']['num_decoder_layers']}\n\n")

        f.write("性能指标:\n")
        f.write("          |   损失   |   准确率  |   困惑度  \n")
        f.write("-" * 45 + "\n")
        # f.write(
            # f"训练集   | {results['train']['loss']:8.4f} | {results['train']['accuracy']:7.2%} | {results['train']['perplexity']:8.2f}\n")
        f.write(
            f"验证集   | {results['validation']['loss']:8.4f} | {results['validation']['accuracy']:7.2%} | {results['validation']['perplexity']:8.2f}\n")
        f.write(
            f"测试集   | {results['test']['loss']:8.4f} | {results['test']['accuracy']:7.2%} | {results['test']['perplexity']:8.2f}\n\n")

        f.write(f"精确匹配率: {results['exact_match_rate']:.2f}%\n")

    logging.info(f"完整结果已保存到: {json_path}")
    logging.info(f"简洁报告已保存到: {txt_path}")


def find_latest_model(checkpoint_dir):
    """查找最新的模型文件"""
    checkpoint_path = Path(checkpoint_dir)
    model_files = list(checkpoint_path.glob("*.pth"))

    if not model_files:
        return None

    # 按修改时间排序，返回最新的
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    return str(latest_model)


def main():
    parser = argparse.ArgumentParser(description='Transformer Model Validation')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='配置文件路径')
    parser.add_argument('--model-path', type=str, default='checkpoints/best_model.pth', help='要验证的模型路径')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='检查点目录')

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

    # 数据预处理 - 这会重新构建分词器
    logging.info('加载数据和构建分词器...')
    data_processor = DataProcessor(config)
    # 重新获取数据加载器来构建分词器
    train_loader, val_loader, test_loader = data_processor.get_data_loaders()

    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_latest_model(args.checkpoint_dir)
        if not model_path:
            logging.error("没有找到模型文件，请使用 --model-path 指定模型路径")
            return
        logging.info(f"使用最新模型: {model_path}")

    # 进行验证
    results = validate_model(config, model_path, data_processor, device)

    # 打印结果
    print_results(results)

    # 保存结果
    save_results(results)


if __name__ == '__main__':
=======
#!/usr/bin/env python3
import torch
import logging
import argparse
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

from src.model import Transformer
from src.data_loader import DataProcessor
from src.utils import set_seed, setup_logging, load_config


class ModelValidator:
    def __init__(self, model, config, data_processor, device):
        self.model = model
        self.config = config
        self.data_processor = data_processor
        self.device = device
        self.model.eval()

    def calculate_loss_and_accuracy(self, data_loader):
        """计算数据集的平均损失和准确率"""
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        total_loss = 0
        total_tokens = 0
        correct_tokens = 0

        with torch.no_grad():
            for src, tgt in tqdm(data_loader, desc="计算损失和准确率"):
                src, tgt = src.to(self.device), tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
                tgt_mask = self._create_decoder_mask(tgt_input)

                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                # 计算准确率
                predictions = torch.argmax(output, dim=-1)
                mask = (tgt_output != 0)
                correct = (predictions == tgt_output) & mask

                batch_tokens = mask.sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                correct_tokens += correct.sum().item()

        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
        perplexity = np.exp(avg_loss)

        return avg_loss, accuracy, perplexity

    def calculate_exact_match(self, data_loader, num_batches=5):
        """计算精确匹配率"""
        exact_matches = 0
        total_samples = 0

        # 获取特殊token的ID
        pad_id = 0
        bos_id = self.data_processor.src_tokenizer.token_to_id("[BOS]")
        eos_id = self.data_processor.src_tokenizer.token_to_id("[EOS]")
        special_tokens = {pad_id, bos_id, eos_id}

        with torch.no_grad():
            batch_count = 0
            for src, tgt in data_loader:
                if batch_count >= num_batches:
                    break

                src, tgt = src.to(self.device), tgt.to(self.device)

                # 使用teacher forcing计算输出
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
                tgt_mask = self._create_decoder_mask(tgt_input)

                output = self.model(src, tgt_input, src_mask, tgt_mask)
                predictions = torch.argmax(output, dim=-1)

                # 检查精确匹配
                for i in range(predictions.size(0)):
                    pred_seq = predictions[i].cpu().numpy()
                    target_seq = tgt_output[i].cpu().numpy()

                    # 移除padding和特殊token
                    pred_tokens = [token for token in pred_seq if token not in special_tokens]
                    target_tokens = [token for token in target_seq if token not in special_tokens]

                    if len(pred_tokens) == len(target_tokens) and all(
                            p == t for p, t in zip(pred_tokens, target_tokens)):
                        exact_matches += 1
                    total_samples += 1

                batch_count += 1

        exact_match_rate = (exact_matches / total_samples) * 100 if total_samples > 0 else 0
        return exact_match_rate

    def _create_decoder_mask(self, tgt):
        """创建解码器掩码"""
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device=self.device), diagonal=1)).bool()
        tgt_mask = tgt_pad_mask & nopeak_mask
        return tgt_mask


def load_trained_model(config, model_path, data_processor, device):
    """加载训练好的模型"""
    logging.info(f"加载模型: {model_path}")

    # 获取词汇表大小
    src_vocab_size, tgt_vocab_size = data_processor.get_vocab_sizes()

    # 创建模型结构
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

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("从检查点加载模型状态")
    else:
        model.load_state_dict(checkpoint)
        logging.info("直接加载模型权重")

    logging.info("模型加载成功")
    return model


def validate_model(config, model_path, data_processor, device):
    """完整的模型验证流程"""
    # 加载模型
    model = load_trained_model(config, model_path, data_processor, device)

    # 创建验证器
    validator = ModelValidator(
        model=model,
        config=config,
        data_processor=data_processor,
        device=device
    )

    # 获取数据加载器
    train_loader, val_loader, test_loader = data_processor.get_data_loaders()

    results = {
        'model_path': model_path,
        'config': {
            'model': config['model'],
            'data': config['data']
        }
    }

    # # 计算训练集指标
    # logging.info("计算训练集指标...")
    # train_loss, train_accuracy, train_perplexity = validator.calculate_loss_and_accuracy(train_loader)
    # results['train'] = {
    #     'loss': train_loss,
    #     'accuracy': train_accuracy,
    #     'perplexity': train_perplexity
    # }

    # 计算验证集指标
    logging.info("计算验证集指标...")
    val_loss, val_accuracy, val_perplexity = validator.calculate_loss_and_accuracy(val_loader)
    results['validation'] = {
        'loss': val_loss,
        'accuracy': val_accuracy,
        'perplexity': val_perplexity
    }

    # 计算测试集指标
    logging.info("计算测试集指标...")
    test_loss, test_accuracy, test_perplexity = validator.calculate_loss_and_accuracy(test_loader)
    results['test'] = {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'perplexity': test_perplexity
    }

    # 计算精确匹配率（只在测试集上计算，减少计算量）
    logging.info("计算精确匹配率...")
    exact_match_rate = validator.calculate_exact_match(test_loader, num_batches=3)
    results['exact_match_rate'] = exact_match_rate

    return results


def print_results(results):
    """打印验证结果"""
    print("=" * 60)
    print("模型验证结果")
    print("=" * 60)
    print(f"模型路径: {results['model_path']}")

    print(f"\n模型配置:")
    print(f"  d_model: {results['config']['model']['d_model']}")
    print(f"  nhead: {results['config']['model']['nhead']}")
    print(f"  编码器层数: {results['config']['model']['num_encoder_layers']}")
    print(f"  解码器层数: {results['config']['model']['num_decoder_layers']}")

    print(f"\n性能指标:")
    print("          |   损失   |   准确率  |   困惑度  ")
    print("-" * 50)
    # print(
        # f"训练集   | {results['train']['loss']:8.4f} | {results['train']['accuracy']:7.2%} | {results['train']['perplexity']:8.2f}")
    print(
        f"验证集   | {results['validation']['loss']:8.4f} | {results['validation']['accuracy']:7.2%} | {results['validation']['perplexity']:8.2f}")
    print(
        f"测试集   | {results['test']['loss']:8.4f} | {results['test']['accuracy']:7.2%} | {results['test']['perplexity']:8.2f}")

    print(f"\n其他指标:")
    print(f"精确匹配率: {results['exact_match_rate']:.2f}%")


def save_results(results, save_dir="results"):
    """保存验证结果到文件"""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    # 保存JSON格式的完整结果
    json_path = save_path / "validation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 保存简洁的文本报告
    txt_path = save_path / "validation_report.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("模型验证报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"模型路径: {results['model_path']}\n\n")

        f.write("模型配置:\n")
        f.write(f"  d_model: {results['config']['model']['d_model']}\n")
        f.write(f"  nhead: {results['config']['model']['nhead']}\n")
        f.write(f"  编码器层数: {results['config']['model']['num_encoder_layers']}\n")
        f.write(f"  解码器层数: {results['config']['model']['num_decoder_layers']}\n\n")

        f.write("性能指标:\n")
        f.write("          |   损失   |   准确率  |   困惑度  \n")
        f.write("-" * 45 + "\n")
        # f.write(
            # f"训练集   | {results['train']['loss']:8.4f} | {results['train']['accuracy']:7.2%} | {results['train']['perplexity']:8.2f}\n")
        f.write(
            f"验证集   | {results['validation']['loss']:8.4f} | {results['validation']['accuracy']:7.2%} | {results['validation']['perplexity']:8.2f}\n")
        f.write(
            f"测试集   | {results['test']['loss']:8.4f} | {results['test']['accuracy']:7.2%} | {results['test']['perplexity']:8.2f}\n\n")

        f.write(f"精确匹配率: {results['exact_match_rate']:.2f}%\n")

    logging.info(f"完整结果已保存到: {json_path}")
    logging.info(f"简洁报告已保存到: {txt_path}")


def find_latest_model(checkpoint_dir):
    """查找最新的模型文件"""
    checkpoint_path = Path(checkpoint_dir)
    model_files = list(checkpoint_path.glob("*.pth"))

    if not model_files:
        return None

    # 按修改时间排序，返回最新的
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    return str(latest_model)


def main():
    parser = argparse.ArgumentParser(description='Transformer Model Validation')
    parser.add_argument('--config', type=str, default='configs/base.yaml', help='配置文件路径')
    parser.add_argument('--model-path', type=str, default='checkpoints/best_model.pth', help='要验证的模型路径')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='检查点目录')

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

    # 数据预处理 - 这会重新构建分词器
    logging.info('加载数据和构建分词器...')
    data_processor = DataProcessor(config)
    # 重新获取数据加载器来构建分词器
    train_loader, val_loader, test_loader = data_processor.get_data_loaders()

    # 确定模型路径
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = find_latest_model(args.checkpoint_dir)
        if not model_path:
            logging.error("没有找到模型文件，请使用 --model-path 指定模型路径")
            return
        logging.info(f"使用最新模型: {model_path}")

    # 进行验证
    results = validate_model(config, model_path, data_processor, device)

    # 打印结果
    print_results(results)

    # 保存结果
    save_results(results)


if __name__ == '__main__':
    main()
