<<<<<<< HEAD
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


class TransformerTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 train_loader, val_loader, device: torch.device):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 优化器
        self.optimizer = Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=config['training']['betas'],
            eps=config['training']['eps'],
            weight_decay=config['training']['weight_decay']
        )

        # 学习率调度器
        self.lr_scheduler = self._get_lr_scheduler()

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # 创建保存目录
        os.makedirs(config['system']['save_dir'], exist_ok=True)

    def _get_lr_scheduler(self):
        """使用余弦退火调度器"""
        total_steps = len(self.train_loader) * self.config['training']['epochs']

        # 先预热，然后余弦退火
        warmup_scheduler = LambdaLR(self.optimizer,
                                    lambda step: min(1.0, step / self.config['training']['warmup_steps']))
        cosine_scheduler = CosineAnnealingLR(self.optimizer,
                                             T_max=total_steps - self.config['training']['warmup_steps'])

        # 组合调度器
        from torch.optim.lr_scheduler import SequentialLR
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config['training']['warmup_steps']]
        )

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)

            # 准备输入和目标
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 创建掩码
            src_mask, tgt_mask = self._create_masks(src, tgt_input)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)

            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['clip_grad'])

            # 更新参数
            self.optimizer.step()
            self.lr_scheduler.step()

            # 记录损失
            batch_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
                self.learning_rates.append(current_lr)

        return total_loss / total_tokens if total_tokens > 0 else 0

    def validate(self) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask, tgt_mask = self._create_masks(src, tgt_input)

                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                batch_tokens = (tgt_output != 0).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        return total_loss / total_tokens if total_tokens > 0 else 0

    def _create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建注意力掩码"""
        # 源序列掩码 (padding mask)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # 目标序列掩码 (padding mask + future mask)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device=self.device), diagonal=1)).bool()
        tgt_mask = tgt_pad_mask & nopeak_mask

        return src_mask, tgt_mask

    def train(self):
        """训练模型"""
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        logging.info("开始训练...")

        for epoch in range(self.config['training']['epochs']):
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            logging.info(f'Epoch {epoch + 1}/{self.config["training"]["epochs"]}: '
                         f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= patience:
                logging.info(f'Early stopping at epoch {epoch + 1}')
                break

            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

        # 保存最终模型
        self.save_checkpoint('final_model.pth')

        # 绘制训练曲线
        self.plot_training_curves()

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        save_path = os.path.join(self.config['system']['save_dir'], filename)
        torch.save(checkpoint, save_path)
        logging.info(f'模型已保存到: {save_path}')

    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # 学习率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['system']['save_dir'], 'training_curves.png'))
=======
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import logging
import os
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np


class TransformerTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 train_loader, val_loader, device: torch.device):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # 优化器
        self.optimizer = Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            betas=config['training']['betas'],
            eps=config['training']['eps'],
            weight_decay=config['training']['weight_decay']
        )

        # 学习率调度器
        self.lr_scheduler = self._get_lr_scheduler()

        # 损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []

        # 创建保存目录
        os.makedirs(config['system']['save_dir'], exist_ok=True)

    def _get_lr_scheduler(self):
        """使用余弦退火调度器"""
        total_steps = len(self.train_loader) * self.config['training']['epochs']

        # 先预热，然后余弦退火
        warmup_scheduler = LambdaLR(self.optimizer,
                                    lambda step: min(1.0, step / self.config['training']['warmup_steps']))
        cosine_scheduler = CosineAnnealingLR(self.optimizer,
                                             T_max=total_steps - self.config['training']['warmup_steps'])

        # 组合调度器
        from torch.optim.lr_scheduler import SequentialLR
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config['training']['warmup_steps']]
        )

    def train_epoch(self) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        for batch_idx, (src, tgt) in enumerate(self.train_loader):
            src, tgt = src.to(self.device), tgt.to(self.device)

            # 准备输入和目标
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 创建掩码
            src_mask, tgt_mask = self._create_masks(src, tgt_input)

            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input, src_mask, tgt_mask)

            # 计算损失
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['clip_grad'])

            # 更新参数
            self.optimizer.step()
            self.lr_scheduler.step()

            # 记录损失
            batch_tokens = (tgt_output != 0).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logging.info(f'Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
                self.learning_rates.append(current_lr)

        return total_loss / total_tokens if total_tokens > 0 else 0

    def validate(self) -> float:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for src, tgt in self.val_loader:
                src, tgt = src.to(self.device), tgt.to(self.device)

                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                src_mask, tgt_mask = self._create_masks(src, tgt_input)

                output = self.model(src, tgt_input, src_mask, tgt_mask)
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                batch_tokens = (tgt_output != 0).sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        return total_loss / total_tokens if total_tokens > 0 else 0

    def _create_masks(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """创建注意力掩码"""
        # 源序列掩码 (padding mask)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        # 目标序列掩码 (padding mask + future mask)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_len = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_len, seq_len, device=self.device), diagonal=1)).bool()
        tgt_mask = tgt_pad_mask & nopeak_mask

        return src_mask, tgt_mask

    def train(self):
        """训练模型"""
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        logging.info("开始训练...")

        for epoch in range(self.config['training']['epochs']):
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # 验证
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']

            logging.info(f'Epoch {epoch + 1}/{self.config["training"]["epochs"]}: '
                         f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}')

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1

            # 早停
            if patience_counter >= patience:
                logging.info(f'Early stopping at epoch {epoch + 1}')
                break

            # 每10个epoch保存一次检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')

        # 保存最终模型
        self.save_checkpoint('final_model.pth')

        # 绘制训练曲线
        self.plot_training_curves()

    def save_checkpoint(self, filename: str):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        save_path = os.path.join(self.config['system']['save_dir'], filename)
        torch.save(checkpoint, save_path)
        logging.info(f'模型已保存到: {save_path}')

    def plot_training_curves(self):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')

        # 学习率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.learning_rates)
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')

        plt.tight_layout()
        plt.savefig(os.path.join(self.config['system']['save_dir'], 'training_curves.png'))
>>>>>>> ea7271bb0dd6cc6e23ef1476823578defda08b95
        plt.close()