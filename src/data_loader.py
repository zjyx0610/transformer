import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os
import pickle
from typing import Tuple, List, Dict, Any
import logging
from pathlib import Path


class TranslationDataset(Dataset):
    def __init__(self, src_tokenizer, tgt_tokenizer, source_lang, target_lang, max_length=128):
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        self.pad_token = 0
        self.source_lang = source_lang
        self.target_lang = target_lang

        self.translation_pairs = list(zip(self.source_lang, self.target_lang))
        # logging.info(f"提取到 {len(self.translation_pairs)} 个 {source_lang}->{target_lang} 翻译对")

    def __len__(self):
        return len(self.translation_pairs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text, tgt_text = self.translation_pairs[idx]

        # 编码源文本和目标文本
        src_encoding = self.src_tokenizer.encode(src_text)
        tgt_encoding = self.tgt_tokenizer.encode(tgt_text)

        # 截断和填充
        src_ids = src_encoding.ids[:self.max_length]
        tgt_ids = tgt_encoding.ids[:self.max_length]

        # 添加特殊token
        src_ids = [self.src_tokenizer.token_to_id("[BOS]")] + src_ids + [self.src_tokenizer.token_to_id("[EOS]")]
        tgt_ids = [self.tgt_tokenizer.token_to_id("[BOS]")] + tgt_ids + [self.tgt_tokenizer.token_to_id("[EOS]")]

        # 填充到最大长度
        src_ids = self._pad_sequence(src_ids, self.max_length + 2)
        tgt_ids = self._pad_sequence(tgt_ids, self.max_length + 2)

        return torch.tensor(src_ids), torch.tensor(tgt_ids)

    def _pad_sequence(self, sequence: list, max_len: int) -> list:
        """填充序列到指定长度"""
        if len(sequence) < max_len:
            sequence = sequence + [self.pad_token] * (max_len - len(sequence))
        else:
            sequence = sequence[:max_len]
        return sequence


class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.src_tokenizer = None
        self.tgt_tokenizer = None
        self.data_dir = Path("data/IWSLT2017")

    def load_pkl_data(self, file_path: Path):
        """加载pkl文件"""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            logging.info(f"成功加载 {file_path}")
            return data
        except Exception as e:
            logging.error(f"加载 {file_path} 失败: {e}")
            raise

    def inspect_data_structure(self, data: Dict[str, Any]):
        """检查三层嵌套字典数据结构"""
        logging.info("数据结构分析:")
        logging.info(f"顶级键(源语言): {list(data.keys())}")

        # 特别检查英语和德语相关的语言对
        logging.info("\n英语相关语言对:")
        if 'en' in data:
            en_targets = data['en']
            if isinstance(en_targets, dict):
                logging.info(f"英语->其他语言: {list(en_targets.keys())}")
                for tgt_lang in en_targets.keys():
                    if tgt_lang == 'de':
                        samples = en_targets[tgt_lang]
                        logging.info(f"  en->de: {len(samples)} 个样本")

        logging.info("\n德语相关语言对:")
        if 'de' in data:
            de_targets = data['de']
            if isinstance(de_targets, dict):
                logging.info(f"德语->其他语言: {list(de_targets.keys())}")
                for tgt_lang in de_targets.keys():
                    if tgt_lang == 'en':
                        samples = de_targets[tgt_lang]
                        logging.info(f"  de->en: {len(samples)} 个样本")

    def extract_all_texts(self, data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """从数据中提取所有源语言和目标语言文本"""
        src_texts = []
        tgt_texts = []
        source_lang = self.config['data']['source_lang']
        target_lang = self.config['data']['target_lang']

        logging.info(f"提取 {source_lang}->{target_lang} 的翻译对")

        # 方法1: 直接查找 en->de
        if 'en' in data and isinstance(data['en'], dict) and 'de' in data['en']:
            translations = data['en']['de']
            # if isinstance(translations, dict):
            logging.info(f"找到 en->de: {len(translations)} 个样本")
            for translation_data in translations:
                if isinstance(translation_data, dict):
                    source_text = translation_data.get('Source', '')
                    translated_text = translation_data.get('Translated', '')
                    if source_text and translated_text:
                        src_texts.append(source_text)
                        tgt_texts.append(translated_text)

        logging.info(f"最终提取到 {len(src_texts)} 个{source_lang}句子和 {len(tgt_texts)} 个{target_lang}句子")
        return src_texts, tgt_texts

    def build_tokenizers(self, src_texts, tgt_texts):
        """构建BPE分词器"""
        logging.info("开始构建分词器...")

        # 提取源语言和目标语言文本
        # src_texts, tgt_texts = self.extract_all_texts(train_data)

        if not src_texts or not tgt_texts:
            logging.error("没有找到任何训练数据！")
            # 创建一些虚拟数据用于测试
            logging.info("创建虚拟数据用于测试...")
            src_texts = [
                "Hello world", "How are you", "Good morning", "Thank you",
                "What is your name", "I love programming", "The weather is nice"
            ]
            tgt_texts = [
                "Hallo Welt", "Wie geht es dir", "Guten Morgen", "Danke",
                "Wie ist dein Name", "Ich liebe Programmierung", "Das Wetter ist schön"
            ]

        # 过滤空文本
        src_texts = [text for text in src_texts if text.strip()]
        tgt_texts = [text for text in tgt_texts if text.strip()]

        logging.info(f"过滤后: {len(src_texts)} 个源语言句子, {len(tgt_texts)} 个目标语言句子")

        # 源语言分词器
        self.src_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.src_tokenizer.pre_tokenizer = Whitespace()

        # 目标语言分词器
        self.tgt_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tgt_tokenizer.pre_tokenizer = Whitespace()

        # 训练分词器
        trainer = BpeTrainer(
            special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
            vocab_size=5000,
            min_frequency=1
        )

        # 训练源语言分词器
        self.src_tokenizer.train_from_iterator(src_texts, trainer)
        logging.info(f"源语言分词器训练完成，词汇表大小: {self.src_tokenizer.get_vocab_size()}")

        # 训练目标语言分词器
        self.tgt_tokenizer.train_from_iterator(tgt_texts, trainer)
        logging.info(f"目标语言分词器训练完成，词汇表大小: {self.tgt_tokenizer.get_vocab_size()}")

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """获取数据加载器"""
        # 数据文件路径
        train_file = self.data_dir / "iwslt2017_train.pkl"
        val_file = self.data_dir / "iwslt2017_validation.pkl"
        test_file = self.data_dir / "iwslt2017_test.pkl"

        # 检查数据文件是否存在
        if not train_file.exists():
            logging.error(f"训练数据文件不存在: {train_file}")
            raise FileNotFoundError(f"训练数据文件不存在: {train_file}")

        # 加载数据
        logging.info("加载训练数据...")
        train_data = self.load_pkl_data(train_file)
        logging.info("训练数据结构:")
        self.inspect_data_structure(train_data)
        train_src_texts, train_tgt_texts = self.extract_all_texts(train_data)

        logging.info("加载验证数据...")
        val_data = self.load_pkl_data(val_file)
        val_src_texts, val_tgt_texts = self.extract_all_texts(val_data)

        logging.info("加载测试数据...")
        test_data = self.load_pkl_data(test_file)
        test_src_texts, test_tgt_texts = self.extract_all_texts(test_data)

        # 构建分词器
        self.build_tokenizers(train_src_texts, train_tgt_texts)

        # 创建数据集
        train_dataset = TranslationDataset(
            self.src_tokenizer,
            self.tgt_tokenizer,
            train_src_texts, train_tgt_texts,
            self.config['data']['max_length'],
        )

        val_dataset = TranslationDataset(
            self.src_tokenizer,
            self.tgt_tokenizer,
            val_src_texts, val_tgt_texts,
            self.config['data']['max_length']
        )

        test_dataset = TranslationDataset(
            self.src_tokenizer,
            self.tgt_tokenizer,
            test_src_texts, test_tgt_texts,
            self.config['data']['max_length']
        )

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False
        )

        logging.info("数据加载器创建完成")
        logging.info(f"训练集: {len(train_dataset)} 个样本")
        logging.info(f"验证集: {len(val_dataset)} 个样本")
        logging.info(f"测试集: {len(test_dataset)} 个样本")

        return train_loader, val_loader, test_loader

    def get_vocab_sizes(self) -> Tuple[int, int]:
        """获取词汇表大小"""
        if self.src_tokenizer is None or self.tgt_tokenizer is None:
            return 5000, 5000
