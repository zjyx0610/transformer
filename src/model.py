import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        # 线性变换并分头
        Q = self.w_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 输出投影
        output = self.w_o(attn_output)

        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接 + 层归一化
        src2 = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(src2))

        # 前馈网络 + 残差连接 + 层归一化
        src2 = self.ffn(src)
        src = self.norm2(src + self.dropout(src2))

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout, activation)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接 + 层归一化
        tgt2 = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))

        # 交叉注意力 + 残差连接 + 层归一化
        tgt2 = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))

        # 前馈网络 + 残差连接 + 层归一化
        tgt2 = self.ffn(tgt)
        tgt = self.norm3(tgt + self.dropout(tgt2))

        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt


class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int,
                 d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: str = "relu", max_seq_length: int = 128):
        super().__init__()

        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 位置编码
        self.register_buffer('pos_encoding', self._get_positional_encoding(max_seq_length, d_model))

        # 编码器
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        # 解码器
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

        # 输出层
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

        self.dropout = nn.Dropout(dropout)

        # 初始化参数
        self._init_parameters()

    def _get_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """生成位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_parameters(self):
        """初始化参数"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:

        # 源语言嵌入 + 位置编码
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = src_embedded + self.pos_encoding[:, :src.size(1), :]
        src_embedded = self.dropout(src_embedded)

        # 编码器
        memory = self.encoder(src_embedded, src_mask)

        # 目标语言嵌入 + 位置编码
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = tgt_embedded + self.pos_encoding[:, :tgt.size(1), :]
        tgt_embedded = self.dropout(tgt_embedded)

        # 解码器
        output = self.decoder(tgt_embedded, memory, tgt_mask, src_mask)

        # 输出层
        logits = self.output_layer(output)

        return logits

    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """编码源序列"""
        src_embedded = self.src_embedding(src) * math.sqrt(self.d_model)
        src_embedded = src_embedded + self.pos_encoding[:, :src.size(1), :]
        return self.encoder(src_embedded, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: Optional[torch.Tensor] = None,
               memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """解码目标序列"""
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = tgt_embedded + self.pos_encoding[:, :tgt.size(1), :]
        output = self.decoder(tgt_embedded, memory, tgt_mask, memory_mask)
