import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Sparse Attention模块（Local Window Attention）
class SparseLocalAttention(nn.Module):
    def __init__(self, d_model, window_size=3):
        super(SparseLocalAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.window_size = window_size
        self.scale = np.sqrt(d_model)

    def forward(self, x):
        B, N, D = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        output = torch.zeros_like(Q)

        for i in range(N):
            # 计算邻居范围
            left = max(0, i - self.window_size)
            right = min(N, i + self.window_size + 1)
            q = Q[:, i:i+1, :]
            k = K[:, left:right, :]
            v = V[:, left:right, :]

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
            attn_probs = torch.softmax(attn_scores, dim=-1)
            output[:, i:i+1, :] = torch.matmul(attn_probs, v)

        return output


# 稀疏Transformer Encoder Layer
class SparseTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, window_size):
        super(SparseTransformerEncoderLayer, self).__init__()
        self.self_attn = SparseLocalAttention(d_model, window_size)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


# Sparse Transformer多分类模型
class SparseTransformerOrgMultiModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_layers, dim_feedforward, dropout, window_size=3):
        super(SparseTransformerOrgMultiModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            SparseTransformerEncoderLayer(d_model, dim_feedforward, dropout, window_size)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_dim)
        self.d_model = d_model

    def forward(self, src):
        src = src.permute(1, 0)
        src = self.embedding(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src)
        src = src.permute(1, 0, 2)
        output = torch.mean(src, dim=1)
        output = self.fc(output)
        return output
