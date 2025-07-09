import torch
import torch.nn as nn
import numpy as np

# 位置编码不变
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


# Linear Attention模块
class LinearAttention(nn.Module):
    def __init__(self, d_model):
        super(LinearAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def feature_map(self, x):
        return torch.relu(x) + 1e-6  # 避免除0问题

    def forward(self, x):
        Q = self.feature_map(self.query(x))
        K = self.feature_map(self.key(x))
        V = self.value(x)
        KV = torch.einsum('bnd,bne->bde', K, V)
        Z = 1.0 / (torch.einsum('bnd,bd->bn', Q, K.sum(dim=1)) + 1e-6)
        output = torch.einsum('bnd,bde,bn->bne', Q, KV, Z)
        return output


# Linear Transformer Encoder Layer
class LinearTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(LinearTransformerEncoderLayer, self).__init__()
        self.self_attn = LinearAttention(d_model)
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


# Linear Transformer多分类模型
class LinearTransformerOrgMultiModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_layers, dim_feedforward, dropout):
        super(LinearTransformerOrgMultiModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            LinearTransformerEncoderLayer(d_model, dim_feedforward, dropout)
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
