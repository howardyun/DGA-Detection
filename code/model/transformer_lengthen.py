import math

import torch
import torch.nn as nn


class DGAClassifier_lenthen(nn.Module):
    def __init__(self, input_vocab_size, embed_size, num_heads, num_encoder_layers, num_classes, max_len=255):
        super(DGAClassifier_lenthen, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.positional_encoding = self._generate_positional_encoding(max_len, embed_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.fc_out = nn.Linear(embed_size, num_classes)

    def _generate_positional_encoding(self, max_len, embed_size):
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
        positional_encoding = torch.zeros(max_len, embed_size)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding

    def forward(self, x):
        # 获取嵌入向量
        embeddings = self.embedding(x)

        # 添加位置编码
        positional_encoding = self.positional_encoding[:x.size(1), :].unsqueeze(0)
        embeddings = embeddings + positional_encoding

        # 生成填充掩码
        src_mask = self.generate_square_subsequent_mask(x.size(1))

        # 使用 nn.TransformerEncoder 实现变长注意力
        transformer_output = self.transformer_encoder(embeddings, src_key_padding_mask=src_mask)

        # 取第一个 token 的输出
        out = self.fc_out(transformer_output.mean(dim=0))  # 使用平均池化

        return out

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
