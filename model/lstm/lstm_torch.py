import torch
import torch.nn as nn
import torch.nn.functional as F

# 设定基准参数，按照需求改变，初始全设为0防止报错
MAX_INDEX = 0
MAX_STRING_LENGTH = 0
EMBEDDING_DIMENSION = 0


class LSTMModel(nn.Module):
    def __init__(self, max_index, embedding_dimension=EMBEDDING_DIMENSION, hidden_size=256):
        """
        :param hidden_size: 隐藏层用来规定LSTM的
        """
        super(LSTMModel, self).__init__()

        # Embedding层
        self.embedding = nn.Embedding(num_embeddings=max_index,
                                      embedding_dim=embedding_dimension,
                                      padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dimension,
                            hidden_size=hidden_size,
                            batch_first=True)

        # 防止过拟合
        self.dropout = nn.Dropout(0.5)

        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)
        pass

    def forward(self, x):
        # 输入
        x = self.embedding(x)

        # LSTM层
        lstm_out, _ = self.lstm(x)

        # 选择LSTM时序中最后一此输出
        lstm_out = lstm_out[:, -1, :]

        # 防止过拟合
        x = self.dropout(lstm_out)

        # 全连接层
        x = torch.sigmoid(self.fc(x))

        return x
        pass

    pass
