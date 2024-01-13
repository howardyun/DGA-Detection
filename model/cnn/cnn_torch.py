import torch
import torch.nn as nn
import torch.nn.functional as F

# 设定基准参数，按照需求改变，初始全设为0防止报错
MAX_INDEX = 0
MAX_STRING_LENGTH = 0
max_index = MAX_INDEX
max_string_length = MAX_STRING_LENGTH
EMBEDDING_DIMENSION = 0
NUM_CONV_FILTERS = 0


class CNNModel(nn.Module):
    def __init__(self, max_index, max_string_length, embedding_dimension=EMBEDDING_DIMENSION,
                 num_conv_filters=NUM_CONV_FILTERS):
        """
        :param num_conv_filters: 卷积神经网络输出空间维度
        """
        super(CNNModel, self).__init__()

        # Embedding层
        self.embeddingCNN = nn.Embedding(num_embeddings=max_index,
                                         embedding_dim=embedding_dimension,
                                         padding_idx=0)

        # 五层平行的卷积神经网络，卷积核不同
        self.conv2 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=4)
        self.conv5 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=5)
        self.conv6 = nn.Conv1d(in_channels=embedding_dimension, out_channels=num_conv_filters,
                               kernel_size=6)

        # 全局池化层，kersa中使用的GlobalMaxPool1D求取最大值，但前面输入的卷积神经网络是一维的，最终得到一维最大数，
        # 因此可用torch的AdaptiveMaxPool1d(1)达到相同目的
        self.pool2 = nn.AdaptiveMaxPool1d(1)
        self.pool3 = nn.AdaptiveMaxPool1d(1)
        self.pool4 = nn.AdaptiveMaxPool1d(1)
        self.pool5 = nn.AdaptiveMaxPool1d(1)
        self.pool6 = nn.AdaptiveMaxPool1d(1)

        # 全连接层
        self.densecnn = nn.Linear(num_conv_filters * 5, num_conv_filters)
        self.dropoutcnnmid = nn.Dropout(0.5)
        self.dropoutcnn = nn.Dropout(0.5)

        # 输出层
        self.output = nn.Linear(num_conv_filters, 1)
        pass

    def forward(self, x):
        # 输入
        x = self.embeddingCNN(x)

        # 五层卷积层，这里可能要改改，pytorch希望channel在最后一个维度
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x4 = F.relu(self.conv4(x))
        x5 = F.relu(self.conv5(x))
        x6 = F.relu(self.conv6(x))

        # 池化层
        x2 = self.pool2(x2).squeeze()
        x3 = self.pool3(x3).squeeze()
        x4 = self.pool4(x4).squeeze()
        x5 = self.pool5(x5).squeeze()
        x6 = self.pool6(x6).squeeze()

        # 模态融合
        x = torch.cat([x2, x3, x4, x5, x6], dim=1)

        # 全连接层
        x = self.dropoutcnnmid(x)
        x = F.relu(self.densecnn(x))
        x = self.dropoutcnn(x)

        # 激活，输出
        x = torch.sigmoid(self.output(x))

        return x
        pass

    pass
