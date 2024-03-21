import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

dcfrom DGADataset import DGATrueDataset, DGAFalseDataset
from utils.engine import train

NUM_EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(TransformerModel, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True),
            num_layers
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Input x shape expected: [batch_size, seq_length]
        embedded = self.embedding(x)  # Output shape: [batch_size, seq_length, hidden_dim]
        transformed = self.transformer(embedded)  # Output shape: [batch_size, seq_length, hidden_dim]
        output = self.fc(transformed.mean(dim=1))  # Output shape: [batch_size, output_dim]
        return output


# 定义自定义数据集类
class DGADataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        return x, y


if __name__ == '__main__':

    # 准备训练数据和标签
    print("获取数据集")
    dga_true_train_dataset = DGATrueDataset(f'../data/Benign', True)
    dga_false_train_dataset = DGAFalseDataset(f'../data/DGA/2016-09-19-dgarchive_full', True)
    # 合并正样本和负样本数据集
    combined_dataset = ConcatDataset([dga_true_train_dataset, dga_false_train_dataset])

    # 获取合并后数据集的大小
    combined_dataset_size = len(combined_dataset)
    print(f"合并后的数据集大小: {combined_dataset_size}")

    # 打乱合并后的数据集顺序
    indices = torch.randperm(combined_dataset_size)
    combined_dataset = torch.utils.data.Subset(combined_dataset, indices)

    # 划分数据集为80%训练集，20%验证集
    print("划分数据集")
    train_size = int(0.8 * combined_dataset_size)
    test_size = combined_dataset_size - train_size

    # 分割训练集和验证集
    print("划分train和test")
    train_dataset, test_dataset = torch.utils.data.random_split(combined_dataset, [train_size, test_size])

    print("创建dataLoader")
    # 创建训练集和验证集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"确定模型,设备为: {device}")

    # 定义模型超参数
    input_dim = 255  # 输入维度，例如DGA域名的向量维度
    output_dim = 2  # 输出类别数量
    hidden_dim = 256  # 隐层维度
    num_layers = 2  # Transformer层数
    num_heads = 4  # Transformer头数

    # 创建Transformer模型实例
    model = TransformerModel(input_dim, output_dim, hidden_dim, num_layers, num_heads)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    # 训练模型
    num_epochs = 5
    for epoch in range(num_epochs):
        for batch_data, batch_labels in train_loader:
            # 前向传播
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 打印每个epoch的损失
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # 使用训练好的模型进行预测
    test_data = [...]  # 测试数据
    # test_dataset = DGADataset(test_data, labels=None)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")

    # 根据predicted进行相应的处理，例如输出预测结果或保存到文件等
