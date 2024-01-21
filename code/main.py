import torch
from torch import nn
from code.DGADataset import DGATrueDataset, DGAFalseDataset
from torch.utils.data import DataLoader, random_split
# 所有可用模型
from model.cnn.cnn_torch import CNNModel
from model.lstm.lstm_torch import LSTMModel
from model.mit.mit_torch import MITModel
from model.ann.ann_torch import Net
from model.bilbohybrid.bilbohybrid_torch import BilBoHybridModel
# 所有工具类函数
from utils.engine import train
from utils.saveModel import SaveModel

NUM_EPOCHS = 5
BATCH_SIZE = 32
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def readData():
    pass


if __name__ == '__main__':
    # 返回训练集逻辑dataset
    print("获取数据集")
    dga_true_train_dataset = DGATrueDataset(f'../data/Benign', True)
    dga_false_train_dataset = DGAFalseDataset(f'../data/DGA/2016-09-19-dgarchive_full', True)

    dga_true_train_dataset_size = len(dga_true_train_dataset)
    dga_false_train_dataset_size = len(dga_false_train_dataset)

    # 划分数据集为80%训练集，20%验证集
    print("划分数据集")
    true_train_size = int(0.8 * dga_true_train_dataset_size)
    true_test_size = dga_true_train_dataset_size - true_train_size
    true_train_dataset, true_test_dataset = random_split(dga_true_train_dataset,
                                                         [true_train_size, true_test_size])
    false_train_size = int(0.8 * dga_false_train_dataset_size)
    false_test_size = dga_false_train_dataset_size - false_train_size
    false_train_dataset, false_test_dataset = random_split(dga_false_train_dataset,
                                                           [false_train_size, false_test_size])

    print("创建dataLoader")
    # 创建标签为false数据加载成的训练集
    false_train_loader = DataLoader(false_train_dataset, batch_size=32)
    false_test_loader = DataLoader(false_test_dataset, batch_size=32)
    # 创建标签为true数据加载成的训练集
    true_train_loader = DataLoader(true_train_dataset, batch_size=32)
    true_test_loader = DataLoader(true_test_dataset, batch_size=32)

    print(f"f确定模型,设备为: {device}")
    # 确定训练模型
    model = CNNModel(255, 10, 10)
    # 二分类函数损失函数和优化器
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.1)

    print("训练模型开始")
    # 训练模型，标签为False
    train(model=model,
          train_dataloader=false_train_loader,
          test_dataloader=false_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer,
          epochs=NUM_EPOCHS,
          device=device)
    # 训练模型，标签为True
    train(model=model,
          train_dataloader=true_train_loader,
          test_dataloader=true_test_loader,
          loss_fn=loss_fn,
          optimizer=optimizer,
          epochs=NUM_EPOCHS,
          device=device)
    print("训练模型结束")

    print("保存模型")
    SaveModel(model=model,
              target_dir="models",
              model_name="CNNModel.pth")

    pass
