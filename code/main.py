import pandas as pd
from code.DGADataset import DGATrueDataset, DGAFalseDataset
from torch.utils.data import DataLoader, random_split


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
    # 创建标签为true数据加载成的训练集
    true_train_loader = DataLoader(true_train_dataset, batch_size=32)

    pass
