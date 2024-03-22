import csv
import os
import string
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from itertools import islice
from DataIterator import DataIterator, MultiDataIterator
from DGADataset_ysx import DGATrueDataset_ysx


# 单步模型训练函数
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    :param model: 要训练的pytorch模型
    :param dataloader: 训练模型dataLoader实例
    :param loss_fn: pytorch损失函数
    :param optimizer: pytorch优化器
    :param device: 目标设备,cuda或者cpu
    :return:
    """
    # 模型进入训练模式
    model.train()

    # 训练损失值和训练准确值
    train_loss, train_acc = 0, 0

    # 抽取dataLoader中的数据
    for batch, (X, y) in enumerate(dataloader):
        # 设备无关代码
        X, y = X.to(device), y.to(device)

        # 预测
        y_pred = model(X).squeeze()
        y = y.float()

        # 计算和累积损失
        try:
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            pass
        except:
            y_pred = torch.unsqueeze(y_pred, dim=0)
            loss = loss_fn(y_pred, y)
            train_loss += loss.item()
            pass

        # 这里没再次sigmoid，模型中已经激化过
        # 二分类训练计算
        y_label = torch.round(y_pred)
        train_acc += torch.eq(y_label, y).sum().item() / len(y_label)

        # 优化器设置零梯度
        optimizer.zero_grad()

        # 反向求导
        loss.backward()

        # 优化器步进
        optimizer.step()

        pass

    # 调整指标以获得每个批次的平均损失和准确性
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
    pass


# 模型名
global_model_name = ""
# 训练源文件
global_test_file = ""
# 训练结果
global_target_dir = ""
# 又训练结果产生的F1
global_acc_pre_f1_file_path = ""


def initResult(init_model_name: str, init_target_dir: str, init_acc_pre_f1_file_path: str, init_test_file: str):
    """
    # 初始化全局变量
    :return:
    """
    global global_model_name, global_test_file, global_target_dir, global_acc_pre_f1_file_path
    global_model_name = init_model_name
    global_test_file = init_test_file
    global_target_dir = init_target_dir
    global_acc_pre_f1_file_path = init_acc_pre_f1_file_path
    pass


def SaveResults(save_model_name, y, label, target_path):
    """
    存放训练结果
    """
    # 每个模型预测结果文件路径
    file_path = str(target_path) + '/' + save_model_name + '.csv'
    # 写入文件
    if not Path(file_path).exists():
        with open(str(file_path), mode='w', newline="") as csvfile:
            pass
        pass
    # 判断文件大小
    if os.path.getsize(file_path) == 0:
        # 写入标题
        df = pd.DataFrame(columns=['label', save_model_name], index=None)
        df.to_csv(file_path, index=False)
        # 不使用pandas进行插入,高内存低io,用csv插入,低内存高io
        for y, label in zip(y, label):
            row_item = [y.item(), label.item()]
            with open(str(file_path), mode='a', newline="", errors='ignore') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_item)
                pass
            pass
        pass
    else:
        for y, label in zip(y, label):
            row_item = [y.item(), label.item()]
            with open(str(file_path), mode='a', newline="", errors='ignore') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_item)
                pass
            pass
        pass
    return file_path
    pass


def SaveAccPreF1(epoch: int, model_name: str, data_file_path: str, results_path: str, acc_pre_f1_path: str):
    # 不存在建立文件
    if not Path(acc_pre_f1_path).exists():
        with open(str(acc_pre_f1_path), mode='w', newline="") as csvfile:
            # 写入标题
            csv_title = ['model_name', 'epoch', 'file_name', 'model_accuracy', 'model_precision', 'model_recall',
                         'model_f1']
            with open(str(acc_pre_f1_path), mode='a', newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(csv_title)
                pass
            pass
        pass

    # 数据帧
    # 计算准确率精准率f1
    df = pd.read_csv(results_path)
    # 变成tensor
    tensor_label = torch.tensor(df['label'].to_numpy())
    tensor_pred = torch.tensor(df[model_name].to_numpy())
    # 用sklearn库直接计算
    accuracy = accuracy_score(tensor_label, tensor_pred)
    precision = precision_score(tensor_label, tensor_pred, zero_division=0)
    recall = recall_score(tensor_label, tensor_pred, zero_division=0)
    f1 = f1_score(tensor_label, tensor_pred, zero_division=0)

    # 写入
    csv_item = [model_name, f"epoch: {epoch + 1}", data_file_path, accuracy, precision, recall, f1]
    with open(str(acc_pre_f1_path), mode='a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        # 写入一行数据
        writer.writerow(csv_item)
        pass
    pass


# 单步模型测试函数
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device):
    """
    :param model: 要训练的pytorch模型
    :param dataloader: 训练模型dataLoader实例
    :param loss_fn: pytorch损失函数
    :param device: 目标设备,cuda或者cpu
    :return:
    """
    model.eval()

    test_loss, test_acc = 0, 0
    results_path = ""

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X).squeeze()
            y = y.float()

            # 处理tensor张量计算失误问题
            try:
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                pass
            except:
                test_pred_logits = torch.unsqueeze(test_pred_logits, dim=0)
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.item()
                pass

            # 这里没再次sigmoid，模型中已经激化过
            # 二分类训练计算
            test_label = torch.round(test_pred_logits)
            test_acc += torch.eq(test_label, y).sum().item() / len(test_label)

            # 写入结果
            if global_model_name and global_target_dir:
                results_path = SaveResults(save_model_name=global_model_name, y=y.cpu(), label=test_label.cpu(),
                                           target_path=global_target_dir)
                pass
            pass
        pass

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, results_path
    pass


# 主要训练函数
def train_ysx(model: torch.nn.Module,
              train_file: string,
              test_file: string,
              optimizer: torch.optim.Optimizer,
              loss_fn: torch.nn.Module,
              epochs: int,
              device: torch.device,
              BATCH_SIZE: int) -> Dict[str, List]:
    """
    :param model: pytorch模型
    :param train_file: 训练数据文件
    :param test_file: 测试数据文件
    :param optimizer: 优化器
    :param loss_fn: 损失函数
    :param epochs: 训练次数
    :param device: 目标设备
    :param BATCH_SIZE: 训练批次
    :return:
    """
    # 最终需要的准确率数据
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
               }

    # 设备无关代码
    model.to(device)
    train_loss, train_acc = 0, 0
    test_loss, test_acc = 0, 0
    # 循环训练
    for epoch in tqdm(range(epochs)):
        # 优化训练集和测试集读取，都采用迭代器读取，原因是全数据训练集四千万+，测试集一千万+
        # 最终迭代器步进因改为训练集一百万一次，测试集二十五万一次
        # 这个迭代器对象不可重置读取位置，只能重新创建充值读取位置
        train_data_iterator = DataIterator(train_file, chunksize=1000000)
        test_data_iterator = DataIterator(test_file, chunksize=250000)

        # data_flag是True时全数据集，False时非全数据集
        # 非全数据集总量是data_iter * 上面设置的chunsize
        for data_chunk in train_data_iterator:
            train_loader = DataLoader(data_chunk, batch_size=BATCH_SIZE, shuffle=True)
            # 获取训练数据
            train_loss, train_acc = train_step(model=model,
                                               dataloader=train_loader,
                                               loss_fn=loss_fn,
                                               optimizer=optimizer,
                                               device=device)
            pass
        for data_chunk in test_data_iterator:
            test_loader = DataLoader(data_chunk, batch_size=BATCH_SIZE, shuffle=True)
            # 获取测试数据
            test_loss, test_acc, results_path = test_step(model=model,
                                                          dataloader=test_loader,
                                                          loss_fn=loss_fn,
                                                          device=device)
            # 计算最终结果
            # 一个模型预测结果
            if global_model_name and global_target_dir:
                SaveAccPreF1(epoch=epoch, model_name=global_model_name, data_file_path=global_test_file,
                             results_path=results_path,
                             acc_pre_f1_path=global_acc_pre_f1_file_path)
                pass
            pass
        # 每轮信息
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        # 数据加入数据字典
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        pass

    # 返回最终数据
    return results
    pass
