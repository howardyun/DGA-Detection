import torch
from typing import Dict, List, Tuple
import string
from tqdm.auto import tqdm
from DataIterator import MultiDataIterator
from torch.utils.data import DataLoader

def train_multi_step(model: torch.nn.Module,
                     dataloader: torch.utils.data.DataLoader,
                     loss_fn: torch.nn.Module,
                     optimizer: torch.optim.Optimizer,
                     device: torch.device) -> Tuple[float, float]:
    """
    多分类
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
        # 预测,模型最后锁了softmax操作,这里就不再进行softmax
        # 仅仅进行argmax
        y_logits = model(X).squeeze()
        y_pred = y_logits.argmax(dim=1)

        # 计算和累积损失
        try:
            loss = loss_fn(y_logits, y)
            train_loss += loss.item()
            pass
        except:
            y_pred = torch.unsqueeze(y_logits, dim=0)
            loss = loss_fn(y_logits, y)
            train_loss += loss.item()
            pass

        #  多分类训练计算
        train_acc += torch.eq(y_pred, y).sum().item() / len(y_pred)

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


def test_multi_step(model: torch.nn.Module,
                    dataloader: torch.utils.data.DataLoader,
                    loss_fn: torch.nn.Module,
                    device: torch.device) -> Tuple[float, float]:
    """
    :param model: 要训练的pytorch模型
    :param dataloader: 训练模型dataLoader实例
    :param loss_fn: pytorch损失函数
    :param device: 目标设备,cuda或者cpu
    :return:
    """
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X).squeeze()
            test_pred_pred = test_pred_logits.argmax(dim=1)

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

            # 二分类训练计算
            test_acc += torch.eq(test_pred_pred, y).sum().item() / len(test_pred_pred)
            pass
        pass

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
    pass


def train_multi(model: torch.nn.Module,
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
        train_data_iterator = MultiDataIterator(train_file, chunksize=1000000)
        test_data_iterator = MultiDataIterator(test_file, chunksize=250000)

        # 非全数据集总量是data_iter * 上面设置的chunsize
        for data_chunk in train_data_iterator:
            train_loader = DataLoader(data_chunk, batch_size=BATCH_SIZE, shuffle=True)
            # 获取训练数据
            train_loss, train_acc = train_multi_step(model=model,
                                                     dataloader=train_loader,
                                                     loss_fn=loss_fn,
                                                     optimizer=optimizer,
                                                     device=device)
            pass
        for data_chunk in test_data_iterator:
            test_loader = DataLoader(data_chunk, batch_size=BATCH_SIZE, shuffle=True)
            # 获取测试数据
            test_loss, test_acc = test_multi_step(model=model,
                                                  dataloader=test_loader,
                                                  loss_fn=loss_fn,
                                                  device=device)
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