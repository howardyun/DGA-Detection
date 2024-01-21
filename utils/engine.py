import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Dict, List, Tuple


# 单步模型训练函数
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """
    :param model: 要训练的pytorch模型
    :param dataloader: 训练模型dataLoader实例
    :param loss_fn: 最小化pytorch损失函数
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
        y_pred = model(X)

        # 计算和累积损失
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 优化器设置零梯度
        optimizer.zero_grad()

        # 反向求导
        loss.backward()

        # 优化器步进
        optimizer.step()

        # 计算并累积所有批次的准确性指标
        # 这里可能要改改,因为是二分类sigmoid激化之后,可能不需要argmax
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        pass

    # 调整指标以获得每个批次的平均损失和准确性
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc
    pass


# 单步模型测试函数
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """
    :param model: 要训练的pytorch模型
    :param dataloader: 训练模型dataLoader实例
    :param loss_fn: 最小化pytorch损失函数
    :param device: 目标设备,cuda或者cpu
    :return:
    """
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 这里可能要改改,因为是二分类sigmoid激化之后,可能不需要argmax
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))
            pass
        pass

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
    pass


# 主要训练函数
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
    """
    :param model: pytorch模型
    :param train_dataloader: 训练模型dataLoader
    :param test_dataloader: 测试模型dataLoader
    :param optimizer: 优化器
    :param loss_fn: 损失函数
    :param epochs: 训练次数
    :param device: 目标设备
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

    # 循环训练
    for epoch in tqdm(range(epochs)):
        # 获取训练数据
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        # 获取测试数据
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)

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
