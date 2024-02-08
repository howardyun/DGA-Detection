import torch
from torch.utils.data import DataLoader


# 预测函数
def Predictions(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
                device: torch.device
                ):
    """
    :param model: 预测是用的模型
    :param dataloader: 进行预测的域名
    :param device: 设备
    :return:
    """
    # 设备设置模型
    model.to(device)

    # 打开模型评估模式和推理模式
    model.eval()

    # 评估预测准确率
    pred_acc = 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            pred_logits = model(X).squeeze()
            y = y.float()

            # 这里没再次sigmoid，模型中已经激化过
            # 二分类训练计算
            pred_label = torch.round(pred_logits)
            pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
            pass
        pass

    pred_acc = pred_acc / len(dataloader)
    return pred_acc
    pass
