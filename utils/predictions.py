import torch

# 设置设备无关
device = "cuda" if torch.cuda.is_available() else "cpu"


# 预测函数
def Predictions(model: torch.nn.Module,
                dga_domain: str,
                device: torch.device = device
                ):
    """
    :param model: 预测是用的模型
    :param dga_domain: 进行预测的域名
    :param device: 设备
    :return:
    """
    # 设备设置模型
    model.to(device)

    # 打开模型评估模式和推理模式
    model.eval()

    with torch.inference_mode():
        # 预测标签
        target_pred = model(dga_domain.to(device))
        # target_pred = model(dga_domain)
        pass

    target_pred_probs = torch.softmax(target_pred, dim=1)
    target_pred_label = torch.argmax(target_pred_probs, dim=1)

    return target_pred_label
    pass
