import torch
from pathlib import Path


# 保存模型工具函数
def SaveModel(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    :param model: 保存的模型
    :param target_dir: 目标目录
    :param model_name: 模型名.pth结尾
    :return:
    """
    # 创建文件夹
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # 断言保证文件类型
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "模型结尾需为 'pt' or 'pth'"
    model_save_path = target_dir_path / model_name

    # 保存model模型参数字典
    print(f"保存模型路径: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

    pass
