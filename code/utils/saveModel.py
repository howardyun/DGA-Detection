import csv
import datetime

import torch
from pathlib import Path
from datetime import datetime


def GetCurrentTime():
    """
    获取当前时间,用于成为文件夹名字
    :return:
    """
    # 获取当前年月日
    # 获取当前日期和时间
    current_datetime = datetime.now()
    current_year = current_datetime.year
    current_month = current_datetime.month
    current_day = current_datetime.day
    current_hour = current_datetime.hour
    year_str = str(current_year)
    month_str = str(current_month).zfill(2)
    day_str = str(current_day).zfill(2)
    hour_str = str(current_hour).zfill(2)

    return year_str + month_str + day_str + hour_str
    pass


# 保存模型工具函数
def SaveModel(model: torch.nn.Module, target_dir: str, model_name: str, lb_flag: bool):
    """
    :param model: 保存的模型
    :param target_dir: 目标目录
    :param model_name: 模型名.pth结尾
    :param lb_flag: 是否为鲁棒性测试,
    :return:
    """

    # 断言保证文件类型
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "模型结尾需为 'pt' or 'pth'"

    # 创建第一层目标文件夹
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
    # 创建第二层文件夹,如果是鲁棒性测试就是lb,不是就是dga
    if lb_flag:
        # 第二层路径
        second_dir = 'lb'
        second_dir_path = target_dir_path / second_dir
        second_dir_path.mkdir(parents=True, exist_ok=True)
        # 第三层路径
        third_dir = GetCurrentTime()
        third_dir_path = second_dir_path / third_dir
        third_dir_path.mkdir(parents=True, exist_ok=True)
        model_save_path = third_dir_path / model_name
        # 保存model模型参数字典
        print(f"保存模型路径: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)
        pass
    else:
        second_dir = 'dga'
        second_dir_path = target_dir_path / second_dir
        second_dir_path.mkdir(parents=True, exist_ok=True)
        third_dir = GetCurrentTime()
        third_dir_path = second_dir_path / third_dir
        third_dir_path.mkdir(parents=True, exist_ok=True)
        model_save_path = third_dir_path / model_name
        # 保存model模型参数字典
        print(f"保存模型路径: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)
        pass
    pass


# 加载模型参数函数
def LoadModel(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    :param model: 要加载参数的模型
    :param target_dir: 模型文件夹
    :param model_name: 模型名
    :return:
    """
    target_dir_path = Path(target_dir)
    # 断言保证文件类型
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "模型结尾需为 'pt' or 'pth'"
    model_load_path = target_dir_path / model_name

    model.load_state_dict(torch.load(f=model_load_path))
    return model
    pass


def SaveResults(model_name: str, model_epoch: int, results: dict, lb_flag: bool):
    """
    保存模型训练数据
    :param model_name: 模型名字
    :param model_epoch: 模型训练次数
    :param results: 结果集
    :param lb_flag: 是否为鲁棒性测试,鲁棒性存在lb文件夹,非鲁棒性存在dga文件夹
    :return:
    """
    # 创建模型记录文件夹
    first_dir_path = Path("../modelRecord/train")
    first_dir_path.mkdir(parents=True, exist_ok=True)

    # 路径为dga还是lb
    second_dir_path = Path("../modelRecord/train/lb") if lb_flag else Path("../modelRecord/train/dga")
    second_dir_path.mkdir(parents=True, exist_ok=True)

    # 日期文件夹
    third_dir = GetCurrentTime()
    third_dir_path = second_dir_path / third_dir
    third_dir_path.mkdir(parents=True, exist_ok=True)

    # 最终统计的csv文件
    csv_file = "record.csv"
    csv_file_path = third_dir_path / csv_file
    # 不存在建立文件
    if not Path(csv_file_path).exists():
        with open(str(csv_file_path), mode='w', newline="") as csvfile:
            pass
        pass

    # 写入标题
    csv_title = [model_name, "train_loss", "train_acc", "test_loss", "test_acc"]
    with open(str(csv_file_path), mode='a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_title)
        pass

    # 写入一行数据
    train_loss, train_acc, test_loss, test_acc = results['train_loss'], results['train_acc'], results['test_loss'], \
        results['test_acc']
    for index in range(model_epoch):
        csv_item = ["", train_loss[index], train_acc[index], test_loss[index], test_acc[index]]
        with open(str(csv_file_path), mode='a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            # 写入一行数据
            writer.writerow(csv_item)
            pass
        pass
    pass
