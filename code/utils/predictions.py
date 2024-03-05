import csv
from datetime import datetime
from pathlib import Path

from DataIterator import DataIterator
import pandas as pd
import torch
from torch.utils.data import DataLoader
from code.DGADataset_ysx import DGATrueDataset_ysx


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


def SavePredictionsResults(results: dict, lb_flag: bool):
    """
    :param results: 结果集
    :param lb_flag: 是否为鲁棒性标识
    :return:
    """
    # 创建模型记录文件夹
    first_dir_path = Path("../modelRecord/predict")
    first_dir_path.mkdir(parents=True, exist_ok=True)

    # 路径为dga还是lb
    second_dir_path = Path("../modelRecord/predict/lb") if lb_flag else Path("../modelRecord/predict/dga")
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
    csv_title = ['predict model', 'predict name', "predict acc"]
    with open(str(csv_file_path), mode='a', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_title)
        pass

    # 写入一行数据
    model_name_list, file_name_list, predict_acc_list = results['predict model'], results['predict name'], results[
        'predict acc']
    for index in range(len(results['predict name'])):
        csv_item = [model_name_list[index], file_name_list[index], predict_acc_list[index]]
        with open(str(csv_file_path), mode='a', newline="") as csvfile:
            writer = csv.writer(csvfile)
            # 写入一行数据
            writer.writerow(csv_item)
            pass
        pass
    pass


# 预测函数
def Predictions(model: torch.nn.Module,
                model_name: str,
                file: str,
                device: torch.device,
                full_flag: bool,
                BATCH_SIZE: int,
                partial_data=1000):
    """
    :param model: 预测是用的模型
    :param model_name: 模型名字
    :param file: 预测文件路径
    :param device: 设备
    :param full_flag: 全数据集标志
    :param BATCH_SIZE: 批次数量
    :param partial_data: 非全数据集数据
    :return:
    """
    results = {
        "predict model": [],
        "predict name": [],
        "predict acc": [],
    }
    # 预测数据的dataLoader
    predict_df = pd.read_csv(file) if full_flag else pd.read_csv(file, nrows=partial_data)
    dataloader = DataLoader(DGATrueDataset_ysx(predict_df, True), batch_size=BATCH_SIZE, shuffle=True)

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
            try:
                pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                pass
            except:
                pred_label = pred_label.unsqueeze(0)
                pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                pass
            pass
        pass

    pred_acc = pred_acc / len(dataloader)
    results["predict model"].append(model_name)
    results["predict name"].append(file)
    results["predict acc"].append(pred_acc)
    return results
    pass


def FindRow(csv_file, text):
    """
    :param csv_file: 查找文本的文件
    :param text: 查找文本
    """
    index = 0
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        # 获取索引
        for i, row in enumerate(reader):
            if text in row:
                index = i
            pass
        pass

    file_list = []
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > index:
                file_list.append(row[0].replace("../../", "../"))
                pass
            pass
        pass
    return file_list
    pass


def PredictionFamily(model: torch.nn.Module,
                     model_name: str,
                     file: str,
                     device: torch.device,
                     full_flag: bool,
                     BATCH_SIZE: int,
                     partial_data=1000):
    """
    预测dga家族
    :param model: 预测是用的模型
    :param file: 预测文件路径
    :param device: 设备
    :param full_flag: 家族预测是否用全数据集
    :param BATCH_SIZE: 批次数量
    :param partial_data: 非全数据集时需要的数据
    :return:
    """
    file_list = FindRow(file, "predict file")
    # 最终返回结果集
    results = {
        "predict model": [],
        "predict name": [],
        "predict acc": [],
    }
    if full_flag:
        # 全数据集
        for file in file_list:
            # 配置数据
            predict_df = pd.read_csv(file)
            dataloader = DataLoader(DGATrueDataset_ysx(predict_df, True), batch_size=BATCH_SIZE, shuffle=True)

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
                    try:
                        pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                        pass
                    except:
                        pred_label = pred_label.unsqueeze(0)
                        pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                        pass
                    pass
                pass

            # 家族预测的准确率
            pred_acc = pred_acc / len(dataloader)
            results["predict model"].append(model_name)
            results["predict name"].append(file)
            results["predict acc"].append(pred_acc)
            pass
        pass
    else:
        # 部分数据集
        for file in file_list:
            # 配置数据
            # 非全数据集
            predict_df = pd.read_csv(file, nrows=partial_data)
            dataloader = DataLoader(DGATrueDataset_ysx(predict_df, True), batch_size=BATCH_SIZE, shuffle=True)

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
                    try:
                        pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                        pass
                    except:
                        pred_label = pred_label.unsqueeze(0)
                        pred_acc += torch.eq(pred_label, y).sum().item() / len(pred_label)
                        pass
                    pass
                pass

            # 家族预测的准确率
            pred_acc = pred_acc / len(dataloader)
            results["predict model"].append(model_name)
            results["predict name"].append(file)
            results["predict acc"].append(pred_acc)
            pass
        pass

    return results
    pass
