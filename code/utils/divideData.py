"""
时间: 2026-3-4
作用: 切分2024年DGA数据工具类
"""

import os
import csv
import glob
import random
import pandas as pd
from pathlib import Path


def Benign_Malicious_Dataset(benign_catalog, malicious_catalog, sample_num, output_catalog):
    """
    获取最终数据集
    :param benign_catalog: 良性目录路径
    :param malicious_catalog: 恶性目录路径
    :param sample_num: 样本数量
    :param output_catalog: 数据集输出
    :return:
    """
    # 良性域名和恶性域名的数量
    benign_sample = int(sample_num / 2)
    malicious_sample = int(sample_num / 2)

    # 返回良性域名和恶性域名的dataframe
    benign_dataframe = Benign_Dataframe(benign_catalog, benign_sample)
    malicious_dataframe = Malicious_Dataframe(malicious_catalog, malicious_sample)

    # 拼接最终dataframe,输出csv文件
    final_dataframe = pd.concat([benign_dataframe, malicious_dataframe], axis=0, ignore_index=True)
    final_dataframe.to_csv(output_catalog, index=False, header=False)
    pass


def Malicious_Dataframe(malicious_catalog, sample_num):
    # 恶性域名数据帧
    sampled_df = None

    # 读取所有恶性域名文件
    csv_files = glob.glob(os.path.join(malicious_catalog, '*.csv'))
    # 循环每个文件
    for file in csv_files:
        print(f"filename: {file}")
        dataframe = pd.read_csv(file, header=None)
        # 只要域名列
        dataframe = dataframe.iloc[:, 0:1]
        dataframe.columns = [0]

        # 添加新的一列,全是恶性域名标志"1"
        dataframe.insert(loc=1, column=1, value=1)

        # 处理大写
        dataframe[0] = dataframe[0].str.lower()

        # 每个文件随机抽取n行,抽样数量大于样本总数量时,直接抽取全部
        if dataframe.shape[0] < int(sample_num / len(csv_files)):
            item_df = dataframe.sample(n=dataframe.shape[0])
            pass
        else:
            item_df = dataframe.sample(n=int(sample_num / len(csv_files)))
            pass

        # 拼接每个文件的结果
        sampled_df = pd.concat([sampled_df, item_df], axis=0, ignore_index=True)
        pass

    return sampled_df
    pass


def Benign_Dataframe(benign_catalog, sample_num):
    """
    获取良性数据结果
    :param benign_catalog: 良性数据目录
    :param sample_num: 样本总数
    :return:
    """
    # 良性域名数据帧
    sampled_df = None

    # 读取所有良性域名文件
    csv_files = glob.glob(os.path.join(benign_catalog, '*.csv'))
    # 循环每个文件
    for file in csv_files:
        print(f"filename: {file}")
        # 添加新的一列,全是良性域名标志"0"
        dataframe = pd.read_csv(file, header=None)
        dataframe.insert(loc=2, column=2, value=0)

        # 处理大写
        dataframe[1] = dataframe[1].str.lower()

        # 按列切割,重新命名,去掉命名行
        dataframe = dataframe.iloc[1:, 1:3]
        dataframe.columns = [0, 1]

        # 每个文件随机抽取n行
        item_df = dataframe.sample(n=int(sample_num / len(csv_files)))

        # 拼接每个文件的结果
        sampled_df = pd.concat([sampled_df, item_df], axis=0, ignore_index=True)
        pass
    return sampled_df


if __name__ == '__main__':
    # 良性域名目录
    benign_catalog = '../../data/benign'
    # 恶性域名目录
    malicious_catalog = '../../data/malicious'
    # 数据集总量
    sample_num = 100000
    # 输出目录
    output_catalog = f'../../dataOutPut/output_{sample_num}.csv'

    # 获取数据集
    Benign_Malicious_Dataset(benign_catalog, malicious_catalog, sample_num, output_catalog)
    pass
