"""
时间: 2026-3-4
作用: 切分2024年DGA数据工具类
"""
import os
from pathlib import Path
import random
import pandas as pd
import csv


def count_csv_rows_pandas(folder_path):
    """
    输出目录下每个csv文件目录总量
    :param folder_path: 文件路径
    :return: 输出每个csv文件数据总量
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            try:
                df = pd.read_csv(file_path)
                # 数据行数（自动跳过表头）
                rows = df.shape[0]
                print(f"{filename}: {rows} 行")
                pass
            except Exception as e:
                print(f"{filename}: 读取失败 - {e}")
                pass
            pass
        pass
    pass


def merge_sampled_csv(folder_path, base_value, output_file, encoding='utf-8'):
    """
    从文件夹中的每个 CSV 文件抽取一定行数，合并到输出文件。
    参数:
        folder_path (str): 包含 CSV 文件的文件夹路径
        base_value (int): 基准值（总目标抽取行数）
        output_file (str): 输出 CSV 文件路径
        encoding (str): 文件编码，默认 utf-8
    """
    # 获取所有 CSV 文件
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]
    file_count = len(csv_files)
    if file_count == 0:
        print("文件夹中没有 CSV 文件。")
        pass

    # 计算每个文件应抽取的行数
    per_sample = base_value // file_count
    if per_sample == 0:
        print(f"警告：基准值 {base_value} 小于文件数量 {file_count}，每个文件抽取 0 行，结果文件为空。")
        pass

    # 存储所有抽取的数据行
    all_sampled_rows = []
    header = None  # 表头将从第一个成功读取的文件获取

    # 遍历每个 CSV 文件
    for filename in csv_files:
        file_path = os.path.join(folder_path, filename)
        try:
            # 打开文件读取csv数据
            with open(file_path, 'r', newline='', encoding=encoding) as f:
                reader = csv.reader(f)
                rows = list(reader)
                pass

            # 空文件跳过
            if not rows:
                print(f"{filename} 是空文件，跳过。")
                continue

            # 分离表头和数据行
            file_header = rows[0]
            data_rows = rows[1:]

            # 如果还没有设置表头，用当前文件的表头
            if header is None:
                header = file_header
                pass

            # 确定抽取数量
            sample_count = per_sample
            if len(data_rows) <= sample_count:
                # 数据行不足，全部选取
                selected = data_rows
                pass
            else:
                # 随机抽取 sample_count 行
                selected = random.sample(data_rows, sample_count)
                pass

            # 组合最终数据
            all_sampled_rows.extend(selected)
            print(f"{filename}: 数据行数 {len(data_rows)}，抽取 {len(selected)} 行")
            pass

        except Exception as e:
            print(f"读取 {filename} 时出错: {e}")
            pass

    # 写入输出文件
    with open(output_file, 'w', newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(all_sampled_rows)

    print(f"\n合并完成，共抽取 {len(all_sampled_rows)} 行数据，已保存至 {output_file}")


if __name__ == '__main__':
    # folder = "../../data"
    # count_csv_rows_pandas(folder)

    folder_path = "../../data"
    base_value = 10000
    output_file = f'../../dataOutPut/dataset_{base_value}.csv'
    encoding = 'utf-8'
    merge_sampled_csv(folder_path, base_value, output_file, encoding)
    pass
