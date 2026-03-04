"""
时间: 2026-3-4
作用: 统计良性,恶性,整体的平均,最大,最小字符串长度
"""
import glob
import os
import pandas as pd


def benign_length(benign_catalog):
    """
    # 仅计算良性数据的字符串特征
    :param benign_catalog: 良性数据目录
    :return:
    """
    csv_files = glob.glob(os.path.join(benign_catalog, '*.csv'))
    # 收集所有域名的长度
    all_lengths = []

    for file in csv_files:
        # print(f"filename: {file}")
        dataframe = pd.read_csv(file, header=None)
        dataframe = dataframe.iloc[:, 1:2]
        dataframe.columns = [0]
        lengths = dataframe.iloc[:, 0].astype(str).str.len()
        all_lengths.extend(lengths.tolist())
        pass

    # 良性数据的字符
    print('良性数据的字符')
    print(f"平均长度：{sum(all_lengths) / len(all_lengths):.2f}")
    print(f"最大长度：{max(all_lengths)}")
    print(f"最小长度：{min(all_lengths)}")
    pass


def malicious_length(malicious_catalog):
    """
    仅计算恶性数据字符串的特征
    :param malicious_catalog: 恶性数据目录
    :return:
    """
    csv_files = glob.glob(os.path.join(malicious_catalog, '*.csv'))
    # 收集所有域名的长度
    all_lengths = []

    for file in csv_files:
        # print(f"filename: {file}")
        dataframe = pd.read_csv(file, header=None)
        dataframe = dataframe.iloc[:, 0:1]
        dataframe.columns = [0]
        lengths = dataframe.iloc[:, 0].astype(str).str.len()
        all_lengths.extend(lengths.tolist())
        pass

    # 恶性数据字符串的特征
    print("恶性数据字符串的特征")
    print(f"平均长度：{sum(all_lengths) / len(all_lengths):.2f}")
    print(f"最大长度：{max(all_lengths)}")
    print(f"最小长度：{min(all_lengths)}")
    pass


def integrity_length(benign_catalog, malicious_catalog):
    # 收集所有域名的长度
    all_lengths = []

    # 良性数据每个字符串长度
    csv_files = glob.glob(os.path.join(benign_catalog, '*.csv'))
    for file in csv_files:
        # print(f"filename: {file}")
        dataframe = pd.read_csv(file, header=None)
        dataframe = dataframe.iloc[:, 1:2]
        dataframe.columns = [0]
        lengths = dataframe.iloc[:, 0].astype(str).str.len()
        all_lengths.extend(lengths.tolist())
        pass

    # 恶性数据每个字符串长度
    csv_files = glob.glob(os.path.join(malicious_catalog, '*.csv'))
    for file in csv_files:
        # print(f"filename: {file}")
        dataframe = pd.read_csv(file, header=None)
        dataframe = dataframe.iloc[:, 0:1]
        dataframe.columns = [0]
        lengths = dataframe.iloc[:, 0].astype(str).str.len()
        all_lengths.extend(lengths.tolist())
        pass

    # 全局统计
    print("全局统计")
    print(f"平均长度：{sum(all_lengths) / len(all_lengths):.2f}")
    print(f"最大长度：{max(all_lengths)}")
    print(f"最小长度：{min(all_lengths)}")
    pass


if __name__ == '__main__':
    # 良性域名目录
    benign_catalog = '../../data/benign'
    # 良性数据字符串特征
    benign_length(benign_catalog)

    # 恶性域名目录
    malicious_catalog = '../../data/malicious'
    malicious_length(malicious_catalog)

    # 全局域名统计
    integrity_length(benign_catalog, malicious_catalog)
    pass
