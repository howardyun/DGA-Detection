import pandas as pd
import os
import glob

# 计算给入域名的长度信息包括（最长，最短，平均长度，中位长度，众数长度，长度范围）
def calculateLen(datas):
    # 计算最小长度
    min_length = min(len(s) for s in datas)

    # 计算最大长度
    max_length = max(len(s) for s in datas)

    # 计算平均长度
    average_length = sum(len(s) for s in datas) / len(datas)

    # 计算中位数长度
    sorted_lengths = sorted(len(s) for s in datas)
    mid = len(sorted_lengths) // 2
    if len(sorted_lengths) % 2 == 0:
        median_length = (sorted_lengths[mid - 1] + sorted_lengths[mid]) / 2
    else:
        median_length = sorted_lengths[mid]

    # 计算众数长度
    length_counts = {}
    for s in datas:
        length = len(s)
        length_counts[length] = length_counts.get(length, 0) + 1
    mode_length = max(length_counts, key=length_counts.get)

    # 计算长度范围
    length_range = max_length - min_length

    return {
        'min_length': min_length,
        'max_length': max_length,
        'average_length': average_length,
        'median_length': median_length,
        'mode_length': mode_length,
        'length_range': length_range
    }


def diff_file_comparation(myfuc):
    benign_folder_path = 'data/Benign'  # 文件夹的路径
    malicious_folder_path = 'data/DGA/2016-09-19-dgarchive_full'
    
    # 良性文件路径
    benign_file_paths = glob.glob(os.path.join(benign_folder_path, '*'))
    
    # 恶性文件路径
    malicious_file_paths = glob.glob(os.path.join(malicious_folder_path, '*'))
    
    benign_data_calculate = []
    malicious_data_calculate = []
    # 良性域名统计
    for file in benign_file_paths:
        data = pd.read_csv(file)
        domains = data['domain'].unique()
        benign_data_calculate.append(myfuc(domains)) 
        
    # 恶意域名统计
    for file in malicious_file_paths:
        data = pd.read_csv(file)
        domains = data['domain'].unique()
        malicious_data_calculate.append(myfuc(domains)) 
    return benign_data_calculate,malicious_data_calculate

print(diff_file_comparation(calculateLen))
    
    