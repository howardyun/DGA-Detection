import pandas as pd
import os
import glob

# DGA（Domain Generation Algorithm）域名特征提取是一种用于检测恶意软件和网络攻击的技术。下面是一些常见的DGA域名特征提取的方向：

# 域名长度特征：DGA生成的域名通常具有特定的长度范围，可以通过统计域名长度分布来进行特征提取。

# 域名字符集特征：DGA生成的域名通常使用特定的字符集，如只包含字母、数字或特定的字符组合。通过分析域名中包含的字符类型和字符集分布，可以提取特征。

# 域名频率特征：DGA生成的域名通常具有较高的频率，即生成的域名在短时间内大量出现。可以通过统计域名的出现频率来进行特征提取。

# 域名结构特征：DGA生成的域名通常具有特定的结构或模式，如特定的前缀、后缀、分隔符等。可以通过分析域名的结构来提取特征。

# 域名时间特征：DGA生成的域名通常具有特定的时间相关特征，如在特定的时间间隔内生成的域名具有相似的特征。可以通过分析域名生成的时间模式来提取特征。

# 域名语义特征：DGA生成的域名通常缺乏语义意义，不符合正常域名的命名规则。可以通过分析域名的语义特征，如是否包含常见的单词、词典中的词等，来进行特征提取。

# 域名排列组合特征：DGA生成的域名通常具有排列组合的特征，即使用特定的字符组合、组件或模式生成域名。可以通过分析域名的排列组合特征来进行特征提取。

# 域名网络行为特征：DGA生成的域名通常与恶意软件的网络行为相关联，如与C&C（Command and Control）服务器进行通信等。可以通过分析域名与其他网络行为的关联来提取特征。

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
# 被用于给域名长度信息作图使用的函数
def calculateLen_figure(benign_data_calculate,malicious_data_calculate):
    return










def diff_file_comparation(myfuc):
    benign_folder_path = 'data/Benign'  # 文件夹的路径
    malicious_folder_path = 'data/DGA/2016-09-19-dgarchive_full'
    
    # 良性文件路径
    benign_file_paths = glob.glob(os.path.join(benign_folder_path, '*'))
    
    # 恶性文件路径
    malicious_file_paths = glob.glob(os.path.join(malicious_folder_path, '*'))
    
    # 记录良性域名的数据
    benign_data_calculate = []
    malicious_data_calculate = []
    # 良性域名统计
    for file in benign_file_paths:
        # 读文件
        data = pd.read_csv(file)
        # 取出域名
        domains = data['domain'].unique()
        # 进行统计，添加结果
        benign_data_calculate.append(['benign',myfuc(domains)]) 
    
    # 恶意域名统计
    for file in malicious_file_paths:
        # 读文件
        data = pd.read_csv(file)
        # 拉出恶意域名
        domains = data.iloc[:,0].tolist()
        # 获取DGA家族
        label = data.iloc[:,0].to_list()[0]
        #获取结果
        malicious_data_calculate.append([label, myfuc(domains)]) 
    return benign_data_calculate,malicious_data_calculate

print(diff_file_comparation(calculateLen))
    
    