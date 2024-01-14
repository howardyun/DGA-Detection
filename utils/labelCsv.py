import pandas as pd
import os
import glob
import csv


# 指定路径下的所有csv添加标签列
def SetLabel(root_csv_path, flag):
    """
    :param root_csv_path: csv文件文件夹根路径
    :param flag: 需要添加的标签
    :return:
    """
    # 获取csv文件列表
    csv_files = glob.glob(os.path.join(root_csv_path, '*.csv'))

    # 为每一行添加一列
    for file in csv_files:
        dataframe = pd.read_csv(file, header=None)
        # 生成行数的标签
        label_num = dataframe.shape[0]
        label_index = dataframe.shape[1]
        label_list = [flag] * label_num

        # 加入新的一列
        dataframe[label_index] = label_list
        dataframe.to_csv(file, index=False, header=False)
        print(f'完成添加：{file}')
        pass
    pass


if __name__ == '__main__':
    SetLabel(f'../data/Benign', True)
    SetLabel(f'../data/DGA/2016-09-19-dgarchive_full', False)
    pass
