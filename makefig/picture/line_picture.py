import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(f"参数:{sys.argv}")
    line_path_dir = './picture/line'
    if not Path(line_path_dir).exists():
        Path(line_path_dir).mkdir(parents=True, exist_ok=True)
        pass

    # 数据文件
    file_path = str(sys.argv[1])
    # 数据文件中一个模型训练的epoch
    epoch = int(sys.argv[2])
    # 文件存放名称
    file_name = str(sys.argv[3])

    dataframe = pd.read_csv(file_path, header=None)
    dataframe_list = []
    # 分割数据文件的每个模型的记录
    for i in range(0, len(dataframe), epoch + 1):
        group_df = dataframe.iloc[i: i + epoch + 1]
        group_df.columns = group_df.iloc[0]
        group_df = group_df[1:]  # 删除第一行数据，因为它已经成为了列标题
        # print(group_df)
        dataframe_list.append(group_df)
        pass

    # 一些路径
    columns_list = ['test_acc', 'test_precision', 'test_recall', 'test_f1']
    columns_y_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    columns_pic_path = [f'picture/line/line_accuracy_{file_name}.png', f'picture/line/line_precision_{file_name}.png',
                        f'picture/line/line_recall_{file_name}.png',
                        f'picture/line/line_f1_{file_name}.png']

    for item in range(len(columns_list)):
        plt.figure(figsize=(15, 10))
        fig, ax = plt.subplots(figsize=(15, 10))

        # 获取x轴
        x = dataframe_list[0].iloc[:, 0].tolist()
        # 不同标点
        marker_list = ['o', '^', 's', 'd', '*']
        y_list = []
        for index in range(len(dataframe_list)):
            y = dataframe_list[index][columns_list[item]].tolist()
            y = [float(num) for num in y]
            y_list = y_list + y
            # 绘制折线图并增加线条粗细
            ax.plot(x, y, linewidth=2)
            ax.plot(x, y, label=str(dataframe_list[index].columns[0]), marker=marker_list[index])
            # 调整x轴文字大小
            plt.xticks(x, fontsize=11, rotation=30)
            pass
        # 处理y轴
        y_min = min(y_list)
        y_max = max(y_list)
        # 刻度间隔
        yticks_interval = 0.1
        y_floor = np.floor(float(y_min) * 100) / 100
        y_ceil = np.ceil(float(y_max) * 100) / 100 + yticks_interval
        yticks_positions = np.arange(y_floor,
                                     y_ceil,
                                     yticks_interval)
        yticks_positions = np.append(yticks_positions[:-1], 1.0) if y_ceil > 1.0 else yticks_positions
        # 处理小数点格式化
        yticks_positions *= 10
        yticks_positions = np.floor(yticks_positions)
        yticks_positions /= 10
        # 生成刻度标签
        yticks_labels = [f"{i:.2f}" for i in yticks_positions[:]]
        # 设置Y轴刻度
        ax.set_yticks(yticks_positions)
        ax.set_yticklabels(yticks_labels)
        # 网格线
        ax.grid(True)
        # 添加标题和标签
        ax.set_title(f'{columns_y_labels[item]}', fontsize=30)
        ax.set_xlabel('Epoch', fontsize=30)
        ax.set_ylabel(columns_y_labels[item], fontsize=30)
        # 添加图例
        ax.legend(loc='upper left')
        # 保存图像到当前文件夹
        fig.savefig(columns_pic_path[item])
        plt.clf()
        pass
    pass
