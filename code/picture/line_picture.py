import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(f"参数:{sys.argv}")

    # 数据文件
    file_path = str(sys.argv[1])
    # 数据文件中一个模型训练的epoch
    epoch = int(sys.argv[2])

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
    columns_list = ['test_acc', 'test_precision', 'test_recall', 'test_f1']
    columns_y_labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    columns_pic_path = ['picture/line_accuracy.png', 'picture/line_precision.png', 'picture/line_recall.png',
                        'picture/line_f1.png']

    # print(dataframe_list[0].iloc[:, 0])
    for item in range(len(columns_list)):
        plt.figure(figsize=(15, 10))
        x = dataframe_list[0].iloc[:, 0].tolist()
        marker_list = ['o', '^', 's', 'd', '*']
        y_list = []
        for index in range(len(dataframe_list)):
            y = dataframe_list[index][columns_list[item]].tolist()
            y_list = y_list + y
            # 绘制折线图并增加线条粗细
            plt.plot(x, y, linewidth=2)
            plt.plot(x, y, label=str(dataframe_list[index].columns[0]), marker=marker_list[index])
            pass

        # 计算 Y 轴刻度位置和标签
        y_min = min(y_list)
        y_max = max(y_list)
        yticks_interval = 0.01  # 刻度间隔
        yticks_positions = np.arange(np.floor(float(y_min) * 100) / 100,
                                     np.ceil(float(y_min) * 100) / 100 + yticks_interval,
                                     yticks_interval)  # 计算刻度位置
        yticks_labels = [f"{i:.2f}" if (int(i * 100 % 5) == 0) else f"" for i in yticks_positions[:]]  # 生成刻度标签
        # 设置 Y 轴刻度
        plt.yticks(yticks_positions, yticks_labels)

        plt.grid()
        # 添加图例
        plt.legend()
        # 添加标题和标签
        plt.title(f'model {columns_y_labels[item]}')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(columns_y_labels[item], fontsize=14)
        # 保存图像到当前文件夹
        plt.savefig(columns_pic_path[item])
        plt.clf()
        pass
