import sys
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


def remove_digits_pth(text):
    # return re.sub(r'\.|[0-9]|pth$|Model', '', text)
    return re.sub(r'\.pth$', '', text)
    pass


if __name__ == '__main__':
    print(f"参数:{sys.argv}")
    line_path_dir = './picture/bar'
    if not Path(line_path_dir).exists():
        Path(line_path_dir).mkdir(parents=True, exist_ok=True)
        pass

    # 数据文件参数
    file_path = str(sys.argv[1])
    # 文件存放名称
    file_name = str(sys.argv[2])

    # 数据帧
    dataframe = pd.read_csv(file_path, header=None, usecols=[1, 2, 3, 4, 5])
    # 每个模型的数据帧
    dataframe_list = []
    # 分割数据文件的每个模型记录
    for index in range(0, len(dataframe.columns)):
        group_df = dataframe.iloc[:, index].tolist()
        dataframe_list.append(group_df)
        pass

    # 准备数据
    # x轴
    x_bar = dataframe.iloc[0].tolist()
    x_bar = list(map(remove_digits_pth, x_bar))
    # y轴
    acc_bar = dataframe.iloc[2].tolist()
    acc_bar = [float(num) for num in acc_bar]
    rec_bar = dataframe.iloc[7].tolist()
    rec_bar = [float(num) for num in rec_bar]
    f1_bar = dataframe.iloc[8].tolist()
    f1_bar = [float(num) for num in f1_bar]
    list_bar = [acc_bar, rec_bar, f1_bar]

    # 图像基本设置
    plt.figure(figsize=(15, 10))
    plt.xticks(fontsize=12)

    # 设置柱状宽度
    bar_width = 0.25
    # 计算每一组柱状图的位置
    x = range(len(x_bar))
    x2 = [i + bar_width for i in x]
    x3 = [i + bar_width * 2 for i in x]

    # 绘制三张不同柱状图
    plt.bar(x, acc_bar, width=bar_width, label='Accuracy', color='blue', alpha=0.8)
    plt.bar(x2, rec_bar, width=bar_width, label='Recall', color='blue', alpha=0.5)
    plt.bar(x3, f1_bar, width=bar_width, label='F1', color='blue', alpha=0.2)
    # 设置一些刻度标签
    plt.xticks([i + bar_width for i in x], x_bar)
    # 设置y轴刻度间隔为0.1
    plt.yticks([i / 10 for i in range(int(max(max(acc_bar), max(rec_bar), max(f1_bar)) * 10) + 2)])
    # 添加辅助文本
    for index in range(len(x_bar)):
        plt.text(x[index], acc_bar[index], '{:.2f}'.format(acc_bar[index]), ha='center', va='bottom')
        plt.text(x2[index], rec_bar[index], '{:.2f}'.format(rec_bar[index]), ha='center', va='bottom')
        plt.text(x3[index], f1_bar[index], '{:.2f}'.format(f1_bar[index]), ha='center', va='bottom')
        pass

    # 添加图例
    plt.legend(loc='upper left')
    # 保存
    plt.savefig('picture/bar/result.png')
    # 画图
    # for index in range(len(columns_list)):
    #     # 坐标轴
    #     plt.figure(figsize=(15, 10))
    #     fig, ax = plt.subplots(figsize=(15, 10))
    #     plt.xticks(fontsize=12)
    #
    #     # 创建柱状图
    #     for item in range(len(x_bar)):
    #         # 绘制柱状
    #         plt.bar(x_bar[item], list_bar[index][item], label=str(x_bar[item]), width=0.5, align='center',
    #                 alpha=list_alpha[item])
    #         # 辅助文本
    #         plt.text(x_bar[item], list_bar[index][item], '{:.2f}'.format(list_bar[index][item]), ha='center',
    #                  va='bottom')
    #         pass
    #
    #     # 添加标题和标签
    #     plt.title('Prediction', fontsize=30)
    #     plt.xlabel(f'Prediciton {columns_y_labels[index]}', fontsize=30)
    #     plt.ylabel('Model', fontsize=30)
    #     # 添加图例
    #     plt.legend(loc='upper left')
    #     # 保存图像到当前文件夹
    #     plt.savefig(columns_pic_path[index])
    #     plt.clf()
    #     pass
    pass
