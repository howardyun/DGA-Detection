import sys
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    print(f"参数:{sys.argv}")

    # 数据文件
    file_path = str(sys.argv[1])
    # 数据文件中一个模型训练的epoch
    epoch = int(sys.argv[2])

    dataframe = pd.read_csv(file_path)
    dataframe_list = []
    # 分割数据文件的每个模型的记录
    for i in range(0, len(dataframe), epoch):
        group_df = dataframe.iloc[i: i + epoch]
        dataframe_list.append(group_df)
        pass

    # 绘制四个图像
    # 准确率
    plt.figure(figsize=(15, 10))
    x = dataframe_list[0]['epoch'].tolist()
    marker_list = ['o', '^', 's', 'd', '*']
    for index in range(len(dataframe_list)):
        y = dataframe_list[index]['model_accuracy'].tolist()
        plt.plot(x, y, label=str(dataframe_list[index]['model_name'].tolist()[0]), marker=marker_list[index])
        pass
    plt.grid()
    # 添加图例
    plt.legend()
    # 添加标题和标签
    plt.title('model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    # 保存图像到当前文件夹
    plt.savefig('picture/line_accuracy.png')
    plt.clf()

    # 精确率
    plt.figure(figsize=(15, 10))
    x = dataframe_list[0]['epoch'].tolist()
    marker_list = ['o', '^', 's', 'd', '*']
    for index in range(len(dataframe_list)):
        y = dataframe_list[index]['model_precision'].tolist()
        plt.plot(x, y, label=str(dataframe_list[index]['model_name'].tolist()[0]), marker=marker_list[index])
        pass
    plt.grid()
    # 添加图例
    plt.legend()
    # 添加标题和标签
    plt.title('model precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    # 保存图像到当前文件夹
    plt.savefig('picture/line_precision.png')
    plt.clf()

    # 召回率
    plt.figure(figsize=(15, 10))
    x = dataframe_list[0]['epoch'].tolist()
    marker_list = ['o', '^', 's', 'd', '*']
    for index in range(len(dataframe_list)):
        y = dataframe_list[index]['model_recall'].tolist()
        plt.plot(x, y, label=str(dataframe_list[index]['model_name'].tolist()[0]), marker=marker_list[index])
        pass
    plt.grid()
    # 添加图例
    plt.legend()
    # 添加标题和标签
    plt.title('model recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    # 保存图像到当前文件夹
    plt.savefig('picture/line_recall.png')
    plt.clf()

    # F1
    plt.figure(figsize=(15, 10))
    x = dataframe_list[0]['epoch'].tolist()
    marker_list = ['o', '^', 's', 'd', '*']
    for index in range(len(dataframe_list)):
        y = dataframe_list[index]['model_f1'].tolist()
        plt.plot(x, y, label=str(dataframe_list[index]['model_name'].tolist()[0]), marker=marker_list[index])
        pass
    plt.grid()
    # 添加图例
    plt.legend()
    # 添加标题和标签
    plt.title('model f1')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    # 保存图像到当前文件夹
    plt.savefig('picture/line_f1.png')
    plt.clf()
    pass
