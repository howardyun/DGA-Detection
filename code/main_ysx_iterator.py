import torch
import os
from torch import nn
import sys
from utils.engine_ysx import train_ysx, train_multi

sys.path.append('model')
# 所有可用模型
from model.cnn.cnn_torch import CNNModel, CNNMultiModel
from model.lstm.lstm_torch import LSTMModel, LSTMMultiModel
from model.mit.mit_torch import MITModel, MITMultiModel
from model.ann.ann_torch import Net, NetMulti
from model.bilbohybrid.bilbohybrid_torch import BilBoHybridModel, BBYBMultiModel
# 所有工具类函数
from utils.saveModel import SaveModel, LoadModel, SaveResults
from utils.predictions import Predictions, PredictionFamily, SaveFilePath

# 训练模型参数
# 按照数据集正负样本比例变化改变
pos_weight_num = 0.0202
NUM_EPOCHS = 5
BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
# 训练设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 训练初始化参数
train_file = ''
test_file = ''
predict_file = ''
predict_full_data_flag = False
lb_flag = False

# 预测初始化参数
family_flag = False
family_full_data_flag = False
family_predict_file = ''

# 预测模型路径
base_path = './modelPth/dga/2024031310'
ann_name = '0.001ANNModel.pth'
cnn_name = '0.001CNNModel.pth'
lstm_name = '0.001LSTMModel.pth'
mit_name = '0.001MITModel.pth'
bbyb_name = '0.001BBYBModel.pth'


def readData():
    pass


def initParam():
    """
    执行初始化流程
    """
    print(f"初始化选项")
    # 初始化全局变量
    global lb_flag, train_file, test_file
    init_flag = input("是否为鲁棒性测试, 不是0, 是1")
    lb_flag = True if int(init_flag) == 1 else False
    if lb_flag:
        # 鲁棒性测试
        flag = input("鲁棒性测试是否使用全数据集, 不是0, 是1")
        if int(flag) == 1:
            # 全数据集
            train_file = '../data/lb_full_data/lb_train2016.csv'
            test_file = '../data/lb_full_data/lb_test2016.csv'
            pass
        else:
            train_file = '../data/lb_partial_data/lb_train2016.csv'
            test_file = '../data/lb_partial_data/lb_test2016.csv'
            pass
        pass
    else:
        # 非鲁棒性测试,正常训练
        flag = input("正常训练是否使用全数据集, 不是0, 是1")
        if int(flag) == 1:
            train_file = '../data/train2016.csv'
            test_file = '../data/test2016.csv'
            pass
        else:
            train_file = '../data/extract_remain_data/2016/train.csv'
            test_file = '../data/extract_remain_data/2016/test.csv'
            pass
        pass

    pass


def initPredictParam():
    """
    初始化预测流程参数
    :return:
    """
    print("初始化选项")
    # 初始化全局变量
    global lb_flag, family_flag, family_full_data_flag, family_predict_file, predict_file, predict_full_data_flag
    init_flag = input("正常预测0, 鲁棒性预测1")
    lb_flag = True if int(init_flag) == 1 else False
    if lb_flag:
        # 鲁棒性
        # 鲁棒性预测
        flag = input("正常数据预测0, 家族数据预测1")
        if int(flag) == 1:
            # 鲁棒-家族预测
            family_flag = True
            flag = input("使用部分数据0, 全部数据1")
            if int(flag) == 1:
                # 全数据集
                family_full_data_flag = True
                family_predict_file = '../data/lb_full_data/lb_type2016.csv'
                pass
            else:
                # 部分数据集
                family_full_data_flag = False
                family_predict_file = '../data/lb_partial_data/lb_type2016.csv'
                pass
            pass
        else:
            # 鲁棒-正常预测
            family_flag = False
            flag = input("使用部分数据0, 全部数据1")
            if int(flag) == 1:
                # 全数据集
                predict_file = '../data/lb_full_data/lb_predict2016.csv'
                pass
            else:
                # 部分数据集
                predict_file = '../data/lb_partial_data/lb_predict2016.csv'
                pass
            pass
        pass
    else:
        # 正常
        flag = input("使用随机数据集预测0, 使用分割数据集预测1")
        if int(flag) == 1:
            # 分割数据集预测
            flag = input("使用部分数据0, 全部数据1")
            if int(flag) == 1:
                predict_full_data_flag = True
                predict_file = '../data/extract_remain_data/2016/predict.csv'
                pass
            else:
                predict_full_data_flag = False
                predict_file = '../data/extract_remain_data/2016/predict.csv'
                pass
            pass
        else:
            # 随机数据集预测0
            flag = input("使用部分数据0, 全部数据1")
            if int(flag) == 1:
                predict_full_data_flag = True
                predict_file = '../data/test2016.csv'
                pass
            else:
                predict_full_data_flag = False
                predict_file = '../data/test2016.csv'
                pass
            pass
        pass
    pass


if __name__ == '__main__':
    input_flag = input("0训练模型, 1模型预测")
    if int(input_flag) == 0:
        initParam()
        print(f"确定模型,设备为: {device}, 是否是鲁棒性测试: {lb_flag}")
        print(f"pos weight: {pos_weight_num}")
        print("请确认训练集是否正确,如不正确修改初始化函数")
        print(f"训练数据集文件为: {train_file}, {test_file}")

        # 确定训练模型
        bni_multi_flag = input("0二分类, 1多分类")
        if int(bni_multi_flag) == 0:
            print("载入二分类模型")
            model_ann = Net(255, 255, 255)
            model_cnn = CNNModel(255, 255, 255, 5)
            model_lstm = LSTMModel(255, 255)
            model_mit = MITModel(255, 255)
            model_bbyb = BilBoHybridModel(255, 255, 5)
            # 二分类函数损失函数和优化器
            # 定义二元交叉熵损失函数，并使用 pos_weight 参数
            # 正样本和负样本比例要按照数据集变化改变
            pos_weight = torch.tensor([pos_weight_num])
            pos_weight = pos_weight.to(device)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            pass
        else:
            print("载入多分类模型")
            # 2016年多分类为64个恶意+一个良性总共65个
            model_ann = NetMulti(255, 255, 255, num_classes=65)
            model_cnn = CNNMultiModel(255, 255, 255, 5, num_classes=65)
            model_lstm = LSTMMultiModel(255, 255, num_classes=65)
            model_mit = MITMultiModel(255, 255, num_classes=65)
            model_bbyb = BBYBMultiModel(255, 255, 5, num_classes=65)
            # 多分类损失函数
            loss_fn = nn.CrossEntropyLoss()
            pass

        # 模型优化器
        # ann,cnn,mit的学习率在大数据境况下较低(1e-4左右),而lstm和bbyb在大数据情况下较高(1e-2左右)
        ann_lr = 0.0001
        cnn_lr = 0.0001
        lstm_lr = 0.0001
        mit_lr = 0.0001
        bbyb_lr = 0.0001
        optimizer_ann = torch.optim.SGD(params=model_ann.parameters(),
                                        lr=ann_lr)
        optimizer_cnn = torch.optim.SGD(params=model_cnn.parameters(),
                                        lr=cnn_lr)
        optimizer_lstm = torch.optim.SGD(params=model_lstm.parameters(),
                                         lr=lstm_lr)
        optimizer_mit = torch.optim.SGD(params=model_mit.parameters(),
                                        lr=mit_lr)
        optimizer_bbyb = torch.optim.SGD(params=model_bbyb.parameters(),
                                         lr=bbyb_lr)

        print("训练模型ANN开始")
        # 训练模型，标签为True
        print("训练模型ANN")
        if int(bni_multi_flag) == 0:
            results = train_ysx(model=model_ann,
                                train_file=train_file,
                                test_file=test_file,
                                loss_fn=loss_fn,
                                optimizer=optimizer_ann,
                                epochs=NUM_EPOCHS,
                                device=device,
                                BATCH_SIZE=BATCH_SIZE)
            pass
        else:
            results = train_multi(model=model_ann,
                                  train_file=train_file,
                                  test_file=test_file,
                                  loss_fn=loss_fn,
                                  optimizer=optimizer_ann,
                                  epochs=NUM_EPOCHS,
                                  device=device,
                                  BATCH_SIZE=BATCH_SIZE)
            pass

        # 保存训练结果
        SaveResults(str(ann_lr) + "ANNModel", NUM_EPOCHS, results, lb_flag)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_ann.zero_grad()
        model_ann.train()

        print("训练模型CNN开始")
        # 训练模型，标签为True
        print("训练模型CNN")
        if int(bni_multi_flag) == 0:
            results = train_ysx(model=model_cnn,
                                train_file=train_file,
                                test_file=test_file,
                                loss_fn=loss_fn,
                                optimizer=optimizer_cnn,
                                epochs=NUM_EPOCHS,
                                device=device,
                                BATCH_SIZE=BATCH_SIZE)
            pass
        else:
            results = train_multi(model=model_cnn,
                                  train_file=train_file,
                                  test_file=test_file,
                                  loss_fn=loss_fn,
                                  optimizer=optimizer_cnn,
                                  epochs=NUM_EPOCHS,
                                  device=device,
                                  BATCH_SIZE=BATCH_SIZE)
            pass

        # 保存训练结果
        SaveResults(str(cnn_lr) + "CNNModel", NUM_EPOCHS, results, lb_flag)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_cnn.zero_grad()
        model_cnn.train()

        print("训练模型LSTM开始")
        # 训练模型，标签为True
        print("训练模型LSTM")
        if int(bni_multi_flag) == 0:
            results = train_ysx(model=model_lstm,
                                train_file=train_file,
                                test_file=test_file,
                                loss_fn=loss_fn,
                                optimizer=optimizer_lstm,
                                epochs=NUM_EPOCHS,
                                device=device,
                                BATCH_SIZE=BATCH_SIZE)
            pass
        else:
            results = train_multi(model=model_lstm,
                                  train_file=train_file,
                                  test_file=test_file,
                                  loss_fn=loss_fn,
                                  optimizer=optimizer_lstm,
                                  epochs=NUM_EPOCHS,
                                  device=device,
                                  BATCH_SIZE=BATCH_SIZE)
            pass
        # 保存训练结果
        SaveResults(str(lstm_lr) + "LSTMModel", NUM_EPOCHS, results, lb_flag)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_lstm.zero_grad()
        model_lstm.train()

        print("训练模型MIT开始")
        # 训练模型，标签为True
        print("训练模型MIT")
        if int(bni_multi_flag) == 0:
            results = train_ysx(model=model_mit,
                                train_file=train_file,
                                test_file=test_file,
                                loss_fn=loss_fn,
                                optimizer=optimizer_mit,
                                epochs=NUM_EPOCHS,
                                device=device,
                                BATCH_SIZE=BATCH_SIZE)
            pass
        else:
            results = train_multi(model=model_mit,
                                  train_file=train_file,
                                  test_file=test_file,
                                  loss_fn=loss_fn,
                                  optimizer=optimizer_mit,
                                  epochs=NUM_EPOCHS,
                                  device=device,
                                  BATCH_SIZE=BATCH_SIZE)
            pass
        # 保存训练结果
        SaveResults(str(mit_lr) + "MITModel", NUM_EPOCHS, results, lb_flag)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_mit.zero_grad()
        model_mit.train()

        print("训练模型BBYB开始")
        # 训练模型，标签为True
        print("训练模型BBYB")
        if int(bni_multi_flag) == 0:
            results = train_ysx(model=model_bbyb,
                                train_file=train_file,
                                test_file=test_file,
                                loss_fn=loss_fn,
                                optimizer=optimizer_bbyb,
                                epochs=NUM_EPOCHS,
                                device=device,
                                BATCH_SIZE=BATCH_SIZE)
            pass
        else:
            results = train_multi(model=model_bbyb,
                                  train_file=train_file,
                                  test_file=test_file,
                                  loss_fn=loss_fn,
                                  optimizer=optimizer_bbyb,
                                  epochs=NUM_EPOCHS,
                                  device=device,
                                  BATCH_SIZE=BATCH_SIZE)
            pass
        # 保存训练结果
        SaveResults(str(bbyb_lr) + "BBYBModel", NUM_EPOCHS, results, lb_flag)
        # 在训练新的数据集之前，清零梯度，并将模型设置为训练状态
        optimizer_bbyb.zero_grad()
        model_bbyb.train()

        save_flag = input("0不保存模型, 1保存模型")
        # save_flag = 1
        if int(save_flag) == 1:
            print("保存模型ANN")
            SaveModel(model=model_ann,
                      target_dir="modelPth",
                      lb_flag=lb_flag,
                      model_name=str(ann_lr) + "ANNModel.pth")
            print("保存模型CNN")
            SaveModel(model=model_cnn,
                      target_dir="modelPth",
                      lb_flag=lb_flag,
                      model_name=str(cnn_lr) + "CNNModel.pth")
            print("保存模型LSTM")
            SaveModel(model=model_lstm,
                      target_dir="modelPth",
                      lb_flag=lb_flag,
                      model_name=str(lstm_lr) + "LSTMModel.pth")
            print("保存模型MIT")
            SaveModel(model=model_mit,
                      target_dir="modelPth",
                      lb_flag=lb_flag,
                      model_name=str(mit_lr) + "MITModel.pth")
            print("保存模型BBYB")
            SaveModel(model=model_bbyb,
                      target_dir="modelPth",
                      lb_flag=lb_flag,
                      model_name=str(bbyb_lr) + "BBYBModel.pth")
            pass
        pass
    else:
        # 初始化
        print("模型预测")
        initPredictParam()
        print(f"确定模型,设备为: {device}")
        print("请确认预测集是否正确,如不正确修改初始化函数")

        # 确定模型基本结构
        model_ann = Net(255, 255, 255)
        model_cnn = CNNModel(255, 255, 255, 5)
        model_lstm = LSTMModel(255, 255)
        model_mit = MITModel(255, 255)
        model_bbyb = BilBoHybridModel(255, 255, 5)

        # 加载模型参数
        model_ann = LoadModel(model_ann, base_path, ann_name)
        model_cnn = LoadModel(model_cnn, base_path, cnn_name)
        model_lstm = LoadModel(model_lstm, base_path, lstm_name)
        model_mit = LoadModel(model_mit, base_path, mit_name)
        model_bbyb = LoadModel(model_bbyb, base_path, bbyb_name)

        # 模型预测
        if lb_flag:
            # 鲁棒性
            print("鲁棒性预测")
            if family_flag:
                # 鲁棒家族
                print(f"获取数据集: {family_predict_file}")
                print("全数据集") if family_full_data_flag else print("部分数据集")
                results_file_dir, acc_pre_f1_file_path = SaveFilePath(lb_flag=lb_flag)
                PredictionFamily(model=model_ann, model_name=ann_name, file=family_predict_file,
                                 results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                                 device=device,
                                 full_flag=family_full_data_flag,
                                 BATCH_SIZE=BATCH_SIZE)
                PredictionFamily(model=model_cnn, model_name=cnn_name, file=family_predict_file,
                                 results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                                 device=device,
                                 full_flag=family_full_data_flag,
                                 BATCH_SIZE=BATCH_SIZE)
                PredictionFamily(model=model_lstm, model_name=lstm_name, file=family_predict_file,
                                 results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                                 device=device,
                                 full_flag=family_full_data_flag,
                                 BATCH_SIZE=BATCH_SIZE)
                PredictionFamily(model=model_mit, model_name=mit_name, file=family_predict_file,
                                 results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                                 device=device,
                                 full_flag=family_full_data_flag,
                                 BATCH_SIZE=BATCH_SIZE)
                PredictionFamily(model=model_bbyb, model_name=bbyb_name, file=family_predict_file,
                                 results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                                 device=device,
                                 full_flag=family_full_data_flag,
                                 BATCH_SIZE=BATCH_SIZE)
                pass
            else:
                # 鲁棒正常
                print(f"获取数据集: {predict_file}")
                print("全数据集") if family_full_data_flag else print("部分数据集")
                results_file_dir, acc_pre_f1_file_path = SaveFilePath(lb_flag=lb_flag)
                Predictions(model=model_ann, model_name=ann_name, file=predict_file,
                            results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                            device=device,
                            full_flag=predict_full_data_flag,
                            BATCH_SIZE=BATCH_SIZE)
                Predictions(model=model_cnn, model_name=cnn_name, file=predict_file,
                            results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                            device=device,
                            full_flag=predict_full_data_flag,
                            BATCH_SIZE=BATCH_SIZE)
                Predictions(model=model_lstm, model_name=lstm_name, file=predict_file,
                            results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                            device=device,
                            full_flag=predict_full_data_flag,
                            BATCH_SIZE=BATCH_SIZE)
                Predictions(model=model_mit, model_name=mit_name, file=predict_file,
                            results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                            device=device,
                            full_flag=predict_full_data_flag,
                            BATCH_SIZE=BATCH_SIZE)
                Predictions(model=model_bbyb, model_name=bbyb_name, file=predict_file,
                            results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                            device=device,
                            full_flag=predict_full_data_flag,
                            BATCH_SIZE=BATCH_SIZE)
                pass
            pass
        else:
            # 正常
            print(f"获取数据集: {predict_file}")
            print("全数据集") if predict_full_data_flag else print("部分数据集")
            results_file_dir, acc_pre_f1_file_path = SaveFilePath(lb_flag=lb_flag)
            Predictions(model=model_ann, model_name=ann_name, file=predict_file,
                        results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                        device=device,
                        full_flag=predict_full_data_flag,
                        BATCH_SIZE=BATCH_SIZE)
            Predictions(model=model_cnn, model_name=cnn_name, file=predict_file,
                        results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                        device=device,
                        full_flag=predict_full_data_flag,
                        BATCH_SIZE=BATCH_SIZE)
            Predictions(model=model_lstm, model_name=lstm_name, file=predict_file,
                        results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                        device=device,
                        full_flag=predict_full_data_flag,
                        BATCH_SIZE=BATCH_SIZE)
            Predictions(model=model_mit, model_name=mit_name, file=predict_file,
                        results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                        device=device,
                        full_flag=predict_full_data_flag,
                        BATCH_SIZE=BATCH_SIZE)
            Predictions(model=model_bbyb, model_name=bbyb_name, file=predict_file,
                        results_file_dir=results_file_dir, acc_pre_f1_file_path=acc_pre_f1_file_path,
                        device=device,
                        full_flag=predict_full_data_flag,
                        BATCH_SIZE=BATCH_SIZE)
            pass
        pass
    pass
