# Run main_predict_bin_lb.py

#python main_predict_bin_lb.py  <A> <B>  <current_name>  <base_path>  <idx1> <ckpt1>  [<idx2> <ckpt2> ...]
#
#<A>：是否做鲁棒-家族预测（int）
#
#1 = 家族预测（会走 PredictionFamily)
#
#0 = 正常预测（会走 Predictions)
#
#<B>：是否使用全数据集（int）
#
#1 = 全数据集
#
#0 = 部分数据集
#
#<current_name>：这次预测任务的名字（会用于结果保存的文件夹命名）
#
#<base_path>：模型文件根目录（和后面的 <ckptX> 拼起来定位权重）
#
#<idxK> <ckptK>：要加载的模型编号 + 该模型的权重相对路径（可给多对，成对出现）
#
#模型编号对照（脚本里固定的 model_name_list）：
#0 -> 'ANN'   (utils/model/ann/ann_torch.Net)
#1 -> 'CNN'   (CNNModel)
#2 -> 'LSTM'  (LSTMModel)
#3 -> 'MIT'   (MITModel)
#4 -> 'BBYB'  (BilBoHybridModel)



# Run main_predict_bin_lb.py

#python main_predict_bin_normal.py  <A> <B  > <current_name> <base_path> <idx1> <ckpt1> [<idx2> <ckpt2> ...]
#<A>：预测类型
#
#0 = 随机数据集预测
#
#1 = 分割数据集预测
#
#<B>：数据量选择
#
#0 = 使用部分数据
#
#1 = 使用全部数据
#
#<current_name>：本次预测任务的名字（会影响结果保存路径）
#
#<base_path>：模型权重的根目录
#
#<idxK> <ckptK>：模型编号和对应的权重文件（成对出现，可以多个）
#
#模型编号对照：
#0 -> ANN
#1 -> CNN
#2 -> LSTM
#3 -> MIT
#4 -> BBYB




