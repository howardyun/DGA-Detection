"""
时间: 2026-3-4
作用: 构建模型对象,统计模型的参数量,FLOPs,吞吐量
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback
import time
import csv
from thop import profile, clever_format

# 二分类模型
from code.model.ann.ann_torch import Net
from code.model.cnn.cnn_torch import CNNModel
from code.model.lstm.lstm_torch import LSTMModel
from code.model.mit.mit_torch import MITModel
from code.model.bilbohybrid.bilbohybrid_torch import BilBoHybridModel
from code.model.transformer.transformer_improve import Trans_DGA

# 多分类模型
from code.model.cnn.cnn_torch import CNNMultiModel
from code.model.lstm.lstm_torch import LSTMMultiModel
from code.model.mit.mit_torch import MITMultiModel
from code.model.ann.ann_torch import NetMulti
from code.model.bilbohybrid.bilbohybrid_torch import BBYBMultiModel
from code.model.transformer.transformer_improve import Trans_DGA_Multi


def measure_throughput(model, input_data, batch_size, num_iterations=100, device='cuda'):
    """
    测量模型吞吐量的辅助函数
    参数:
        model: PyTorch 模型
        input_data: 单个 batch 的输入张量，形状为 (batch_size, ...)
        batch_size: 批处理大小
        num_iterations: 推理次数
        device: 'cuda' 或 'cpu'

    返回:
        throughput: 吞吐量（样本/秒）
    """
    model.eval()
    # 确保输入在正确的设备上
    input_data = input_data.to(device)

    # 预热,稳定测试设备
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
            pass
        pass

    if device == 'cuda':
        torch.cuda.synchronize()
        pass

    # 计时
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_data)
            pass
        pass
    if device == 'cuda':
        torch.cuda.synchronize()
        pass
    end_time = time.perf_counter()

    total_time = end_time - start_time
    total_samples = batch_size * num_iterations
    throughput = total_samples / total_time
    return throughput


def multiModelParameter(output_csv='model_parameters.csv', measure_throughput_flag=False, batch_size=32,
                        num_iterations=100, device='cuda'):
    # 实例化所有多酚类模型
    models = {
        "ANN_Multi": NetMulti(255, 255, 255, num_classes=65),
        "CNN_Multi": CNNMultiModel(255, 255, 255, 5, num_classes=65),
        "LSTM_Multi": LSTMMultiModel(255, 255, num_classes=65),
        "MIT_Multi": MITMultiModel(255, 255, num_classes=65),
        "BilBoHybrid_Multi": BBYBMultiModel(255, 255, 5, num_classes=65),
        "Transformer_Multi": Trans_DGA_Multi(num_classes=65, vocab_size=40)
    }

    # 准备数据列表
    data = []
    for name, model in models.items():
        print(f"\n处理模型: {name}")
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())

        # 计算FLops
        try:
            model.eval()
            if name == 'Transformer_Multi':
                input_seq_len = 255
                input_data_flops = torch.randint(0, 40, (1, input_seq_len))
                pass
            else:
                input_seq_len = 255
                input_data_flops = torch.randint(0, 255, (1, input_seq_len))
                pass
            # 计算 FLOPs
            flops, _ = profile(model, inputs=(input_data_flops,), verbose=False)
            pass
        except Exception as e:
            print(f"计算 {name} 的 FLOPs 时出错: {e}")
            flops = 0
            pass

        # 吞吐量测量
        throughput = None
        if measure_throughput_flag:
            try:
                # 将模型移至指定设备
                model = model.to(device)
                # 构造适合吞吐量测量的输入
                if name == "Transformer_Multi":
                    input_data_throughput = torch.randint(0, 40, (batch_size, input_seq_len))
                    pass
                else:
                    input_data_throughput = torch.randint(0, 255, (batch_size, input_seq_len))
                    pass
                throughput = measure_throughput(model, input_data_throughput, batch_size, num_iterations, device)
                pass
            except Exception as e:
                print(f"计算 {name} 的吞吐量时出错: {e}")
                throughput = 0
                pass
            pass

        # 保存数据
        data.append([name, total_params, flops, throughput] if measure_throughput_flag else [name, total_params, flops])

        # 打印到控制台
        params_str = f"{total_params:,}"
        flops_str = f"{flops:,}" if flops > 0 else "N/A"
        if measure_throughput_flag:
            throughput_str = f"{throughput:.2f}" if throughput is not None and throughput > 0 else "N/A"
            print(
                f"{name:15} 参数量: {params_str:>15} ({total_params / 1e6:.2f} M)  FLOPs: {flops_str:>15}  Throughput: {throughput_str:>10} samples/sec")
            pass
        else:
            print(f"{name:15} 参数量: {params_str:>15} ({total_params / 1e6:.2f} M)  FLOPs: {flops_str:>15}")
            pass

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if measure_throughput_flag:
            writer.writerow(['Model Name', 'Parameters', 'FLOPs', 'Throughput (samples/sec)'])
            pass
        else:
            writer.writerow(['Model Name', 'Parameters', 'FLOPs'])
            pass
        writer.writerows(data)
        pass

    print(f"\n数据已保存至 {output_csv}")
    pass


def modelParameter(output_csv='model_parameters.csv', measure_throughput_flag=False,
                   batch_size=32, num_iterations=100, device='cuda'):
    """
    计算各模型的参数量、FLOPs 和吞吐量（可选）并保存到 CSV 文件

    参数:
        output_csv: 输出 CSV 文件路径
        measure_throughput_flag: 是否测量吞吐量（True/False）
        batch_size: 吞吐量测量时的批处理大小
        num_iterations: 吞吐量测量时的推理次数
        device: 运行设备 ('cuda' 或 'cpu')
    """
    # 实例化所有二分类模型
    models = {
        "ANN": Net(255, 255, 255),
        "CNN": CNNModel(255, 255, 255, 5),
        "LSTM": LSTMModel(255, 255),
        "MIT": MITModel(255, 255),
        "BilBoHybrid": BilBoHybridModel(255, 255, 5),
        "Transformer": Trans_DGA(num_classes=1, vocab_size=40)
    }

    # 准备数据列表
    data = []
    for name, model in models.items():
        print(f"\n处理模型: {name}")
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())

        # 构造适合模型的输入张量（batch_size=1 用于 FLOPs 计算）
        try:
            model.eval()
            if name == "Transformer":
                input_seq_len = 255
                input_data_flops = torch.randint(0, 40, (1, input_seq_len))
            else:
                input_seq_len = 255
                input_data_flops = torch.randint(0, 255, (1, input_seq_len))

            # 计算 FLOPs
            flops, _ = profile(model, inputs=(input_data_flops,), verbose=False)
        except Exception as e:
            print(f"计算 {name} 的 FLOPs 时出错: {e}")
            flops = 0

        # 吞吐量测量
        throughput = None
        if measure_throughput_flag:
            try:
                # 将模型移至指定设备
                model = model.to(device)
                # 构造适合吞吐量测量的输入
                if name == "Transformer":
                    input_data_throughput = torch.randint(0, 40, (batch_size, input_seq_len))
                else:
                    input_data_throughput = torch.randint(0, 255, (batch_size, input_seq_len))

                throughput = measure_throughput(model, input_data_throughput, batch_size, num_iterations, device)
            except Exception as e:
                print(f"计算 {name} 的吞吐量时出错: {e}")
                throughput = 0

        # 保存数据
        data.append([name, total_params, flops, throughput] if measure_throughput_flag else [name, total_params, flops])

        # 打印到控制台
        params_str = f"{total_params:,}"
        flops_str = f"{flops:,}" if flops > 0 else "N/A"
        if measure_throughput_flag:
            throughput_str = f"{throughput:.2f}" if throughput is not None and throughput > 0 else "N/A"
            print(
                f"{name:15} 参数量: {params_str:>15} ({total_params / 1e6:.2f} M)  FLOPs: {flops_str:>15}  Throughput: {throughput_str:>10} samples/sec")
        else:
            print(f"{name:15} 参数量: {params_str:>15} ({total_params / 1e6:.2f} M)  FLOPs: {flops_str:>15}")

    # 写入 CSV 文件
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if measure_throughput_flag:
            writer.writerow(['Model Name', 'Parameters', 'FLOPs', 'Throughput (samples/sec)'])
        else:
            writer.writerow(['Model Name', 'Parameters', 'FLOPs'])
        writer.writerows(data)

    print(f"\n数据已保存至 {output_csv}")
    pass


if __name__ == '__main__':
    output_modelParamPath = '../../dataOutPut/model_param.csv'
    # 二分类模型,不测量吞吐量
    # modelParameter(output_modelParamPath)
    # 二分类模型,测量吞吐量
    modelParameter(output_modelParamPath,
                   measure_throughput_flag=True,
                   batch_size=64,
                   num_iterations=200,
                   device='cuda' if torch.cuda.is_available() else 'cpu')

    # 多分类,不测量吞吐量
    output_multiModelParamPath = '../../dataOutPut/multimodel_param.csv'
    # multiModelParameter(output_multiModelParamPath)
    # 多分类测试,测量吞吐量
    multiModelParameter(output_multiModelParamPath,
                        measure_throughput_flag=True,
                        batch_size=64,
                        num_iterations=200,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
    pass
