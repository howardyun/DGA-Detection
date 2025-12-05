# import matplotlib.pyplot as plt
#
# # Epoch 数
# epochs = list(range(1, 16))
#
# # # 模型对应的 Multi Accuracy 数据
# # cnn_acc = [0.58, 0.67, 0.72, 0.75, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78]
# # lstm_acc = [0.70, 0.80, 0.85, 0.87, 0.88, 0.89, 0.90, 0.88, 0.89, 0.89, 0.90, 0.90, 0.90, 0.90, 0.91]
# # mit_acc = [0.88, 0.90, 0.91, 0.91, 0.91, 0.91, 0.91, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.93]
# # bbyb_acc = [0.55, 0.70, 0.80, 0.82, 0.85, 0.88, 0.89, 0.89, 0.88, 0.88, 0.89, 0.89, 0.90, 0.91, 0.91]
# # transformer_acc = [0.83, 0.85, 0.86, 0.87, 0.87, 0.88, 0.88, 0.88, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89, 0.90]
# # smad_acc = [0.91, 0.92, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93]
#
# # CNN (Blue, Dotted)
# cnn_acc = [0.850, 0.920, 0.930, 0.940, 0.945, 0.940, 0.950, 0.945, 0.950, 0.950, 0.950, 0.940, 0.940, 0.940, 0.940]
#
# # LSTM (Orange, Dashed)
# lstm_acc = [0.700, 0.920, 0.930, 0.940, 0.660, 0.900, 0.930, 0.930, 0.940, 0.945, 0.945, 0.935, 0.930, 0.925, 0.925]
#
# # MIT (Green, Dashed)
# mit_acc = [0.500, 0.500, 0.640, 0.690, 0.720, 0.750, 0.810, 0.870, 0.890, 0.905, 0.920, 0.925, 0.930, 0.930, 0.930]
#
# # BBBYB (Red, Dashed)
# bbyb_acc = [0.880, 0.950, 0.760, 0.930, 0.950, 0.950, 0.940, 0.940, 0.950, 0.955, 0.955, 0.950, 0.950, 0.945, 0.940]
#
# # Transformer (Purple, Dotted)
# transformer_acc = [0.550, 0.820, 0.860, 0.890, 0.910, 0.915, 0.920, 0.920, 0.930, 0.940, 0.950, 0.950, 0.950, 0.950, 0.945]
#
# # SMAD (Brown, Dashed)
# smad_acc = [0.890, 0.960, 0.960, 0.960, 0.960, 0.970, 0.970, 0.970, 0.970, 0.965, 0.965, 0.970, 0.970, 0.970, 0.970]
#
#
# # 绘图
# plt.figure(figsize=(8, 6))
# plt.plot(epochs, cnn_acc, label="CNN", linestyle="--", color="deepskyblue", marker='o')
# plt.plot(epochs, lstm_acc, label="LSTM", linestyle="--", color="orange", marker='x')
# plt.plot(epochs, mit_acc, label="MIT", linestyle="-", color="green", marker='.')
# plt.plot(epochs, bbyb_acc, label="BBYB", linestyle="-", color="red", marker='^')
# plt.plot(epochs, transformer_acc, label="Transformer", linestyle=":", color="purple", marker='*')
# plt.plot(epochs, smad_acc, label="SMAD", linestyle="-.", color="brown", marker='s')
#
# # 图例、标签和网格
# plt.legend(loc="lower right")
# plt.xlabel("Epoch")
# plt.ylabel("Multi Accuracy")
# plt.title("Model Multi Accuracy over Epochs")
# plt.grid(True)
#
# # 保存图像
# plt.savefig("multi_accuracy_plot.png", dpi=300)
#
# # 显示图像
# plt.show()
import matplotlib.pyplot as plt

# ===== 全局设置 =====
plt.rcParams.update({
    "font.size": 20,           # 全局字体
    "axes.titlesize": 18,      # 标题字体
    "axes.labelsize": 18,      # 坐标轴标签字体
    "xtick.labelsize": 16,     # X轴刻度字体
    "ytick.labelsize": 16,     # Y轴刻度字体
    "legend.fontsize": 18,     # 图例字体
    "grid.linestyle": "-",    # 网格线样式
    "grid.linewidth": 0.8,     # 主网格线宽
    "grid.alpha": 1          # 主网格透明度
})



# Epoch 数
epochs = list(range(1, 16))

# 模型数据
cnn_acc = [0.850, 0.920, 0.930, 0.940, 0.945, 0.940, 0.950, 0.945, 0.950, 0.950, 0.950, 0.940, 0.940, 0.940, 0.940]
lstm_acc = [0.700, 0.920, 0.930, 0.940, 0.660, 0.900, 0.930, 0.930, 0.940, 0.945, 0.945, 0.935, 0.930, 0.925, 0.925]
mit_acc = [0.500, 0.575, 0.640, 0.690, 0.720, 0.750, 0.810, 0.870, 0.890, 0.905, 0.920, 0.925, 0.930, 0.930, 0.930]
bbyb_acc = [0.880, 0.950, 0.760, 0.930, 0.950, 0.950, 0.940, 0.940, 0.950, 0.950, 0.950, 0.950, 0.950, 0.955, 0.955]
transformer_acc = [0.550, 0.820, 0.860, 0.890, 0.910, 0.915, 0.920, 0.920, 0.930, 0.940, 0.950, 0.950, 0.950, 0.950, 0.945]
smad_acc = [0.890, 0.960, 0.960, 0.960, 0.960, 0.970, 0.970, 0.970, 0.970, 0.965, 0.965, 0.970, 0.970, 0.970, 0.970]

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(epochs, cnn_acc, label="CNN", linestyle="--", color="deepskyblue", marker='o')
plt.plot(epochs, lstm_acc, label="LSTM", linestyle="--", color="orange", marker='x')
plt.plot(epochs, mit_acc, label="MIT", linestyle="-", color="green", marker='.')
plt.plot(epochs, bbyb_acc, label="BBYB", linestyle="-", color="red", marker='^')
plt.plot(epochs, transformer_acc, label="Transformer", linestyle=":", color="purple", marker='*')
plt.plot(epochs, smad_acc, label="SMAD", linestyle="-.", color="brown", marker='s')

# 图例、标签和网格
plt.legend(loc="lower right")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xticks(epochs)
# plt.title("Model Multi Accuracy over Epochs")
plt.grid(True)

# 保存与显示
plt.tight_layout()
plt.savefig("multi_accuracy_plot.pdf", dpi=300)
plt.show()
