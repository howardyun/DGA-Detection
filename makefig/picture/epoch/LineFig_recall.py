import matplotlib.pyplot as plt

# ===== 全局设置 =====
plt.rcParams.update({
    "font.size": 16,           # 全局字体
    "axes.titlesize": 18,      # 标题字体
    "axes.labelsize": 16,      # 坐标轴标签字体
    "xtick.labelsize": 14,     # X轴刻度字体
    "ytick.labelsize": 14,     # Y轴刻度字体
    "legend.fontsize": 14,     # 图例字体
    "grid.linestyle": "-",    # 网格线样式
    "grid.linewidth": 0.8,     # 主网格线宽
    "grid.alpha": 1          # 主网格透明度
})


# Epoch 数
epochs = list(range(1, 16))

# CNN (Blue, Dotted)
cnn_recall = [0.720, 0.880, 0.900, 0.910, 0.920, 0.910, 0.920, 0.920, 0.930, 0.930, 0.930, 0.930, 0.930, 0.930, 0.925]

# LSTM (Orange, Dashed)
lstm_recall = [0.450, 0.880, 0.870, 0.900, 0.920, 0.820, 0.880, 0.910, 0.910, 0.870, 0.910, 0.940, 0.940, 0.950, 0.960]

# MIT (Green, Dashed)
mit_recall = [0.450, 0.450, 0.460, 0.480, 0.490, 0.520, 0.670, 0.780, 0.820, 0.840, 0.850, 0.870, 0.870, 0.880, 0.880]

# BBBYB (Red, Dashed)
bbyb_recall = [0.840, 0.880, 0.600, 0.900, 0.920, 0.930, 0.915, 0.915, 0.920, 0.920, 0.925, 0.930, 0.940, 0.950, 0.950]

# Transformer (Purple, Dotted)
transformer_recall = [0.100, 0.730, 0.820, 0.860, 0.870, 0.900, 0.910, 0.930, 0.925, 0.930, 0.935, 0.935, 0.935, 0.930, 0.920]

# SMAD (Brown, Dashed)
smad_recall = [0.900, 0.980, 0.970, 0.970, 0.970, 0.970, 0.970, 0.960, 0.970, 0.970, 0.975, 0.975, 0.975, 0.980, 0.980]

# 绘图
plt.figure(figsize=(8, 6))
plt.plot(epochs, cnn_recall, label="CNN", linestyle="--", color="deepskyblue", marker='o')
plt.plot(epochs, lstm_recall, label="LSTM", linestyle="--", color="orange", marker='x')
plt.plot(epochs, mit_recall, label="MIT", linestyle="-", color="green", marker='.')
plt.plot(epochs, bbyb_recall, label="BBYB", linestyle="-", color="red", marker='^')
plt.plot(epochs, transformer_recall, label="Transformer", linestyle=":", color="purple", marker='*')
plt.plot(epochs, smad_recall, label="SMAD", linestyle="-.", color="brown", marker='s')

# 图例、标签和网格
plt.legend(loc="lower right")
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.xticks(epochs)
# plt.title("Model Multi Accuracy over Epochs")
plt.grid(True)

# 保存与显示
plt.tight_layout()
plt.savefig("multi_recall_plot.png", dpi=300)
plt.show()
