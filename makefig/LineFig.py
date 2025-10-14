import matplotlib.pyplot as plt

# Epoch 数
epochs = list(range(1, 16))

# 模型对应的 Multi Accuracy 数据
cnn_acc = [0.58, 0.67, 0.72, 0.75, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78, 0.78]
lstm_acc = [0.70, 0.80, 0.85, 0.87, 0.88, 0.89, 0.90, 0.88, 0.89, 0.89, 0.90, 0.90, 0.90, 0.90, 0.91]
mit_acc = [0.88, 0.90, 0.91, 0.91, 0.91, 0.91, 0.91, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.92, 0.93]
bbyb_acc = [0.55, 0.70, 0.80, 0.82, 0.85, 0.88, 0.89, 0.89, 0.88, 0.88, 0.89, 0.89, 0.90, 0.91, 0.91]
transformer_acc = [0.83, 0.85, 0.86, 0.87, 0.87, 0.88, 0.88, 0.88, 0.88, 0.89, 0.89, 0.89, 0.89, 0.89, 0.90]
smad_acc = [0.91, 0.92, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93]

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
plt.ylabel("Multi Accuracy")
plt.title("Model Multi Accuracy over Epochs")
plt.grid(True)

# 保存图像
plt.savefig("multi_accuracy_plot.png", dpi=300)

# 显示图像
plt.show()
