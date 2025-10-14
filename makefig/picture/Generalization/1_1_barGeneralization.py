import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['Accuracy', 'Recall', 'F1 Score']
cnn_scores = [0.9192, 0.9165, 0.9572]
lstm_scores = [0.9404, 0.9492, 0.9688]
mit_scores = [0.8743, 0.8759, 0.9319]
bbyb_scores = [0.9192, 0.9211, 0.9546]
transformer_scores = [0.9175, 0.9205, 0.9563]
smad_scores = [0.9546, 0.9609, 0.9764]

# Bar positions
x = np.arange(len(categories))
width = 0.085  # Width of the bars
spacing = 0.055  # Space between bars

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars for each category with different patterns and distinct colors
bars1 = ax.bar(x - 2*(width + spacing), cnn_scores, width, label='CNN', color='#D94736', edgecolor='black', hatch='//')
bars2 = ax.bar(x - (width + spacing), lstm_scores, width, label='LSTM', color='#E5744E', edgecolor='black', hatch='\\')
bars3 = ax.bar(x, mit_scores, width, label='MIT', color='#F9E295', edgecolor='black', hatch='X')
bars4 = ax.bar(x + (width + spacing), bbyb_scores, width, label='BBYB', color='#FFFF99', edgecolor='black', hatch='.')
bars5 = ax.bar(x + 2*(width + spacing), transformer_scores, width, label='Transformer', color='#B2D977', edgecolor='black', hatch='*')
bars6 = ax.bar(x + 3*(width + spacing), smad_scores, width, label='SMAD', color='#479254', edgecolor='black', hatch='--')

# Add labels and title with larger font sizes
ax.set_ylabel('Scores', fontsize=15)
# ax.set_title('Comparison of Models', fontsize=16)

# Adjust the x-axis and y-axis tick label sizes
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=15)
ax.tick_params(axis='y', labelsize=15)

# Adjust the y-axis to start from 0.85
ax.set_ylim(0.85, 1.0)

# Adjust the legend to be horizontal at the top and move it downward
ax.legend(ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.01), fancybox=True, shadow=True, fontsize=13)

# Add score values on top of bars
for bars in [bars1, bars2, bars3, bars4, bars5, bars6]:
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.4f}', ha='center', va='bottom', fontsize=10,rotation=45)
plt.grid(axis='y', linestyle='--', color='gray')


# Show the plot
plt.tight_layout()
plt.savefig('200w_1比1_lb_org_bin_lb.pdf', format='pdf')

plt.show()
