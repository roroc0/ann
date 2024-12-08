import matplotlib.pyplot as plt
import numpy as np

# Data
commands = ['SIMD', 'Sequential', 'Parallel']
means = [8055.410, 9616.945, 13204.468]
std_devs = [6.529, 2.155, 47.202]

# Create bar chart
x_pos = np.arange(len(commands))

fig, ax = plt.subplots()
bars = ax.bar(x_pos, means, yerr=std_devs, capsize=5, alpha=0.75, ecolor='black', color='skyblue')

# Add labels and title
ax.set_xlabel('Processing Types')
ax.set_ylabel('Time (s)')
ax.set_title('Benchmarking Training Time')
ax.set_xticks(x_pos)
ax.set_xticklabels(commands)

# Attach a text label above each bar displaying its height (mean value).
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',  # Text to show
                xy=(bar.get_x() + bar.get_width() / 2, height), # Position of the text
                xytext=(0, 3),  # Offset the text
                textcoords="offset points",
                ha='center', va='bottom')

plt.tight_layout() # Adjust layout to make room for labels
plt.show()
