import matplotlib.pyplot as plt

# Sample accuracy metrics
models = ['LSTM', 'GAN + LSTM', 'Linear Regression', 'Random Forest']
mae = [102.45, 94.80, 178.30, 115.67]
rmse = [143.60, 135.25, 210.42, 160.14]
r2 = [0.92, 0.94, 0.78, 0.88]

# Bar positioning
bar_width = 0.25
x = range(len(models))
x1 = [i - bar_width for i in x]
x2 = x
x3 = [i + bar_width for i in x]

# Create the bar chart
plt.figure(figsize=(12, 6))
plt.bar(x1, mae, width=bar_width, color='skyblue', label='MAE')
plt.bar(x2, rmse, width=bar_width, color='orange', label='RMSE')
plt.bar(x3, r2, width=bar_width, color='limegreen', label='R² Score')

# Labeling
plt.xlabel('Models', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Model Accuracy Comparison: LSTM vs GAN + LSTM vs Regression Models', fontsize=14)
plt.xticks(x, models, fontsize=10)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('model_comparison_accuracy.png')
plt.show()
