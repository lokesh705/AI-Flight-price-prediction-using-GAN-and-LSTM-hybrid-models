import matplotlib.pyplot as plt
import numpy as np

# Models and their metric scores
models = ['Regression', 'LSTM', 'GAN + LSTM']
mae = [1753.92, 1521.28, 1303.17]
mse = [5092092.46, 4336115.47, 3680992.87]
rmse = [2256.12, 2082.25, 1918.63]
r2 = [0.6723, 0.7136, 0.7913]

# Function to create a pie chart in a subplot
def create_pie_chart(ax, data, labels, title):
    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'], wedgeprops={'edgecolor': 'black'})
    ax.set_title(title)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Create pie charts for each metric in subplots
create_pie_chart(axes[0, 0], mae, models, 'MAE Score Comparison of Models')
create_pie_chart(axes[0, 1], mse, models, 'MSE Score Comparison of Models')
create_pie_chart(axes[1, 0], rmse, models, 'RMSE Score Comparison of Models')
create_pie_chart
import matplotlib.pyplot as plt
import numpy as np

# Models and their metric scores
models = ['Regression', 'LSTM', 'GAN + LSTM']
mae = [1753.92, 1521.28, 1303.17]
mse = [5092092.46, 4336115.47, 3680992.87]
rmse = [2256.12, 2082.25, 1918.63]
r2 = [0.6723, 0.7136, 0.7913]

# Function to create a pie chart in a subplot
def create_pie_chart(ax, data, labels, title):
    ax.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff', '#99ff99'], wedgeprops={'edgecolor': 'black'})
    ax.set_title(title)

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Create pie charts for each metric in subplots
create_pie_chart(axes[0, 0], mae, models, 'MAE Score Comparison of Models')
create_pie_chart(axes[0, 1], mse, models, 'MSE Score Comparison of Models')
create_pie_chart(axes[1, 0], rmse, models, 'RMSE Score Comparison of Models')
create_pie_chart(axes[1, 1], r2, models, 'R² Score Comparison of Models')

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()
