import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Replace with your actual data
actual_prices = [4500, 4800, 5000, 5200, 5500, 6000, 6500, 7000, 7200]
predicted_prices = [4600, 4700, 4950, 5100, 5400, 5900, 6450, 7100, 7350]

# Create a DataFrame
df = pd.DataFrame({
    'Actual Price': actual_prices,
    'Predicted Price': predicted_prices
})

# Plot the distribution
plt.figure(figsize=(10, 6))
sns.kdeplot(df['Actual Price'], label='Actual Prices', color='blue', fill=True)
sns.kdeplot(df['Predicted Price'], label='Predicted Prices', color='green', fill=True)

plt.title('Distribution of Actual vs Predicted Flight Prices', fontsize=16)
plt.xlabel('Price (INR)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("price_distribution_plot.png")
plt.show()
