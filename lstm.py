import matplotlib.pyplot as plt

# Simulated loss values across 50 epochs (replace with real values if available)
epochs = list(range(1, 51))
training_loss = [0.27 - (i * 0.003) for i in range(50)]
validation_loss = [0.29 - (i * 0.0028) + (0.002 if i % 5 == 0 else 0) for i in range(50)]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, training_loss, label='Training Loss', color='blue', linewidth=2, marker='o')
plt.plot(epochs, validation_loss, label='Validation Loss', color='orange', linewidth=2, marker='x')
plt.title('LSTM Model Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an image (optional)
plt.savefig("lstm_training_validation_loss.png")  # Saved in your project directory

# Display the plot
plt.show()
