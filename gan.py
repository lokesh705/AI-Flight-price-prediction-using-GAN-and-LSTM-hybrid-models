import matplotlib.pyplot as plt
import numpy as np

# Example simulated data
epochs = np.arange(1, 101)
generator_loss = np.exp(-epochs / 30) + np.random.normal(0, 0.02, size=100)
discriminator_loss = 1 - np.exp(-epochs / 30) + np.random.normal(0, 0.02, size=100)

# Plotting the loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, generator_loss, label='Generator Loss', color='blue', linewidth=2)
plt.plot(epochs, discriminator_loss, label='Discriminator Loss', color='orange', linewidth=2)
plt.title('GAN Training Loss Curves', fontsize=16)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
