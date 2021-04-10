import numpy as np
import matplotlib.pyplot as plt

# Create an array for using to make Ideal low pass filter
[x, y] = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
# Set radius and filter times
R = 30
n = 2
# Apply Butterworth low pass filter formula to H array
H = 1 / (1 + ((np.sqrt(x ** 2 + y ** 2) / R) ** (2 ** n)))
# Plot filter
plt.imshow(H)
plt.show()
