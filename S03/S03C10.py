import numpy as np
import matplotlib.pyplot as plt

# Create an array for using to make Ideal high pass filter
[x, y] = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
# Take square root from sum of two x and y and make new array with that
D = np.sqrt(x ** 2 + y ** 2)
# save filter 2D size
HShape = D.shape
# Reshape to (1, n) for working easy with that
D = D.reshape(1, -1)
# Create filter and init with zeros
H = np.zeros(shape=HShape).reshape(1, -1)
# Find correct range for put variables into filter
x = (D[:, :] > 30).reshape(1, -1)
# reinitialize filter to make correct high pass filter
H[np.where(x)] = D[x]
# Plot filter
plt.imshow(H.reshape(HShape), cmap='gray')
plt.show()
