import numpy as np
import matplotlib.pyplot as plt

[x, y] = np.meshgrid(np.arange(-128, 127), np.arange(-128, 127))
D = np.sqrt((x ** 2) + (y ** 2)).reshape(1, -1)
H = np.zeros([255, 255]).reshape(1, -1)
Downs = (D[:, :] < 30).reshape(1, -1)
H[np.where(Downs)] = D[Downs].reshape(1, -1)
plt.imshow(H.reshape(255, 255), cmap='gray', vmin=0, vmax=1)
plt.show()
