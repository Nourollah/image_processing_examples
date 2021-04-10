import numpy as np
import matplotlib.pyplot as plt


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def gaussian(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = np.exp(((-distance((y, x), center) ** 2) / (2 * (D0 ** 2))))
    return base



s1 = 256
s2 = 256
v = 30
H = 1 - gaussian(v, (s1, s2))
plt.imshow(H, cmap='gray')
plt.show()
