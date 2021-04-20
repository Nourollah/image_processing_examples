import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import multivariate_normal
import sys


def style_log2D(p2, std):
    siz = int((p2 - 1) / 2)
    x = y = np.linspace(-siz, siz, 2 * siz + 1)
    x, y = np.meshgrid(x, y)
    arg = -(x ** 2 + y ** 2) / (2 * std ** 2)
    h = np.exp(arg)
    h[h < sys.float_info.epsilon * h.max()] = 0
    h = h / h.sum() if h.sum() != 0 else h
    h1 = h * (x ** 2 + y ** 2 - 2 * std ** 2) / (std ** 4)
    return h1 - h1.mean()



size = 50
sigma1 = 2
sigma2 = 7

X = np.arange(1, size)
Y = np.arange(1, size)
X, Y = np.meshgrid(X, Y)

plt.figure(1)
LoG1 = style_log2D(size, sigma1)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, LoG1, cmap=cm.coolwarm, linewidth=0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('G(X,Y)')
plt.show()

plt.figure(2)
LoG2 = style_log2D(size, sigma2)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, LoG2, cmap=cm.coolwarm, linewidth=0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('G(X,Y)')
plt.show()
