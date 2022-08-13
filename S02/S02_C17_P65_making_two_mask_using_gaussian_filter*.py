import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


s = 50
v1 = 2
v2 = 7

X = np.arange(1, s + 1)
Y = np.arange(1, s + 1)
X, Y = np.meshgrid(X, Y)

plt.figure(1)
g1 = style_gauss2D((s, s), v1)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, g1, cmap=cm.coolwarm, linewidth=0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('G(X,Y)')
plt.show()

plt.figure(2)
g2 = style_gauss2D((s, s), v2)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, g2, cmap=cm.coolwarm, linewidth=0)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('G(X,Y)')
plt.show()
