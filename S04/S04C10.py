import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.util as sut


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


img = cv2.normalize(cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE).astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
I = sut.random_noise(img, 'gaussian', mean=0.0, var=0.01)
method = 'pm2'
N = 8
K = 5
Delta = 0.25
sigma = 0.1
Sy, Sx = I.shape
for _ in range(1, N):
    if sigma > 0:
        Io = I
        g4 = style_gauss2D(sigma=sigma)
        I = cv2.filter2D(I, -1, g4, borderType=cv2.BORDER_REPLICATE)

    dn = [I[1, :], I[1:Sy - 1, :]] - I
    ds = [I[2:Sy, :], I[Sy, :]] - I
    de = [I[:, 2:Sy], I[:, Sx]] - I
    dw = [I[:, 1], I[:, Sx:1]] - I
