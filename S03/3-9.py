import cv2
import numpy as np
import matplotlib.pyplot as plt

[x, y] = np.meshgrid(np.arange(-128, 127), np.arange(-128, 127))
D = np.sqrt((x ** 2) + (y ** 2)).reshape(1, -1)
H = np.zeros([256, 256]).reshape(1, -1)
Downs = (D[:, :] < 30).reshape(1, -1)
H[np.where(Downs)] = D[Downs].reshape(1, -1)
H = H.reshape(256, 256)
img = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
imgF = np.fft.fftshift(np.fft.fft2(img))
M_I = imgF * H
M_Id = np.abs(M_I)
M_IdLog = np.log(1 + M_Id)
Max = np.max(np.max(M_IdLog))
plt.subplot(121)
plt.imshow(M_IdLog / Max, cmap='gray')
IFFT = np.fft.ifft2(np.fft.ifftshift(M_I))
IFFTt = np.abs(IFFT)
Max2 = np.max(np.max(IFFTt))
plt.subplot(122)
plt.imshow(IFFTt / Max2, cmap='gray')
plt.show()
