import cv2
import numpy as np
import scipy.signal
from skimage import color as scl
from matplotlib import pyplot as plt
from skimage.util import random_noise
from sklearn.metrics import mean_squared_error



imgRGB = cv2.cvtColor(cv2.imread("Lenna.jpg", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

#im2double
img = cv2.normalize(imgRGB.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

m,n,k=img.shape

imgn= random_noise(img, mode='s&p', seed=None, clip=True)


imgf=np.zeros([m,n,k])

imgf[:, :, 0] = scipy.signal.medfilt2d(imgn[:, :, 0])
imgf[:, :, 1] = scipy.signal.medfilt2d(imgn[:, :, 1])
imgf[:, :, 2] = scipy.signal.medfilt2d(imgn[:, :, 2])

MSEREn=mean_squared_error(imgf[:, :, 0], img[:, :, 0])
MSEGEn=mean_squared_error(imgf[:, :, 1], img[:, :, 1])
MSEBEn=mean_squared_error(imgf[:, :, 2], img[:, :, 2])
MSERGBEn=(MSEREn+MSEGEn+MSEBEn)/3

plt.subplot(221)
plt.imshow(imgn)
plt.title("Noiseing Image")

plt.subplot(222)
plt.imshow(imgf)
plt.title(f"RGB MSE : {str(MSERGBEn)}")



imgHSV = scl.rgb2hsv(imgn)
imgHSVEn=np.zeros([m,n,k])

imgHSVEn[:, :, 0] = scipy.signal.medfilt2d(imgHSV[:, :, 0])
imgHSVEn[:, :, 1] = scipy.signal.medfilt2d(imgHSV[:, :, 1])
imgHSVEn[:, :, 2] = scipy.signal.medfilt2d(imgHSV[:, :, 2])

MSEHEn=mean_squared_error(imgHSVEn[:, :, 0], img[:, :, 0])
MSESEn=mean_squared_error(imgHSVEn[:, :, 1], img[:, :, 1])
MSEVEn=mean_squared_error(imgHSVEn[:, :, 2], img[:, :, 2])
MSEHSVEn=(MSEHEn+MSESEn+MSEVEn)/3

plt.subplot(223)
plt.imshow(scl.hsv2rgb(imgHSVEn))
plt.title(f"HSV MSE : {str(MSEHSVEn)}")


imgYIQ = scl.rgb2yiq(imgn)
imgYIQEn=np.zeros([m,n,k])

imgYIQEn[:, :, 0] = scipy.signal.medfilt2d(imgYIQ[:, :, 0])
imgYIQEn[:, :, 1] = scipy.signal.medfilt2d(imgYIQ[:, :, 1])
imgYIQEn[:, :, 2] = scipy.signal.medfilt2d(imgYIQ[:, :, 2])

MSEYEn=mean_squared_error(imgYIQEn[:, :, 0], img[:, :, 0])
MSEIEn=mean_squared_error(imgYIQEn[:, :, 1], img[:, :, 1])
MSEQEn=mean_squared_error(imgYIQEn[:, :, 2], img[:, :, 2])
MSEYIQEn=(MSEYEn+MSEIEn+MSEQEn)/3

plt.subplot(224)
plt.imshow(scl.yiq2rgb(imgYIQEn))
plt.title(f"YIQ MSE : {str(MSEYIQEn)}")


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.6)
plt.show()