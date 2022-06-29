import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread("Mazandaran.bmp",0)
A = img.copy()
B = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
Frame=np.zeros(A.shape,dtype=np.uint8)
while (A).any():
    O=cv2.morphologyEx(A, cv2.MORPH_OPEN, B)
    Frame=Frame | A - O
    A=cv2.morphologyEx(A, cv2.MORPH_ERODE, B)
images=[img,Frame]
titles=['Original','result']
fig1 = plt.figure(figsize=(10, 10))
for idx in range(2):
    ax = fig1.add_subplot(1, 2, idx+1)
    ax.imshow(images[idx],cmap='gray')
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
