import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread("Text.BMP",0)
B1 = cv2.imread("Kernel.BMP",0)
B2=cv2.copyMakeBorder(np.zeros(B1.shape,dtype=np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT,value=255)
tb1 = cv2.morphologyEx(img, cv2.MORPH_ERODE, B1)
tb2 = cv2.morphologyEx(~(img), cv2.MORPH_ERODE, B2)
HitOrMiss=tb1&tb2
images=[img,B1,HitOrMiss]
titles=['Text','Kernel','HitOrMiss']
fig1 = plt.figure(figsize=(20, 2))
for idx in range(3):
    ax = fig1.add_subplot(1, 3, idx+1)
    ax.imshow(images[idx],cmap='gray')
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
