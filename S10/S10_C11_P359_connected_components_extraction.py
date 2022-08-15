import cv2
import matplotlib.pyplot as plt
import numpy as np

A = cv2.imread("TextC.bmp",0)
B = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#B = cv2.getStructuringElement(cv2.MORPH_RECT,(20,20))
Px,Py=90,20
X_last=np.zeros(A.shape,dtype=np.uint8)
X_last[Px,Py]=255
X_current=cv2.morphologyEx(X_last, cv2.MORPH_DILATE, B) & A
while (X_current != X_last).any():
    X_last=X_current
    X_current=cv2.morphologyEx(X_last, cv2.MORPH_DILATE, B) & A
result=X_current
# Visualize
images=[A,result]
titles=['Original','result']
fig1 = plt.figure(figsize=(10, 10))
for idx in range(2):
    ax = fig1.add_subplot(1, 2, idx+1)
    ax.imshow(images[idx],cmap='gray')
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
