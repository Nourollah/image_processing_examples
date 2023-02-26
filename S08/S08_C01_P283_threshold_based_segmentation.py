import matplotlib.pyplot as plt
import numpy as np
import cv2

im = cv2.imread('hafez.jpeg', cv2.IMREAD_GRAYSCALE)
plt.figure(1)
plt.subplot(1, 2, 1)
plt.imshow(im)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.hist(im, bins=255, range=[0, 255])
plt.title('Histogram')
plt.xlabel('Intensity Value')
plt.ylabel('Pixel Count')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()

T0 = 1
T = np.mean(im)

while True:
    # R1 is segment 1
    R1 = np.where(im >= T)
    # R2 is segment 2
    R2 = np.where(im < T)
    # M1 is average of segment 1
    M1 = np.mean(im[R1])
    # M2 is average of segment 2
    M2 = np.mean(im[R2])
    # Update threshold
    New_T = (M1 + M2) / 2
    print(New_T)
    if T0 > abs(New_T - T):
        break
    T = New_T

# R us tge segmented image
R = im > T
plt.figure(2)
plt.imshow(R.astype('uint8') * 255)
plt.axis('off')
plt.show()
