import cv2
from skimage.morphology import disk
import matplotlib.pyplot as plt

img = cv2.imread("CT.BMP",0)
SE = disk(5)
#SE = disk(18)
topHat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, SE)      # Top_Hat
blackHat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, SE)  # Bottom_Hat
Enh_img = img + topHat - blackHat                         # Enhancing Contrast

# Visualize
images=[img,Enh_img]
titles=['Orginal','Contrast Enhanced']
fig1 = plt.figure(figsize=(20, 20))
for idx in range(2):
    ax = fig1.add_subplot(1, 2, idx+1)
    ax.imshow(images[idx],cmap='gray')
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
