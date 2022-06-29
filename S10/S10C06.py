import cv2
from skimage.morphology import disk
import matplotlib.pyplot as plt
img = cv2.imread("CT.BMP",0)
SE1 = disk(5)
topHat1 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, SE1)      #Top_Hat
blackHat1 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, SE1)  #Bottom_Hat
Enh_img1 = img + topHat1 - blackHat1
SE2 = disk(18)
topHat2 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, SE2)      #Top_Hat
blackHat2 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, SE2)  #Bottom_Hat
Enh_img2 = img + topHat2 - blackHat2                        #Enhancing contrast
images=[img,Enh_img1,Enh_img2]
titles=['Orginal','Contrast Enhanced (5x5)','Contrast Enhanced (37x37)']
fig1 = plt.figure(figsize=(20, 20))
for idx in range(3):
    ax = fig1.add_subplot(1, 3, idx+1)
    ax.imshow(images[idx],cmap='gray')
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
