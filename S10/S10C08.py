import cv2
import matplotlib.pyplot as plt
img = cv2.imread("coins.BMP",0)
SE = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
#Internal Edge
imgO=cv2.morphologyEx(img, cv2.MORPH_ERODE, SE)
EI=img-imgO
#External Edge
imgD=cv2.morphologyEx(img, cv2.MORPH_DILATE, SE)
EO=imgD-img
#Gradient
EG=imgD-imgO
images=[img,EI,EO,EG]
titles=['Original','Internal Edge','External Edge','Gradient']
fig1 = plt.figure(figsize=(15, 15))
for idx in range(4):
    ax = fig1.add_subplot(1, 4, idx+1)
    ax.imshow(images[idx],cmap='gray')
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
