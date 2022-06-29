import cv2
import matplotlib.pyplot as plt
img = cv2.imread("Text_Shahid_Avini.BMP",0)
SE = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
img1 = cv2.morphologyEx(img, cv2.MORPH_DILATE, SE)
images=[img,img1]
titles=['Orginal','Dilation']
fig1 = plt.figure(figsize=(10, 10))
for idx in range(2):
    ax = fig1.add_subplot(1, 2, idx+1)
    ax.imshow(images[idx],cmap='gray')
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
