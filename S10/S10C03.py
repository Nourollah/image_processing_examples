import cv2
import matplotlib.pyplot as plt
img = cv2.imread("Finger.BMP",0)
SE = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
img1 = cv2.morphologyEx(img, cv2.MORPH_ERODE, SE)
images=[img,img1]
titles=['Orginal','Erosion']
fig1 = plt.figure(figsize=(10, 10))
for idx in range(2):
    ax = fig1.add_subplot(1, 2, idx+1)
    ax.imshow(images[idx],cmap='gray')
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
