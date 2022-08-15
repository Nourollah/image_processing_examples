import cv2
import matplotlib.pyplot as plt
img = cv2.imread("Amirkola_City.bmp",0)

SE = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#SE = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

# Denoising (Openning and Closing)
imgO=cv2.morphologyEx(img, cv2.MORPH_OPEN, SE)
imgDenois=cv2.morphologyEx(imgO, cv2.MORPH_CLOSE, SE)

# Visualize
images=[img,imgDenois]
titles=['Original','Denoised']
fig1 = plt.figure(figsize=(10, 10))
for idx in range(2):
    ax = fig1.add_subplot(1, 2, idx+1)
    ax.imshow(images[idx],cmap='gray')
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
