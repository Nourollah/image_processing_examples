import cv2
from skimage.morphology import square, rectangle, diamond, disk, octagon, star
import matplotlib.pyplot as plt

SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))       # Rectangular Kernel with OpenCV
SE2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))    # Elliptical Kernel with OpenCV
SE3=cv2.getStructuringElement(cv2.MORPH_CROSS,(15,15))      # Cross-shaped Kernel with OpenCV

SE4=square(15)          # square Kernel with skimage
SE5=rectangle(15, 10)   # Rectangular Kernel with skimage
SE6=diamond(7)          # diamond Kernel with skimage
SE7=disk(7)             # disk Kernel with skimage
SE8=octagon(7, 4)       # octagon Kernel with skimage
SE9=star(5)             # star Kernel with skimage

# Visualize the elements.
SES=[SE1,SE2,SE3,SE4,SE5,SE6,SE7,SE8,SE9]
titles=['CV2_rectangle','CV2_Elliptical','CV2_Cross-shaped','skimage_square','skimage_rectangle','skimage_diamond','skimage_disk','skimage_octagon','skimage_star']
fig1 = plt.figure(figsize=(10, 10))
for idx in range(9):
    ax = fig1.add_subplot(3, 3, idx+1)
    ax.imshow(SES[idx], cmap="Paired", vmin=0, vmax=12)
    for i in range(SES[idx].shape[0]):
        for j in range(SES[idx].shape[1]):
            ax.text(j, i, SES[idx][i, j], ha="center", va="center", color="w")
    ax.set_axis_off()
    ax.set_title(titles[idx])
plt.show()
