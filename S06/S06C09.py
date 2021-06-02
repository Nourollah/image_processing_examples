import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read Image from file
img = cv2.imread("Lenna.jpg", cv2.IMREAD_GRAYSCALE)
I = cv2.normalize(img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX)
[y, x] = I.shape
# Create 2 mask for blurring image
Hv = np.ones((9, 1), np.float32) / 9
Hh = np.transpose(Hv)
# Apply mask to image
B_Ver = cv2.filter2D(I, -1, Hv)
B_Hor = cv2.filter2D(I, -1, Hh)
# Show blurred images
plt.subplot(121)
plt.imshow(B_Ver)
plt.subplot(122)
plt.imshow(B_Hor)
plt.show()
# Variation of the input image (Vertical direction)
D_F_Ver = np.abs(I[:, 0:x-2] - I[:, 2:x])
# Variation of the input image (Horizontal direction)
D_F_Hor = np.abs(I[0:y-2, :] - I[2:y, :])
# Variation of the Blurred image (Vertical direction)
D_B_Ver = np.abs(B_Ver[:, 0:x-2] - B_Ver[:, 2:x])
# Variation of the Blurred image (Horizontal direction)
D_B_Hor = np.abs(B_Hor[0:y-2, :] - B_Hor[2:y, :])
# Difference between two vertical variations of 2 images (input and blurred)
T_Ver = D_F_Ver - D_B_Ver
# Difference between two vertical horizontal of 2 images (input and blurred)
T_Hor = D_F_Hor - D_B_Hor

v_Ver = np.copy(T_Ver)
v_Ver[v_Ver < 0] = 0
v_Hor = np.copy(T_Hor)
v_Hor[v_Hor < 0] = 0

S_D_Ver = np.sum(np.sum(D_F_Ver[2:y, 2:x]))
S_D_Hor = np.sum(np.sum(D_F_Hor[2:y, 2:x]))
S_V_Ver = np.sum(np.sum(v_Ver[2:y, 2:x]))
S_V_Hor = np.sum(np.sum(v_Hor[2:y, 2:x]))

blur_F_Ver = (S_D_Ver - S_V_Ver) / S_D_Ver
blur_F_Hor = (S_D_Hor - S_V_Hor) / S_D_Hor
blur = max(blur_F_Ver, blur_F_Hor)