import cv2
import numpy as np
import matplotlib.pyplot as plt

I = cv2.imread('hafez.jpeg')
if len(I.shape) == 3:
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
I = np.float32(I)

m, n = I.shape[:2]
# The initial point to grow
x, y = 100, 100
# The threshold value
Threshold = 12
# Output
Output = np.zeros(I.shape, dtype=np.uint8)
# The mean of the segmented region
region_mean = I[x, y]
# Number of pixels in region
region_size = 1
neg_pos = 0
# Distance of the newest segment pixel to the region mean
pixdist = 0
# 4 Neighbor locations
neigb = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])
'''Start region growing until distance between region 
and possible new pixel become higher than a certain threshold'''
while pixdist < Threshold:
    # Add new neighbors pixels
    for j in range(4):
        # Calculate the neighbour coordinate
        xn, yn = x + neigb[j, 0], y + neigb[j, 1]
        # Check if neighbour is inside or outside the image
        ins = (xn > 0) and (yn > 0) and (xn <= m) and (yn <= n)
        # Add neighbour if inside and not already part of the segmented area
        if ins and (Output[xn - 1, yn - 1] == 0):
            neg_pos += 1
            neg_list = np.array([xn, yn, I[xn - 1, yn - 1]], dtype=np.float32)
            neg_list = np.expand_dims(neg_list, axis=0)
            Output[xn - 1, yn - 1] = 1
    # Add pixel with intensity nearest to the mean of the region
    dist = np.abs(neg_list[:, 2] - region_mean)
    index = np.argmin(dist)
    pixdist = dist[index]
    Output[x - 1, y - 1] = 2
    region_size += 1
    # Calculate the new mean of the region
    region_mean = (region_mean * region_size + neg_list[index, 2]) / (region_size + 1)
    # Save the x and y coordinates of the pixel
    x, y = int(neg_list[index, 0]), int(neg_list[index, 1])
    # Remove the pixel from the neighbour (check) list
    neg_list[index] = neg_list[-1]
    neg_pos -= 1
# Return the segmented area as logical matrix
Output = Output > 1
plt.subplot(1, 2, 1)
plt.imshow(np.uint8(I))
plt.title('Original Image')
contours, _ = cv2.findContours(Output.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(I, contours, -1, (0, 0, 255), 2)
plt.subplot(1, 2, 2)
plt.imshow(Output.astype(np.uint8) * 255)
plt.title('Segmented Image')
plt.show()
