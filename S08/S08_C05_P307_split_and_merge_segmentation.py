import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt


def func(block):
    block = np.array(block, dtype=np.float32)
    block -= block.mean()
    block /= block.std()
    return np.abs(block).sum()


# Load image
I = cv2.imread('AzadiTowerGray.jpg')

# Convert to grayscale if necessary
if len(I.shape) == 3:
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

# Resize to 512x512
I = cv2.resize(I, (512, 512))

# Store the original image for later use
IORG = I.copy()

# Apply quadtree decomposition
S = pywt.dwtn(I, 'haar')

# Initialize blocks with zeros
blocks = np.zeros_like(I)

# Loop through the decomposition levels
for dim in [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]:

    # Get the number of blocks with the current dimension
    numblocks = np.sum([key == (dim, dim) for key in S.keys()])

    # If there are blocks with the current dimension
    if numblocks > 0:

        # Initialize the values for the current dimension
        values = np.ones((dim, dim, numblocks), dtype=np.uint8)
        values[1:dim, 1:dim, :] = 0

        # Set the blocks with the current dimension to 1
        for key in S.keys():
            if key == (dim, dim):
                row, col = S[key]['dyadic']
                blocks[row:row + dim, col:col + dim] = 1

# Set the border blocks to 1
blocks[-1, :] = 1
blocks[:, -1] = 1

# Display the block image
plt.imshow(blocks)
plt.title('Blocks')
plt.show()

# Combine the blocks with the original image
I2 = blocks.astype(np.float32) * 255 + IORG.astype(np.float32)

# Display the combined image
plt.imshow(I2.astype(np.uint8))
plt.title('Combined')
plt.show()
