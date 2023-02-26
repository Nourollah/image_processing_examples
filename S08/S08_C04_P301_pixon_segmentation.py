import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

I = imread('hafez.jpeg')

# Convert RGB image to grayscale, if necessary
if len(I.shape) == 3:
    I = rgb2gray(I)

# Convert image to double for subsequent computations
I = I.astype(float)

# Get image dimensions
rows, cols = I.shape

# Initialize label image and label counter
Lp = np.zeros((rows, cols), dtype=int)
x = 0  # label counter

# Initialize pixel on/off map
mat = np.zeros((rows, cols), dtype=int)

# Initialize threshold for region growing
THR = 30

# For each pixel in the image
for i in range(rows):
    for j in range(cols):

        # If the pixel has not been labeled
        if Lp[i, j] == 0:

            # Start a new region
            x += 1

            # Label the current pixel and store its intensity and size
            Lp[i, j] = x
            Pixon = np.array([I[i, j], 1], dtype=float)

            # Initialize list of pixel coordinates in the region
            pix_coords = [(i, j)]

            # While there are unlabeled pixels adjacent to the region
            while pix_coords:
                a, b = pix_coords.pop()

                # Check each neighboring pixel
                for dr, dc in [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]:
                    r, c = a + dr, b + dc

                    # If the neighboring pixel is within the image bounds and has not been labeled
                    if 0 <= r < rows and 0 <= c < cols and Lp[r, c] == 0:

                        # If the neighboring pixel is similar in intensity to the current region
                        if abs(Pixon[0] / Pixon[1] - I[r, c]) < THR:
                            # Label the neighboring pixel and add it to the current region
                            Lp[r, c] = x
                            Pixon[0] += I[r, c]
                            Pixon[1] += 1
                            pix_coords.append((r, c))

            # After the region is fully grown, set the intensity of all pixels in the region to the average intensity
            intensity = Pixon[0] / Pixon[1]
            for a, b in pix_coords:
                I[a, b] = intensity

# Show the segmented image
plt.imshow(I, cmap='gray')
plt.axis('off')
plt.show()
