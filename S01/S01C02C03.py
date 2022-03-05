import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.gray()


def img2bitmap(img: np.ndarray) -> list:
    if img.dtype != np.uint8 or img.ndim > 2:
        raise ValueError("Image is not uint8 or gray")
    bit_mat = [np.zeros(img.shape, dtype=np.uint8) for _ in range(8)]
    for row_number in range(img.shape[0]):
        for column_number in range(img.shape[1]):
            binary = format(img[row_number][column_number], 'b')
            for idx, bit in enumerate("".join(reversed(binary))[:]):
                bit_mat[idx][row_number, column_number] = 2 ** idx if int(bit) == 1 else 0
    return bit_mat


# Read image from file in default mode
img = cv2.imread('AzadiTower.jpg', cv2.IMREAD_GRAYSCALE)
# Make empty list to buffering bit plane
out = img2bitmap(img)

# Show every bit plane as separated images
plt.figure(1)
plt.subplot(2, 4, 1)
for i in range(8):
    plt.subplot(2, 4, i + 1)
    # Convert [0 - 255] to [0.0 - 1.0]
    out[i] = cv2.normalize(out[i].astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    plt.imshow(np.array(out[i]))
    plt.title(f"bit plane {i}")
plt.show()

OriginalImage = np.zeros(img.shape, dtype=np.uint8)
for i in range(OriginalImage.shape[0]):
    for j in range(OriginalImage.shape[1]):
        for data in range(8):
            x = np.array([OriginalImage[i, j]], dtype=np.uint8)
            data = np.array([data], dtype=np.uint8)
            flag = np.array([0 if out[data[0]][i, j] == 0 else 1], dtype=np.uint8)
            mask = flag << data[0]
            x[0] = (x[0] & ~mask) | ((flag[0] << data[0]) & mask)
            OriginalImage[i, j] = x[0]

plt.figure(2)
plt.imshow(OriginalImage)
plt.title("Original image")
plt.show()
