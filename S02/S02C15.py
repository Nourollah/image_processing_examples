import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
im1 = cv2.imread("11.png", cv2.IMREAD_GRAYSCALE)
im2 = cv2.imread("22.png", cv2.IMREAD_GRAYSCALE)
p = cv2.calcHist([im1], [0], None, [256], [0, 256])
p = np.transpose(p) + 256
q = cv2.calcHist([im2], [0], None, [256], [0, 256])
q = np.transpose(q) + 256

newp = p / np.sum(p)
newq = q / np.sum(q)
t1 = newp * np.log2(newp)
t2= newq * np.log2(newq)
t3 = (newp + newq) * np.log2((newp + newq) / 2)
dis = (t1 + t2 - t3)
# Calc Jensen
J = (1 / 2) * np.sum(dis, 1)

