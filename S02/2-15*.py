import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image from file in gray scale mode and  convert to numpy array
im1 = np.array(cv2.imread("11.png", cv2.IMREAD_GRAYSCALE))
im2 = np.array(cv2.imread("22.png", cv2.IMREAD_GRAYSCALE))
p = cv2.calcHist()

