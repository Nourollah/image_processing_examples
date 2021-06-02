import numpy as np

# 3-1 code
# Define an array to apply Fourier transform on it
A = np.array([1, 2, 3, 4, 5])
# Take Fourier transform from A with default coefficient size
AF1 = np.fft.fft(A)
# Take Fourier transform from A with coefficient equal 8
AF2 = np.fft.fft(A, 8)

# Calculation of domain of AF1 from 3-1 code
a = np.abs(AF1)
# Calculation of phase angle
t = np.angle(AF1)
# Convert radian to degree
d = np.rad2deg(t)

