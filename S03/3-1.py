import numpy as np

# Define an array to apply Fourier transform on it
A = np.array([1, 2, 3, 4, 5])
# Take Fourier transform from A with default coefficient size
AF1 = np.fft.fft(A)
# Take Fourier transform from A with coefficient equal 8
AF2 = np.fft.fft(A, 8)

