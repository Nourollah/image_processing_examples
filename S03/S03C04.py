import numpy as np

# Create numpy array
x = np.array([1, 2, 3, 4])
# Take Fourier transform from A with default coefficient size
AF = np.fft.fft(x)
# Shifting Fourier transform, Using numpy broadcasting in this line
x1 = np.array([1, 2, 3, 4]) * ((-1) ** np.arange(0, 4))
AF1 = np.fft.fft(x1)

print(f"AF:  {AF}\n")
print(f"AF1: {AF1}")
