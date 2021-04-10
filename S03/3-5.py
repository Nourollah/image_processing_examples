import numpy as np

x = [1, 2, 3, 4]
AF = np.fft.fft(x)
AFShift = np.fft.fftshift(AF)

print(f"Shifted AF: {AFShift}")
