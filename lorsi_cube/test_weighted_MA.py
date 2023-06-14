import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 1)
# Sinus of x
a = np.sin(x)
# Random numbers
b = np.random.rand(len(a))

# c = np.array([1, 2, 1])
c = np.array([1, 10, 1])
c = c / np.sum(c)

# Convolve a and b


d = np.convolve(a, c, mode="same")


# Plot the numpy array count
# plt.plot(a)
plt.plot(b)
plt.plot(c)
plt.plot(d)
plt.show()
