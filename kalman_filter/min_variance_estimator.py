import numpy as np
from numpy.linalg import inv
           
# Define the time points for start and end of each measurement
t_start = np.array([0, 1, 2, 3, 4])
t_end = np.array([2, 3, 4, 5, 6])

# Assume the current time t
t = 2.5

# Define the noise variance
sigma_n = 0.1

# Create the A matrix and b vector
A = np.zeros((len(t_start), len(t_start)))
b = np.zeros(len(t_start))

for k in range(len(t_start)):
    for i in range(len(t_start)):
        if i == k:
            A[k, i] = 2 * ((t_end[i] - t_start[i]) + sigma_n**2)
        else:
            A[k, i] = 2 * (min(t_end[k], t_end[i]) - min(t_end[k], t_start[i]) - min(t_start[k], t_end[i]) + min(t_start[k], t_start[i]))

    b[k] = 2 * (min(t, t_end[k]) - min(t, t_start[k]))

# Solve for the weights
alpha = np.matmul(inv(A), b)

print("Optimal weights:", alpha)