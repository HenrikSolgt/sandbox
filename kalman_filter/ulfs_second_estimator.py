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

sigma_n_zero = 0


# Define the parameters
np.random.seed(0) # for reproducibility
t_start = np.sort(np.random.uniform(0, 2*np.pi, 50))
t_end = np.sort(np.random.uniform(0, 2*np.pi, 50))

# Ensure that t_end is always greater than t_start
mask = t_start > t_end
t_start[mask], t_end[mask] = t_end[mask], t_start[mask]

# Generate a sine wave as the true Brownian motion
b = np.sin(np.linspace(0, 2*np.pi, int(t_end[-1]*20+2)))

# Create the noisy measurements with zero noise variance
y_zero_noise = np.array([b[int(ti1*20)] - b[int(ti0*20)] + np.random.normal(0, sigma_n_zero) for ti0, ti1 in zip(t_start, t_end)])


# Plot t_start and t_end
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.scatter(t_start, np.zeros_like(t_start), label='t_start')
plt.scatter(t_end, np.zeros_like(t_end), label='t_end')
plt.xlabel('Time')
plt.ylabel('b(t)')
plt.legend()
plt.title('True Brownian Motion with t_start and t_end')
plt.grid(True)
plt.show()



# Calculate the weights and the estimated Brownian motion for each time point
b_hat_zero_noise = np.zeros_like(b)
for t in range(len(b)):
    # Create the A matrix and b vector
    A = np.zeros((len(t_start), len(t_start)))
    b_vector = np.zeros(len(t_start))

    for k in range(len(t_start)):
        for i in range(len(t_start)):
            if i == k:
                A[k, i] = 2 * ((t_end[i] - t_start[i]) + sigma_n_zero**2)
            else:
                A[k, i] = 2 * (min(t_end[k], t_end[i]) - min(t_end[k], t_start[i]) - min(t_start[k], t_end[i]) + min(t_start[k], t_start[i]))

        b_vector[k] = 2 * (min(t/20, t_end[k]) - min(t/20, t_start[k]))

    # Solve for the weights
    alpha = np.matmul(inv(A), b_vector)

    # Construct the estimator as a weighted sum of the measurements
    b_hat_zero_noise[t] = np.sum(alpha * y_zero_noise)

# Shift the Brownian motion and the estimated Brownian motion so that they start at 0
b = b - b[0]
b_hat_zero_noise = b_hat_zero_noise - b_hat_zero_noise[0]

# Print the true and estimated Brownian motion with zero noise
print("True Brownian motion:", b)
print("Estimated Brownian motion with zero noise:", b_hat_zero_noise)

# b = b - b.mean()
b_hat_zero_noise2 = b_hat_zero_noise - b_hat_zero_noise.mean()


import matplotlib.pyplot as plt
# Plot the true and estimated Brownian motion with zero noise
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, 2*np.pi, int(t_end[-1]*20+2)), b, label='True Brownian motion')
plt.plot(np.linspace(0, 2*np.pi, int(t_end[-1]*20+2)), b_hat_zero_noise2, label='Estimated Brownian motion with zero noise')
plt.xlabel('Time')
plt.ylabel('b(t)')
plt.legend()
plt.title('True and Estimated Brownian Motion with Zero Noise as a Function of Time')
plt.grid(True)
plt.show()