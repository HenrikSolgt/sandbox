import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df_comb_new = pd.read_csv("C:\Code\py\data\RSI_comb.csv")
df_comb_new['date'] = pd.to_datetime(df_comb_new['date'])
df_comb_new.set_index('date', inplace=True)
df_comb_new.drop(columns=['Unnamed: 0'], inplace=True)

# Take the log of the data
df_comb_new_log = np.log(df_comb_new)

# Estimate the process noise and measurement noise covariance matrices
#Q = df_comb_new_log.diff().cov().to_numpy()
#R = (df_comb_new_log - df_comb_new_log.shift()).cov().to_numpy()*10
Q = np.diag([1, 1, 1])*.0001
R = np.diag([0.00045, 0.00048, 0.00281])

# Define the Kalman filter
def kalman_filter(y, Q, R):
    # Initialize state
    x = np.zeros(y.shape)
    
    # Initialize estimation error covariance
    P = np.zeros((y.shape[0], Q.shape[0], Q.shape[1]))
    P[0] = np.eye(Q.shape[0])

    # Kalman filter
    for t in range(1, y.shape[0]):
        # Prediction
        x_pred = x[t-1]
        P_pred = P[t-1] + Q

        # Kalman gain
        S = np.dot(np.dot(np.eye(Q.shape[0]), P_pred), np.eye(Q.shape[0]).T) + R
        K = np.dot(np.dot(P_pred, np.eye(Q.shape[0]).T), np.linalg.inv(S))

        # Update
        x[t] = x_pred + np.dot(K, (y[t] - np.dot(np.eye(Q.shape[0]), x_pred)))
        P[t] = np.dot((np.eye(Q.shape[0]) - np.dot(K, np.eye(Q.shape[0]))), P_pred)

    return x, P

# Apply the Kalman filter to the log-transformed data
estimated_log, P = kalman_filter(df_comb_new_log.to_numpy(), Q, R)

# Convert the state estimates back to the original scale
estimated_price_indices = np.exp(estimated_log)

# Create a dataframe for the estimated price indices
df_estimated = pd.DataFrame(estimated_price_indices, columns=df_comb_new_log.columns, index=df_comb_new_log.index)

# Plot the unfiltered and filtered timeseries
plt.figure(figsize=(14,7))
plt.plot(df_comb_new, linestyle='dotted')
plt.plot(df_estimated)
plt.title('Unfiltered and Filtered Repeat Sales Price Indices')
plt.xlabel('Date')
plt.ylabel('Price Index')
plt.legend(['RSI60 Unfiltered', 'RSI60-90 Unfiltered', 'RSI90 Unfiltered', 'RSI60 Filtered', 'RSI60-90 Filtered', 'RSI90 Filtered'])
plt.show()