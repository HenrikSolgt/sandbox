import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import graph object
import plotly.graph_objs as go

# Load the data
df_comb_new = pd.read_csv("C:\Code\py\data\RSI_comb.csv")
df_comb_new['date'] = pd.to_datetime(df_comb_new['date'])
df_comb_new.set_index('date', inplace=True)
df_comb_new.drop(columns=['Unnamed: 0'], inplace=True)

# Set the log-mean to zero

RSI_cols = ['RSI0', 'RSI60', 'RSI90']

dummy = np.log(df_comb_new[RSI_cols])
df_comb_new_log = dummy - dummy.mean()
df_comb_new[RSI_cols] = np.exp(df_comb_new_log)

# Estimate the process noise and measurement noise covariance matrices
Q = df_comb_new_log.diff().cov()
R = (df_comb_new_log - df_comb_new_log.shift()).cov()*10

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
y = df_comb_new_log
estimated_log, P = kalman_filter(df_comb_new_log.to_numpy(), Q.to_numpy(), R.to_numpy())

# Convert the state estimates back to the original scale
estimated_price_indices = np.exp(estimated_log)

# Create a dataframe for the estimated price indices
df_estimated = pd.DataFrame(estimated_price_indices, columns=df_comb_new_log.columns, index=df_comb_new_log.index)



# Plot the unfiltered and filtered timeseries
fig = go.Figure()
fig = fig.add_trace(go.Scatter(x=df_comb_new.index, y=df_comb_new['RSI0'], mode='lines', name='RSI0 Unfiltered'))
fig = fig.add_trace(go.Scatter(x=df_comb_new.index, y=df_comb_new['RSI60'], mode='lines', name='RSI60 Unfiltered'))
fig = fig.add_trace(go.Scatter(x=df_comb_new.index, y=df_comb_new['RSI90'], mode='lines', name='RSI90 Unfiltered'))
fig = fig.add_trace(go.Scatter(x=df_estimated.index, y=df_estimated['RSI0'], mode='lines', name='RSI0 Filtered'))
fig = fig.add_trace(go.Scatter(x=df_estimated.index, y=df_estimated['RSI60'], mode='lines', name='RSI60 Filtered'))
fig = fig.add_trace(go.Scatter(x=df_estimated.index, y=df_estimated['RSI90'], mode='lines', name='RSI90 Filtered'))
fig.show()



# Make a random dataframe a
a = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))

a.diff() - (a - a.shift())


# plt.figure(figsize=(14,7))
# plt.plot(df_comb_new, linestyle='dotted')
# plt.plot(df_estimated)
# plt.title('Unfiltered and Filtered Repeat Sales Price Indices')
# plt.xlabel('Date')
# plt.ylabel('Price Index')
# plt.legend(['RSI0 Unfiltered', 'RSI60 Unfiltered', 'RSI90 Unfiltered', 'RSI0 Filtered', 'RSI60 Filtered', 'RSI90 Filtered'])
# plt.show()