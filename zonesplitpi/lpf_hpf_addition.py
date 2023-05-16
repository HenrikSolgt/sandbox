# Standard Python packages
import datetime
import pandas as pd
import numpy as np

# Solgt packages
from solgt.priceindex.repeatsales import get_RSI, add_derived_MT_columns, get_repeated_idx, get_df_ttp_from_RS_idx, create_and_solve_OLS_problem
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import smooth_w

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from zone_analysis import get_zone_geometry, get_zone_neighbors
from create_OLS import load_MT_data, score_RSI_split, get_OLS_and_count, get_zone_OLS_and_count, compute_zone_OLS_weighted

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

zone_div = 100

# TODO: Do a test-train-split: Train on just a subset of the data, and test on the rest

period = "quarterly"
# period = "monthly"





"""
MAIN PROGRAM
"""

# Define time period
date0 = datetime.date(2014, 1, 1)
date1 = datetime.date(2023, 1, 1)

"""
CREATE FOR PERIOD MONTHLY
"""
period = "monthly"

[t0, t1] = convert_date_to_t([date0, date1], period)
# Load MT data in the correct formats and with time index "t"
df_MT = load_MT_data(zone_div, period, date0)

# Get list of repeated sales
R_idx = get_repeated_idx(df_MT)

# Get OLS and count for MT data
OLS_a, OLS_a_count = get_OLS_and_count(df_MT, t0, t1)

# Fetch information about the zones
zones_geometry = get_zone_geometry(zone_div)
zones_arr = zones_geometry["zone"]
zones_neighbors = get_zone_neighbors(zones_geometry)

# Create OLS for all zones
OLS_z, OLS_z_count = get_zone_OLS_and_count(df_MT, t0, t1, zones_arr)


# Create a smoothing weight matrix based on the zone counts
zone = 9
OLS = OLS_z[zone]
count = OLS_z_count[zone]

N = len(OLS)
M = np.zeros((N, N))
for i in range(N):
    M[i, i] = count.iloc[i]
    for j in [i-1, i+1]:
        if 0 <= j < N:
            M[i, j] = count.iloc[j]

# Divide each row by the sum of the row
for i in range(N):
    M[i, :] = M[i, :] / np.sum(M[i, :])

OLS_w = np.dot(M, OLS)

# Plot the OLS and OLS_w using go
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig = fig.add_trace(go.Scatter(x=OLS.index, y=OLS, mode="lines", name="OLS"), row=1, col=1)
fig = fig.add_trace(go.Scatter(x=OLS.index, y=OLS_w, mode="lines", name="OLS_w"), row=1, col=1)
fig.show()