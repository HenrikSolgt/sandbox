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
from scipy import interpolate

from zone_analysis import get_zone_geometry, get_zone_neighbors
from create_OLS import load_MT_data, score_RSI_split, get_OLS_and_count, get_zone_OLS_and_count, compute_zone_OLS_weighted

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

zone_div = 100

# period = "quarterly"
period = "monthly"






def create_lpf_matrix_from_weights(weights):
    """
    Create a low pass filter matrix from a list of weights.
    """
    N = len(weights)
    M_int = np.zeros((N, N))
    for i in range(N):
        M_int[i, i] = weights.iloc[i]
        for j in [i-1, i+1]:
            if 0 <= j < N:
                M_int[i, j] = weights.iloc[j]

    # Divide each row by the sum of the row
    for i in range(N):
        M_int[i, :] = M_int[i, :] / np.sum(M_int[i, :])

    return M_int



def fill_in_missing_zones(OLS_z, OLS_z_count, zones_arr):
    """
    Fill in for the missing zones in OLS_z_m and OLS_z_count_m with zeros.
    """
    for zone in zones_arr:
        if zone not in OLS_z.keys():
            OLS_z[zone] = np.zeros(len(OLS_z))
            OLS_z_count[zone] = np.zeros(len(OLS_z_count)).astype(int)

    # Sort columns in OLS_z_m and OLS_z_count_m
    OLS_z = OLS_z[sorted(OLS_z.columns)]
    OLS_z_count = OLS_z_count[sorted(OLS_z_count.columns)]
    OLS_z, OLS_z_count

    return OLS_z, OLS_z_count


def get_OLS_and_count_for_period(date0, date1, period, compute_zones=False):
    [t0, t1] = convert_date_to_t([date0, date1], period)

    # Load data
    df_MT = load_MT_data()

    # Derived columns used by this module, and zone column
    df_MT = add_derived_MT_columns(df_MT, period, date0)
    df_MT["zone"] = df_MT["grunnkrets"] // zone_div

    if compute_zones:
        # Get OLS and count for the zones
        OLS, OLS_count = get_zone_OLS_and_count(df_MT, t0, t1) # Create OLS and count for the zones
    else:
        # Get OLS and count for the whole region
        OLS, OLS_count = get_OLS_and_count(df_MT, t0, t1)

    # Set date as index
    OLS = OLS.set_axis(convert_t_to_date(OLS.index, period))
    OLS_count = OLS_count.set_axis(convert_t_to_date(OLS_count.index, period))

    return OLS, OLS_count

"""
MAIN PROGRAM
"""

# Define time period
date0 = datetime.date(2014, 1, 1)
date1 = datetime.date(2022, 1, 1)

# Fetch information about the zones
zones_arr, zones_neighbors = get_zones_and_neighbors(zone_div=100)

# Get monthly OLS and count for whole region and count for the zones
OLS_a_m, OLS_count_a_m = get_OLS_and_count_for_period(date0, date1, "monthly", compute_zones=False)
OLS_z_m, OLS_count_z_m = get_OLS_and_count_for_period(date0, date1, "monthly", compute_zones=True)

# Get quarterly OLS and count for the zones
OLS_z_q, OLS_count_z_q = get_OLS_and_count_for_period(date0, date1, "quarterly", compute_zones=True)

# Fill in for missing zones
OLS_z_m, OLS_count_z_m = fill_in_missing_zones(OLS_z_m, OLS_count_z_m, zones_arr) # Fill in for missing zones
OLS_z_q, OLS_count_z_q = fill_in_missing_zones(OLS_z_q, OLS_count_z_q, zones_arr) # Fill in for missing zones


"""
Resample quarterly to monthly
"""
index_m = convert_date_to_t(OLS_a_m.index, 1, date0)
index_q = convert_date_to_t(OLS_z_q.index, 1, date0)
OLS_z_q_resamp = pd.DataFrame(index=index_m, columns=OLS_z_q.columns)

for col in OLS_z_q.columns:
    OLS_z_q_resamp[col] = interpolate.interp1d(index_q, OLS_z_q[col], fill_value='extrapolate')(index_m)

OLS_z_q_resamp.index = convert_t_to_date(OLS_z_q_resamp.index, 1, date0)

# Shift to start at 0
OLS_z_q_resamp = OLS_z_q_resamp - OLS_z_q_resamp.iloc[0, :]



"""
Create new OLS consisting of LPF and HPF
"""
zones_arr2 = [2, 11, 12, 16, 19]
OLS_f = pd.DataFrame().reindex_like(OLS_z_q_resamp)
for zone in zones_arr2:
    M_lpf = create_lpf_matrix_from_weights(OLS_count_z_m[zone])
    M_hpf = np.identity(len(M_lpf)) - M_lpf
    OLS_f[zone] = np.dot(M_lpf, OLS_z_q_resamp[zone]) + np.dot(M_hpf, OLS_a_m["pred"])



# Plot for selected zones
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig = fig.add_trace(go.Scatter(x=OLS_a_m.index, y=OLS_a_m["pred"], mode="lines", name="Monthly: All "), row=1, col=1)
for zone in zones_arr2:
    fig = fig.add_trace(go.Scatter(x=OLS_z_q_resamp.index, y=OLS_z_q_resamp[zone], mode="lines", name="Quarterly, resampled, " + str(zone)), row=1, col=1)
    fig = fig.add_trace(go.Scatter(x=OLS_f.index, y=OLS_f[zone], mode="lines", name="Filtered: " + str(zone)), row=1, col=1)

fig.show()

