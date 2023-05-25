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
from compute_LORSI_for_zones import load_MT_data, score_RSI_split, get_OLS_and_count, get_zone_OLS_and_count, compute_zone_OLS_weighted

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

# Compute the weighted OLS based on the neighboring zones
OLS_z_w, OLS_z_count_w = compute_zone_OLS_weighted(OLS_z, OLS_z_count, zones_neighbors)

# Score the RSI
df_ttp_zone = score_RSI_split(df_MT, t0, t1, OLS_a, OLS_z)
df_ttp_zone_w = score_RSI_split(df_MT, t0, t1, OLS_a, OLS_z_w)


"""
CREATE FOR PERIOD QUARTERLY
"""
period = "quarterly"

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

# Compute the weighted OLS based on the neighboring zones
OLS_z_w, OLS_z_count_w = compute_zone_OLS_weighted(OLS_z, OLS_z_count, zones_neighbors)

# Score the RSI
df_ttp_zone_q = score_RSI_split(df_MT, t0, t1, OLS_a, OLS_z)
df_ttp_zone_q_w = score_RSI_split(df_MT, t0, t1, OLS_a, OLS_z_w)


# Measure how similar the two price indexes are: L1 norm
print("Original:")
print("L1 norm, dp-dp_est: ", abs(df_ttp_zone["dp_e"]).mean())
print("Monthly:")
print("L1 norm, dp-dp_est_z: ", abs(df_ttp_zone["dp_e_z"]).mean())
print("L1 norm, dp-dp_est_z_w: ", abs(df_ttp_zone_w["dp_e_z"]).mean())
print("Quarterly:")
print("L1 norm, dp-dp_est_z_q: ", abs(df_ttp_zone_q["dp_e_z"]).mean())
print("L1 norm, dp-dp_est_z_q_w: ", abs(df_ttp_zone_q_w["dp_e_z"]).mean())

