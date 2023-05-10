# Standard Python packages
import datetime
import pandas as pd
import numpy as np

# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df
from solgt.priceindex.repeatsales import get_RSI, convert_MT_data_to_ttp, create_and_solve_OLS_problem, convert_OLS_res_to_RSI
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import smooth_w

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from zone_analysis import get_zone_geometry, get_zone_neighbors, get_zone_controid_distances

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'


zone_div = 100

# TODO: Include all zones as identified by get_zone_geometry in the analysis


# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
# period = "quarterly"
period = "monthly"

def load_MT_data(zone_div=100):
    # Load raw data
    df_raw = get_parquet_as_df("C:\Code\data\MT.parquet")

    # Copy and preprocess
    df_a = df_raw.copy()
        # Remove entries without a valid grunnkrets
    df_a = df_a[~df_a[gr_krets].isna()].reset_index(drop=True)
        # Typecast to required types
    df_a[date_col] = df_a[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
    df_a[gr_krets] = df_a[gr_krets].astype(int)
    df_a["zone"] = df_a[gr_krets] // zone_div

    return df_a


def get_zone_OLS_and_count(df_a, t0, t1, period="monthly"):
    T_arr = np.arange(t0, t1)
    zone_arr = pd.Series(df_a["zone"].unique()).sort_values().reset_index(drop=True)
    zone_OLS = pd.DataFrame(index=T_arr, columns=zone_arr)
    zone_counts = pd.DataFrame(index=T_arr, columns=zone_arr)

    for (i, zone_no) in enumerate(zone_arr):
        df_zone = df_a[df_a["zone"] == zone_no].reset_index(drop=True)
        print("Zone number: " + str(zone_no) + ". Number of transactions: " + str(len(df_zone)))
        try:
            df_ttp = convert_MT_data_to_ttp(df_zone, period)
            OLS_res = create_and_solve_OLS_problem(df_ttp)

            OLS_res.set_index("t", inplace=True)
            zone_OLS[zone_no] = OLS_res["pred"]
            zone_counts[zone_no] = OLS_res["count"]
        except Exception as e:
            rsi = pd.DataFrame(columns=["count", "date", "price"])
            print("Exception: " + str(e))

    # Substitute NaN with 0
    zone_OLS.fillna(0, inplace=True)
    zone_counts.fillna(0, inplace=True)

    return zone_OLS, zone_counts


"""
MAIN PROGRAM
"""

# Load all data
df_a = load_MT_data(zone_div)

# Loop all zones and create RSI for them all
date0 = datetime.date(2014, 1, 1)
date1 = datetime.date(2023, 1, 1)
[t0, t1] = convert_date_to_t([date0, date1], period)
# Get OLS result from all MT data

df_ttp_a = convert_MT_data_to_ttp(df_a, period)
OLS_res_a = create_and_solve_OLS_problem(df_ttp_a)
OLS_res_a = OLS_res_a[(OLS_res_a["t"] >= t0) & (OLS_res_a["t"] < t1)].reset_index(drop=True)

# Create RSI for all zones
zone_OLS, zone_counts = get_zone_OLS_and_count(df_a, t0, t1, period)

# Normalize zone_OLS to start at 0
zone_OLS = zone_OLS - zone_OLS.iloc[0]

# Volume weighted price index based on zone neighbors
zones_geometry = get_zone_geometry(zone_div)
zones_neighbors = get_zone_neighbors(zones_geometry)
np.fill_diagonal(zones_neighbors.values, 0)

# Remove zones that has no repeated sales
cols = zone_OLS.columns
zones_neighbors = zones_neighbors[cols].loc[cols]

# Make zone_OLS_w as a copy of zone_OLS with all NaNs
zone_OLS_w = zone_OLS.copy()
zone_counts_w = zone_counts.copy()
zone_OLS_w[:] = np.nan
zone_counts_w[:] = np.nan

central_zone_w = 1

# TODO: Weight should be higher on the central zone
for zone in zone_OLS.columns:
    neighbors = zones_neighbors[zones_neighbors[zone] == 1].index

    neighbors_OLS_diff = zone_OLS[neighbors]
    neighbors_count = zone_counts[neighbors]

    weighted_sum = neighbors_OLS_diff.multiply(neighbors_count).sum(axis=1) + zone_OLS[zone] * central_zone_w * zone_counts[zone]
    count = neighbors_count.sum(axis=1) + central_zone_w * zone_counts[zone]

    zone_OLS_w[zone] = weighted_sum / count
    zone_counts_w[zone] = count


# Filter in time
OLS_res_a["pred_t"] = smooth_w(OLS_res_a["pred"], OLS_res_a["count"], 3)

zone_OLS_w_t = zone_OLS_w.copy()
for col in zone_OLS.columns:
    zone_OLS_w_t[col] = smooth_w(zone_OLS_w[col], zone_counts_w[col], 3)


# Normalize: Shift all columns so that they all are centered around zero
OLS_res_a["pred"] = OLS_res_a["pred"] - OLS_res_a["pred"].mean()
OLS_res_a["pred_t"] = OLS_res_a["pred_t"] - OLS_res_a["pred_t"].mean()
zone_OLS = zone_OLS.sub(zone_OLS.mean(axis=0), axis=1)
zone_OLS_w = zone_OLS_w.sub(zone_OLS_w.mean(axis=0), axis=1)
zone_OLS_w_t = zone_OLS_w_t.sub(zone_OLS_w_t.mean(axis=0), axis=1)

# Convert to RSI
rsi_a = convert_OLS_res_to_RSI(OLS_res_a, period)
rsi_zones = np.exp(zone_OLS)
rsi_zones_w = np.exp(zone_OLS_w)

# Print all price indexes for all zones
dates = convert_t_to_date(pd.Series(OLS_res_a["t"]), period)

# Figure 1
fig1 = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

fig1.append_trace(
    go.Scatter(x=OLS_res_a["t"], y=OLS_res_a["pred"], name="Price, all"),
    row=1,
    col=1,
)

for col in rsi_zones.columns:
    fig1.append_trace(
        go.Scatter(x=OLS_res_a["t"], y=zone_OLS[col], name="Price, zone " + str(col)),
        row=1,
        col=1,
    )
    
fig1.show()

# fig1.write_html("output/fig1_" + period + "_raw.html")


# Figure 2
fig2 = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

fig2.append_trace(
    go.Scatter(x=OLS_res_a["t"], y=OLS_res_a["pred"], name="Price, all"),
    row=1,
    col=1,
)

for col in rsi_zones_w.columns:
    fig2.append_trace(
        go.Scatter(x=OLS_res_a["t"], y=zone_OLS_w[col], name="Price, zone " + str(col)),
        row=1,
        col=1,
    )


fig2.show()

# fig2.write_html("output/fig2_" + period + "_spatial_filtered.html")



# Figure 3
fig3 = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)

fig3.append_trace(
    go.Scatter(x=OLS_res_a["t"], y=OLS_res_a["pred"], name="Price, all"),
    row=1,
    col=1,
)

fig3.append_trace(
    go.Scatter(x=OLS_res_a["t"], y=OLS_res_a["pred_t"], name="Price, all, filtered"),
    row=1,
    col=1,
)

for col in rsi_zones_w.columns:
    fig3.append_trace(
        go.Scatter(x=OLS_res_a["t"], y=zone_OLS_w_t[col], name="Price, zone " + str(col)),
        row=1,
        col=1,
    )


fig3.show()
# fig3.write_html("output/fig3_" + period + "_time_filtered.html")



# Plot counts
