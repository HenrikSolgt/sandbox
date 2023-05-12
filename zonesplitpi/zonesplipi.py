# Standard Python packages
import datetime
import pandas as pd
import numpy as np

# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df
from solgt.priceindex.repeatsales import get_RSI, add_derived_MT_columns, get_repeated_idx, get_df_ttp_from_RS_idx, create_and_solve_OLS_problem
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import smooth_w

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt

from zone_analysis import get_zone_geometry, get_zone_neighbors

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

zone_div = 100

# TODO: Include all zones as identified by get_zone_geometry in the analysis

# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
columns_of_interest = [key_col, date_col, price_col, gr_krets]
# period = "quarterly"
period = "monthly"


def load_MT_data(zone_div=100, period="monthly", date0=None):
    # Load raw data
    df_raw = get_parquet_as_df("C:\Code\data\MT.parquet")

    # Copy and preprocess
    df = df_raw[columns_of_interest].copy()

    # Preprocess: manipulation of already existing columns
        # Remove entries without a valid grunnkrets
    df = df[~df[gr_krets].isna()].reset_index(drop=True)
        # Typecast to required types
    df[date_col] = df[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
    df[gr_krets] = df[gr_krets].astype(int)

    # Derived columns used by this module, and zone column
    df = add_derived_MT_columns(df, period, date0)
    df["zone"] = df[gr_krets] // zone_div

    return df


def get_OLS_and_count(df, t0, t1):
    """
    Get OLS and count for a all matched transactions in DataFrame df, for the time period [t0, t1).
    All raw transactions from df is used. The filtering is performed after the transactions have been matched.
    Inputs:
        - df: DataFrame with columns "id", "y", "t"
        - t0: Start time
        - t1: End time
    """

    T_arr = np.arange(t0, t1)

    # Get repeated sales index and create df_ttp
    R_idx = get_repeated_idx(df)
    df_ttp = get_df_ttp_from_RS_idx(df, R_idx)

    if (len(df_ttp) > 0):
        OLS_res = create_and_solve_OLS_problem(df_ttp)
        OLS_res = OLS_res[(OLS_res["t"] >= t0) & (OLS_res["t"] < t1)].reset_index(drop=True)

        OLS = OLS_res[["t", "pred"]].set_index(["t"]).reindex(T_arr)
        OLS_count = OLS_res[["t", "count"]].set_index(["t"]).reindex(T_arr, fill_value=0)

        return OLS, OLS_count
    else:
        return pd.DataFrame(index=T_arr, columns=["pred"]), pd.DataFrame(index=T_arr, columns=["count"]).fillna(0)



# Create OLS for all zones
def get_zone_OLS_and_count(df, t0, t1, zones_arr):
    """
    Get OLS and count for a all matched transactions in DataFrame df, for the time period [t0, t1).
    All raw transactions from df is used. The filtering on time period [t0, t1) is performed after the transactions have been matched.
    Inputs:
        - df: DataFrame with columns "unitkey", "price_inc_debt", "zone", "t"
        - t0: Start time
        - t1: End time
        - zones_arr: Array of zones to compute OLS for
    """

    T_arr = np.arange(t0, t1)
    # Create empty dataframes
    zone_OLS = pd.DataFrame(index=T_arr, columns=zones_arr)
    zone_counts = pd.DataFrame(index=T_arr, columns=zones_arr)

    for zone_no in zones_arr:
        # Filter the MT data for the current zone
        df_zone = df[df["zone"] == zone_no].reset_index(drop=True)
        print("Zone number: " + str(zone_no) + ". Number of transactions: " + str(len(df_zone)))

        OLS, OLS_count = get_OLS_and_count(df_zone, t0, t1)

        zone_OLS[zone_no] = OLS["pred"]
        zone_counts[zone_no] = OLS_count["count"]

    # Substitute NaN with 0
    zone_OLS.fillna(0, inplace=True)
    zone_counts.fillna(0, inplace=True)

    # Convert to int
    zone_counts = zone_counts.astype(int)

    # Normalize zone_OLS to start at 0
    zone_OLS = zone_OLS - zone_OLS.iloc[0]

    return zone_OLS, zone_counts



def compute_zone_OLS_weighted(OLS_z, OLS_z_count):
    """
    Compute a volume-weighted OLS for all zones in OLS_z.
    Each zone is weighted by the number of transactions in the zone itself and its neighboring zones.
    Inputs:
        - OLS_z: DataFrame of OLS values with index "t" and zone numbers as column names
        - OLS_z_count: DataFrame of counts with index "t" and zone numbers as column names
    Returns:
        - OLS_z_w: DataFrame of volume-weighted OLS values with index "t" and zone numbers as column names
        - OLS_z_count_w: DataFrame with total number of transactions used in computation of OLS_z_w
    """

    OLS_z_w = OLS_z.copy()
    OLS_z_count_w = OLS_z_count.copy()
    OLS_z_w[:] = np.NaN
    OLS_z_count_w[:] = np.NaN

    central_zone_w = 1
    # Volume-weighted OLS by neighboring zones
    for zone in OLS_z.columns:
        neighbors = zones_neighbors[zones_neighbors[zone] == 1].index

        neighbors_OLS_diff = OLS_z[neighbors]
        neighbors_count = OLS_z_count[neighbors]

        weighted_sum = neighbors_OLS_diff.multiply(neighbors_count).sum(axis=1) + OLS_z[zone] * central_zone_w * OLS_z_count[zone]
        count = neighbors_count.sum(axis=1) + central_zone_w * OLS_z_count[zone]

        OLS_z_w[zone] = weighted_sum / count
        OLS_z_count_w[zone] = count.astype(int)

    return OLS_z_w, OLS_z_count_w



"""
MAIN PROGRAM
"""


# Define time period
date0 = datetime.date(2014, 1, 1)
date1 = datetime.date(2023, 1, 1)
[t0, t1] = convert_date_to_t([date0, date1], period)

# Load MT data in the correct formats and with time index "t"
df_MT = load_MT_data(zone_div, period, date0)

# Get OLS and count for MT data
OLS_a, OLS_a_count = get_OLS_and_count(df_MT, t0, t1)

# Fetch information about the zones
zones_geometry = get_zone_geometry(zone_div)
zones_arr = zones_geometry["zone"]
zones_neighbors = get_zone_neighbors(zones_geometry)

# Create OLS for all zones
OLS_z, OLS_z_count = get_zone_OLS_and_count(df_MT, t0, t1, zones_arr)

# Compute the weighted OLS based on the neighboring zones
OLS_z_w, OLS_z_count_w = compute_zone_OLS_weighted(OLS_z, OLS_z_count)


"""
Test the new price indexes agains the old one on a set of transactions
"""

# Get repeated sales index
R_idx = get_repeated_idx(df_MT)

# Convert repeated sales to ttp format by extracting information from the dataframe df, and add the zone information
df_ttp_zone = get_df_ttp_from_RS_idx(df_MT, R_idx)
df_ttp_zone["zone"] = df_MT.iloc[R_idx["I0"]]["zone"].reset_index(drop=True)

# Filter away all transactions that are not in the time period [t0, t1)
df_ttp_zone = df_ttp_zone[(df_ttp_zone["t0"] >= t0) & (df_ttp_zone["t0"] < t1)].reset_index(drop=True)
df_ttp_zone = df_ttp_zone[(df_ttp_zone["t1"] >= t0) & (df_ttp_zone["t1"] < t1)].reset_index(drop=True)

# Use the above information to compute the estimated dp from the OLS
pred0 = np.zeros(len(df_ttp_zone))
pred1 = np.zeros(len(df_ttp_zone))
pred0_z = np.zeros(len(df_ttp_zone))
pred1_z = np.zeros(len(df_ttp_zone))

for i in range(len(df_ttp_zone)):
    pred0[i] = OLS_a.loc[df_ttp_zone["t0"][i]]
    pred1[i] = OLS_a.loc[df_ttp_zone["t1"][i]]

    pred0_z[i] = OLS_z_w[df_ttp_zone["zone"][i]].loc[df_ttp_zone["t0"][i]]
    pred1_z[i] = OLS_z_w[df_ttp_zone["zone"][i]].loc[df_ttp_zone["t1"][i]]

df_ttp_zone["dp_est"] = pred1 - pred0
df_ttp_zone["dp_est_z"] = pred1_z - pred0_z
df_ttp_zone["dp_e"] = df_ttp_zone["dp"] - df_ttp_zone["dp_est"]
df_ttp_zone["dp_e_z"] = df_ttp_zone["dp"] - df_ttp_zone["dp_est_z"]

# L1 norm:
df_ttp_zone["L1_norm"] = (np.abs(df_ttp_zone["dp_e"] / df_ttp_zone["dp"])).sum() / len(df_ttp_zone)
df_ttp_zone["L1_norm_z"] = (np.abs(df_ttp_zone["dp_e_z"] / df_ttp_zone["dp"])).sum() / len(df_ttp_zone)

# Plot dp vs dp_est
plt.figure()
plt.scatter(df_ttp_zone["dp"], df_ttp_zone["dp_est"], s=1)
plt.xlabel("dp")
plt.ylabel("dp_est")
plt.title("dp vs dp_est")
plt.show(block=False)

# Plot dp vs dp_est
plt.figure()
plt.scatter(df_ttp_zone["dp"], df_ttp_zone["dp_est_z"], s=1)
plt.xlabel("dp")
plt.ylabel("dp_est_z")
plt.title("dp vs dp_est_z")
plt.show(block=False)

# Plot dp vs dp_est
plt.figure()
plt.scatter(df_ttp_zone["dp_est"], df_ttp_zone["dp_est_z"], s=1)
plt.xlabel("dp_est")
plt.ylabel("dp_est_z")
plt.title("dp_est vs dp_est_z")
plt.show(block=False)


# Measure how similar the two price indexes are
print("Correlation between dp and dp_est: ", df_ttp_zone["dp"].corr(df_ttp_zone["dp_est"]))
print("Correlation between dp and dp_est_z: ", df_ttp_zone["dp"].corr(df_ttp_zone["dp_est_z"]))



# Compute MAE and MAPE
MAE = np.mean(np.abs(df_ttp_zone["dp_e"]))
MAE_z = np.mean(np.abs(df_ttp_zone["dp_e_z"]))