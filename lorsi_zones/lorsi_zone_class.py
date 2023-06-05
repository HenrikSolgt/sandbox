# Standard Python packages
import datetime
import pandas as pd
import numpy as np

# Solgt packages
from solgt.priceindex.repeatsales import get_RSI, add_derived_MT_columns, get_repeated_idx, get_RS_idx_lines, get_df_ttp_from_RS_idx, create_and_solve_LORSI_OLS_problem
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import smooth_w
from solgt.db.MT_parquet import get_parquet_as_df, update_MT_parquet_file

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.model_selection import train_test_split

from zone_analysis import get_zones_and_neighbors


# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
postcode = "postcode"


def get_LORSI_and_count_for_zones(df, t0, t1):
    """
    Get LORSI and count for all matched transactions in DataFrame df, for the time period [t0, t1).
    All raw transactions from df are used. The filtering on time period [t0, t1) is performed after the transactions have been matched.
    Note that only the zones occuring in df are included in the output.
    Inputs:
        - df: DataFrame with columns "unitkey", "price_inc_debt", "zone", "t"
        - t0: Start time
        - t1: End time
    Returns:
        - lorsi: LORSI for all zones in df, as a numpy array
        - count: Number of transactions for all zones in df, as a numpy array
        - zones_arr: List of zones, as a numpy array
    """

    zones_arr = df["zone"].unique()
    zones_arr.sort()

    T_arr = np.arange(t0, t1)
    # Create empty dataframes
    zone_LORSI = pd.DataFrame(index=T_arr, columns=zones_arr)
    zone_counts = pd.DataFrame(index=T_arr, columns=zones_arr).fillna(0)

    for zone_no in zones_arr:
        # Filter the MT data for the current zone
        df_zone = df[df["zone"] == zone_no].reset_index(drop=True)
        df_zone = add_derived_MT_columns(df_zone)
        print("Zone number: " + str(zone_no) + ". Number of transactions: " + str(len(df_zone)))

        # Get repeated sales index and create df_ttp
        RS_idx = get_repeated_idx(df_zone)
        df_ttp = get_df_ttp_from_RS_idx(df_zone, RS_idx)

        if (len(df_ttp) > 0):
            LORSI_res = create_and_solve_LORSI_OLS_problem(df_ttp)
            LORSI_res = LORSI_res[(LORSI_res["t"] >= t0) & (LORSI_res["t"] < t1)].reset_index(drop=True)

            # Split LORSI_res into LORSI and count dataframes
            zone_LORSI[zone_no] = LORSI_res[["t", "pred"]].set_index(["t"]).reindex(T_arr).values
            zone_counts[zone_no] = LORSI_res[["t", "count"]].set_index(["t"]).reindex(T_arr, fill_value=0).values

    # Substitute NaN with 0
    zone_LORSI.fillna(0, inplace=True)
    zone_counts.fillna(0, inplace=True)

    return zone_LORSI.values, zone_counts.values, zones_arr


def default_zone_func(df_MT):
    """
    Default zone function. Returns a all-zero numpy array of same length as df_MT.
    """
    return np.zeros(len(df_MT))


def zone_func_div100(df_MT):
    return df_MT[gr_krets] // 100


class lorsi_zone:
    def compute_zone(self, df):
        """
        Computes the zone, using all information that it has available, including the zone information from the original df_MT.
        Takes DataFrame df of matched transactions as input, and returns a column of zones, with the same length as df.
        """ 
        res_zones = self.zone_func(df)
        # res_zones = zone_func_div100(df)
        res_idx_nan = res_zones[res_zones.isna()].index

        if self.src_MT is not None:
            src_zones  = self.zone_func(self.src_MT)
            src_idx_nan = src_zones[src_zones.isna()].index

            ok_entries = pd.concat([df.drop(res_idx_nan), df_MT.drop(src_idx_nan)])
            nan_entries = res_zones.loc[res_idx_nan]

        res_zones.loc[nan_entries] = nan_entries["postcode"]
        


    def __init__(self, df_MT=None, date0=None, date1=None, period="monthly", zone_func=default_zone_func, df_MT_all=None):
        """
        Create a zone_LORSI_class object. This object contains the LORSI for each zone, and the count of sales for each zone. 
        The matched transactions are loaded from df_MT, and only data between date0 and date1 is used.
        The default value of zone_func is a function returning all zeros, producing in a single zone: hence all transactions are treated to belong to the same zone.
        """
        if df_MT is not None:
            self.src_MT = df_MT   # Stores a reference to the original df_MT, which will be used for Zone identification later
            self.date0 = date0
            self.date1 = date1
            self.period = period
            self.zone_func = zone_func

            # Fill in the zone column using the zone_func
            df = df_MT.copy()
            df["zone"] = zone_func(df)

            if df_MT_all is not None:
                df_all = df_MT_all.copy()
                df_all["zone"]= zone_func(df_all)
            else:
                df_all = df
            df = fill_in_nan_zones(df, df_all)
            # TODO: Fix this: dropna should not be necessary
            df = df.dropna(subset=["zone"])

            # Add the columns derived from the matched transactions
            df = add_derived_MT_columns(df, period, date0)

            [self.t0, self.t1] = convert_date_to_t([date0, date1], period)
            LORSI, count = get_LORSI_and_count_for_zones(df, self.t0, self.t1) # Create OLS and count for the zones
            self.LORSI = LORSI.values
            self.count = count.values
            self.zones = LORSI.columns.values
            self.t = LORSI.index.values
        else:
            self.LORSI = None
            self.count = None
            self.zones = None
            self.zone_func = zone_func
            self.t = None




"""
MAIN PROGRAM
"""


# Define time period
date0 = datetime.date(2012, 1, 1)
date1 = datetime.date(2022, 1, 1)

period = "monthly"


df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[postcode] = df_MT[postcode].astype(int)

df_MT["zone"] = zone_func_div100(df_MT)
# Remove transactions with zone nan
df_MT = df_MT[df_MT["zone"].notna()].reset_index(drop=True)


df_MT = add_derived_MT_columns(df_MT, period, date0)
[t0, t1] = convert_date_to_t([date0, date1], period)


lorsi, count, zones_arr = get_LORSI_and_count_for_zones(df_MT, t0, t1)



src_MT = df_MT.copy()

zone_func = zone_func_div100


# Choose df as half of df_MT
df = df_MT.sample(frac=0.5).reset_index(drop=True).copy()

df["zone"] = zone_func(df)
# res_zones = zone_func_div100(df)
res_idx_nan = df[df["zone"] .isna()].index

if src_MT is not None:
    src_MT["zone"]  = zone_func(src_MT)
    src_idx_nan = src_MT[src_MT["zone"].isna()].index

    
    ok_entries = pd.concat([df.drop(res_idx_nan), src_MT.drop(src_idx_nan)])
    nan_entries = df.loc[res_idx_nan]

# TODO: Missing "zone" in ok_entries
df.loc[nan_entries.index, "zone"] = nan_entries["postcode"].map(ok_entries.groupby("postcode")["zone"].agg(lambda x: x.value_counts().index[0]))

res = df.loc[nan_entries.index, "zone"]