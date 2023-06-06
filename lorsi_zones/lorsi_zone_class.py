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

    t_arr = np.arange(t0, t1)
    # Create empty dataframes
    zone_LORSI = pd.DataFrame(index=t_arr, columns=zones_arr)
    zone_counts = pd.DataFrame(index=t_arr, columns=zones_arr).fillna(0)

    for zone_no in zones_arr:
        # Filter the MT data for the current zone
        df_zone = df[df["zone"] == zone_no].reset_index(drop=True)
        print("Zone number: " + str(zone_no) + ". Number of transactions: " + str(len(df_zone)))

        # Get repeated sales index and create df_ttp
        RS_idx = get_repeated_idx(df_zone)
        df_ttp = get_df_ttp_from_RS_idx(df_zone, RS_idx)

        if (len(df_ttp) > 0):
            LORSI_res = create_and_solve_LORSI_OLS_problem(df_ttp)
            LORSI_res = LORSI_res[(LORSI_res["t"] >= t0) & (LORSI_res["t"] < t1)].reset_index(drop=True) # Filters wrongly

            # Split LORSI_res into LORSI and count dataframes
            zone_LORSI[zone_no] = LORSI_res[["t", "pred"]].set_index(["t"]).reindex(t_arr).values
            zone_counts[zone_no] = LORSI_res[["t", "count"]].set_index(["t"]).reindex(t_arr, fill_value=0).values

    # Substitute NaN with 0
    zone_LORSI.fillna(0, inplace=True)
    zone_counts.fillna(0, inplace=True)

    return zone_LORSI.values, zone_counts.values, zones_arr, t_arr


def default_zone_func(df_MT):
    """
    Default zone function. Returns a all-zero numpy array of same length as df_MT.
    """
    return np.zeros(len(df_MT))


def zone_func_div100(df_MT):
    return df_MT[gr_krets] // 100


class LORSI_zone_class:
    def compute_zones(self, df, src_MT=None):
        """
        Computes the zones for the matched transactions in df, using all information that it has available, including the zone information from the original df_MT.
        Takes DataFrame df of matched transactions as input, and returns a column of zones with the same index and length as df
        """ 

        res = df
        res["zone"] = self.zone_func(res)
        res_idx_nan = res[res["zone"].isna()].index

        if src_MT is not None:
            src_MT["zone"] = self.zone_func(src_MT)
            src_idx_nan = src_MT[src_MT["zone"].isna()].index

            ok_entries = pd.concat([res.drop(res_idx_nan), src_MT.drop(src_idx_nan)])
        else:
            ok_entries = res.drop(res_idx_nan)

        nan_entries = res.loc[res_idx_nan]
        res.loc[res_idx_nan, "zone"] = nan_entries["postcode"].map(ok_entries.groupby("postcode")["zone"].agg(lambda x: x.value_counts().index[0]))

        return res["zone"]
        

    def __init__(self, df_MT=None, date0=None, date1=None, period="monthly", zone_func=default_zone_func):
        """
        Create a LORSI_zone_class object. This object contains the LORSI for each zone, and the count of sales for each zone. 
        The matched transactions are loaded from df_MT, and only data between date0 and date1 is used.
        The default value of zone_func is a function returning all zeros, producing in a single zone: hence all transactions are treated to belong to the same zone.
        """
        if df_MT is not None:
            # Store the input data
            self.src_MT = df_MT   # Stores a reference to the original df_MT, which will be used for Zone identification later
            self.date0 = date0
            self.date1 = date1
            self.period = period
            self.zone_func = zone_func

            # Fill in the zone column using the zone_func
            df = df_MT.copy()
            df["zone"] = self.compute_zones(df, src_MT=None)

            # Add the columns derived from the matched transactions
            df = add_derived_MT_columns(df, period, date0)

            # Convert date0 and date1 to t0 and t1 and get LORSI and count for the zones
            [t0, t1] = convert_date_to_t([date0, date1], period)
            [self.LORSI, self.count, self.zones_arr, self.t_arr] = get_LORSI_and_count_for_zones(df, t0, t1) # Create OLS and count for the zones

            # Store source material and the computed zones
            self.src_MT_zones = df["zone"]   # A list of zones for each matched transaction in src_MT
        else:
            # Create an empty object
            # Original data
            self.src_MT = None
            self.date0 = date0
            self.date1 = date1
            self.period = period
            self.zone_func = zone_func

            # Computed data
            self.LORSI = None
            self.count = None
            self.zones_arr = None
            self.t_arr = None
            self.src_MT_zones = None


    def copy(self):
        # Create a copy of the object
        res = LORSI_zone_class()
        # Original data
        res.src_MT = self.src_MT    # Stores a reference to the original df_MT: Hence: NOT a copy
        res.date0 = self.date0
        res.date1 = self.date1
        res.period = self.period
        res.zone_func = self.zone_func

        # Computed data
        res.LORSI = self.LORSI.copy()
        res.count = self.count.copy()
        res.zones_arr = self.zones_arr.copy()
        res.t_arr = self.t_arr.copy()
        res.src_MT_zones = self.src_MT_zones.copy()

        return res
    

    def get_dates(self):
        # Returns the dates corresponding to the time indices
        # Does not update anything internally
        return convert_t_to_date(self.t_arr, self.period, self.date0)
    

    def get_LORSI_df(self):
        # Returns the LORSI as a dataframe with dates as index and zones as columns
        return pd.DataFrame(self.LORSI, index=self.get_dates(), columns=self.zones_arr)


    def get_count_df(self):
        # Returns the counts as a dataframe with dates as index and zones as columns
        return pd.DataFrame(self.count, index=self.get_dates(), columns=self.zones_arr)
        

    def insert_missing_zones(self, new_zones_arr):
        # Augments the stored zones_arr with the new zones in new_zones_arr

        LORSI = pd.DataFrame(self.LORSI, columns=self.zones_arr)
        count = pd.DataFrame(self.count, columns=self.zones_arr)
        
        for zone in new_zones_arr:
            if zone not in LORSI.keys():
                LORSI[zone] = np.zeros(len(LORSI))
                count[zone] = np.zeros(len(count)).astype(int)

        res = self.copy()
        res.LORSI = LORSI[sorted(LORSI.columns)].values
        res.count = count[sorted(count.columns)].values
        res.zones_arr = new_zones_arr.values

        return res


    def convert_to_t_days(self, dates):
        # Converts the dates to time indices
        return convert_date_to_t(dates, 1, self.date0)


    def set_LORSI_to_zero_mean(self):
        # Shifts the LORSI so that every entry has zero mean.
        self.LORSI = self.LORSI - np.mean(self.LORSI, axis=0)


    def interpolate_to_dates(self, new_dates, kind="linear"):
        """
        Returns two dataframes, LORSI_interp and count_interp, with the LORSI and count interpolated to the given dates, with extrapolation.
        The given dates can be arbitrary, and does not have to be within the original time interval. 
        NOTE: The total count is preserved in the process, but estimated at the new dates.
        """

        index_days = self.convert_to_t_days(self.get_dates())
        new_index_days = self.convert_to_t_days(new_dates)

        LORSI_interp = pd.DataFrame(index=new_dates, columns=self.zones_arr)
        count_interp = pd.DataFrame(index=new_dates, columns=self.zones_arr)
        for (i, zone) in enumerate(self.zones_arr):
            LORSI_interp[zone] = interpolate.interp1d(index_days, self.LORSI[:, i], kind=kind, fill_value='extrapolate')(new_index_days)
            count_interp[zone] = interpolate.interp1d(index_days, self.count[:, i], kind=kind, fill_value='extrapolate')(new_index_days)
            # Normalize count so that the total equals the original total, if the total is non-zero
            zone_sum = np.sum(count_interp[zone])
            if zone_sum > 0:
                count_interp[zone] = count_interp[zone] * np.sum(self.count[:, i]) / zone_sum

        return LORSI_interp, count_interp


    def convert_to_period(self, new_period, kind="linear"):
        """
        Converts the LORSI to new period given. date0 is kept the same and used as ref_date if new_period is an integer.
        The information in count is interpolated to create approximate values for the new period. 
        NOTE: The total count is preserved in the process, but estimated at the new dates.
        """

        res = self.copy()

        if self.period != new_period:
            [new_t0, new_t1] = convert_date_to_t([res.date0, res.date1], new_period, res.date0)

            new_t_arr = np.arange(new_t0, new_t1)
            new_dates = convert_t_to_date(new_t_arr, new_period, res.date0)

            new_LORSI, new_count = res.interpolate_to_dates(new_dates, kind=kind)
            res.LORSI = new_LORSI.values
            res.count = new_count.values

            res.t_arr = new_t_arr
            res.period = new_period

        return res


    def filter_LORSI_in_time(self, weights=None, window_size=5):
        # Filters the LORSI values in time using the count as weights. Returns the result as a class of the same type.

        if weights is None:
            weights = self.count

        LORSI_w = np.zeros(self.LORSI.shape)
        for (i, _) in enumerate(self.zones_arr):
            LORSI_w[:, i] = smooth_w(self.LORSI[:, i], weights[:, i], window_size // 2)

        res = self.copy()
        res.LORSI = LORSI_w

        return res
    

    def filter_LORSI_in_space(self, zones_neighbors):
        # Do a spatial filtering: compute the weighted average of the LORSIs of the zone itself and its neighbors.
        LORSI = self.get_LORSI_df()
        count = self.get_count_df()

        LORSI_w = LORSI.copy()
        count_w = count.copy()

        central_zone_w = 5

        for zone in LORSI_w.columns:
            neighbors = zones_neighbors[zones_neighbors[zone] == 1].index

            neighbors_LORSI = LORSI[neighbors]
            neighbors_count = count[neighbors]
            
            weighted_sum = neighbors_LORSI.multiply(neighbors_count).sum(axis=1) + LORSI[zone] * central_zone_w * count[zone]
            count_sum = neighbors_count.sum(axis=1) + central_zone_w * count[zone]

            LORSI_w[zone] = weighted_sum / count_sum
            count_w[zone] = count_sum

            # Insert 0 for nan in case of division by 0
            LORSI_w[zone] = LORSI_w[zone].fillna(0)

        res = self.copy()
        res.LORSI = LORSI_w.values
        res.count = count_w.values

        return res


    def add_scatter(self, fig, desc="", row=1, col=1, zone=0, mode="lines"):
        # Add a scatter plot of the LORSI values
        df = self.get_LORSI_df()
        name = "Period: " + self.period + ", Zone: " + str(zone)
        if desc != "":
            name += ", " + desc
        return fig.add_trace(go.Scatter(x=df.index, y=df[zone], mode=mode, name=name), row=row, col=col)



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


zone_func = default_zone_func

# Create the LORSI classes
all_LORSI_w = LORSI_zone_class(df_MT, date0, date1, "weekly")
zone_LORSI_m = LORSI_zone_class(df_MT, date0, date1, "monthly", zone_func=zone_func_div100)

# Filter all of Oslo in time, with a 5 week window size
all_LORSI_w_f = all_LORSI_w.filter_LORSI_in_time(window_size=5)


# PLOTTING
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig = all_LORSI_w.add_scatter(fig, mode="markers")
fig = all_LORSI_w_f.add_scatter(fig, mode="markers")
fig.show()