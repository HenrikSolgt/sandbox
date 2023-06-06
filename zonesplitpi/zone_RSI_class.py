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


# Create LORSI and count for all zones
def get_LORSI_and_count_for_zones(df, t0, t1):
    """
    Get LORSI and count for all matched transactions in DataFrame df, for the time period [t0, t1).
    All raw transactions from df are used. The filtering on time period [t0, t1) is performed after the transactions have been matched.
    Note that only the zones occuring in df are included in the output.
    Inputs:
        - df: DataFrame with columns "unitkey", "price_inc_debt", "zone", "t"
        - t0: Start time
        - t1: End time
    """

    zones_arr = df["zone"].unique()
    zones_arr.sort()

    T_arr = np.arange(t0, t1)
    # Create empty dataframes
    zone_LORSI = pd.DataFrame(index=T_arr, columns=zones_arr)
    zone_counts = pd.DataFrame(index=T_arr, columns=zones_arr)

    for zone_no in zones_arr:
        # Filter the MT data for the current zone
        df_zone = df[df["zone"] == zone_no].reset_index(drop=True)
        print("Zone number: " + str(zone_no) + ". Number of transactions: " + str(len(df_zone)))

        # Get repeated sales index and create df_ttp
        RS_idx = get_repeated_idx(df_zone)
        df_ttp = get_df_ttp_from_RS_idx(df_zone, RS_idx)

        if (len(df_ttp) > 0):
            LORSI_res = create_and_solve_LORSI_OLS_problem(df_ttp)
            LORSI_res = LORSI_res[(LORSI_res["t"] >= t0) & (LORSI_res["t"] < t1)].reset_index(drop=True)

            # Split LORSI_res into LORSI and count dataframes
            LORSI = LORSI_res[["t", "pred"]].set_index(["t"]).reindex(T_arr)
            count = LORSI_res[["t", "count"]].set_index(["t"]).reindex(T_arr, fill_value=0)
        else:
            LORSI = pd.DataFrame(index=T_arr, columns=["pred"])
            count = pd.DataFrame(index=T_arr, columns=["count"]).fillna(0)

        # Remove index names
        LORSI.index.name = None
        count.index.name = None

        zone_LORSI[zone_no] = LORSI
        zone_counts[zone_no] = count

    # Substitute NaN with 0
    zone_LORSI.fillna(0, inplace=True)
    zone_counts.fillna(0, inplace=True)

    # Convert to int
    zone_counts = zone_counts.astype(int)

    return zone_LORSI, zone_counts


def default_zone_func(df_MT):
    """
    Default zone function. Returns a all-zero numpy array of same length as df_MT.
    """
    return np.zeros(len(df_MT))


def zone_func_div100(df_MT):
    return df_MT[gr_krets] // 100


def fill_in_nan_zones(df, df_all):
    res = df.copy()

    all_idx_nan = df_all[df_all["zone"].isna()].index
    res_idx_nan = res[res["zone"].isna()].index

    # Net ok_entries be all non-nan entries in df and df_all
    ok_entries = pd.concat([df.drop(res_idx_nan), df_all.drop(all_idx_nan)])
    nan_entries = res.loc[res_idx_nan]

    # We will now use ok_entries to fill in for the nan_entries 
    res.loc[nan_entries.index, "zone"] = nan_entries["postcode"].map(ok_entries.groupby("postcode")["zone"].agg(lambda x: x.value_counts().index[0]))

    return res




class LORSI_zone_class:

    def __init__(self, df_MT=None, date0=None, date1=None, period="monthly", zone_func=default_zone_func, df_MT_all=None):
        """
        Create a zone_LORSI_class object. This object contains the LORSI for each zone, and the count of sales for each zone. 
        The matched transactions are loaded from df_MT, and only data between date0 and date1 is used.
        The default value of zone_func is a function returning all zeros, producing in a single zone: hence all transactions are treated to belong to the same zone.
        """
        if df_MT is not None:
            self.df_MT = df_MT   # Stores a reference to the original df_MT, which will be used for Zone identification later
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


    def copy(self):
        # Create a copy of the object
        res = LORSI_zone_class()
        res.df_MT = self.df_MT    # Stores a reference to the original df_MT: Hence: NOT a copy
        res.LORSI = self.LORSI.copy()
        res.count = self.count.copy()
        res.zones = self.zones.copy()
        res.t = self.t.copy()
        res.zone_func = self.zone_func
        res.date0 = self.date0
        res.date1 = self.date1
        res.period = self.period
        res.t0 = self.t0
        res.t1 = self.t1
        return res


    def get_dates(self):
        # Returns the dates corresponding to the time indices
        # Does not update anything internally
        return convert_t_to_date(self.t, self.period, self.date0)
    

    def get_LORSI_df(self):
        # Returns the LORSI as a dataframe with dates as index and zones as columns
        return pd.DataFrame(self.LORSI, index=self.get_dates(), columns=self.zones)


    def get_count_df(self):
        # Returns the counts as a dataframe with dates as index and zones as columns
        return pd.DataFrame(self.count, index=self.get_dates(), columns=self.zones)


    def insert_missing_zones(self, zones_arr):
        # Augments the stored zones with the missing zones in zones_arr

        LORSI = pd.DataFrame(self.LORSI, columns=self.zones)
        count = pd.DataFrame(self.count, columns=self.zones)
        
        for zone in zones_arr:
            if zone not in LORSI.keys():
                LORSI[zone] = np.zeros(len(LORSI))
                count[zone] = np.zeros(len(count)).astype(int)

        res = self.copy()
        res.LORSI = LORSI[sorted(LORSI.columns)].values
        res.count = count[sorted(count.columns)].values
        res.zones = zones_arr.values

        return res


    def convert_to_t_days(self, dates):
        # Converts the dates to time indices
        return convert_date_to_t(dates, 1, self.date0)


    def set_LORSI_to_start_at_zero(self):
        # Shifts the LORSI so that it starts at zero for all entries.
        self.LORSI = self.LORSI - self.LORSI.iloc[0, :]


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

        LORSI_interp = pd.DataFrame(index=new_dates, columns=self.zones)
        count_interp = pd.DataFrame(index=new_dates, columns=self.zones)
        for (i, zone) in enumerate(self.zones):
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

            new_T_arr = np.arange(new_t0, new_t1)
            new_dates = convert_t_to_date(new_T_arr, new_period, res.date0)

            new_LORSI, new_count = res.interpolate_to_dates(new_dates, kind=kind)
            res.LORSI = new_LORSI.values
            res.count = new_count.values

            res.t = new_T_arr
            res.period = new_period

        return res


    def filter_LORSI_in_time(self, weights=None, window_size=5):
        # Filters the LORSI values in time using the count as weights. Returns the result as a class of the same type.

        if weights is None:
            weights = self.count

        LORSI_w = np.zeros(self.LORSI.shape)
        for (i, zone) in enumerate(self.zones):
            LORSI_w[:, i] = smooth_w(self.LORSI[:, i], weights[:, i], window_size // 2)

        res = self.copy()
        res.LORSI = LORSI_w

        return res


    def filter_LORSI_in_space(self, neighbors):
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
            count_w[zone] = count_sum.astype(int)

            # Insert 0 for nan in case of division by 0
            LORSI_w[zone] = LORSI_w[zone].fillna(0)

        res = self.copy()
        res.LORSI = LORSI_w.values
        res.count = count_w.values

        return res



    def score_LORSI(self, df_MT_test):
        """
        Compute the LORSI score for the LORSI predictions created by instance self applied to df_MT_test.
        df_MT_test contains the matched transactions, with a time indicator "t". This one is filtered on the period [t0, t1).
        Inputs:
        Computes the score of the LORSI on the test set df_MT_test.
        - df_MT_test: Dataframe with matched transactions. Must contain the following columns: "unitkey", "sold_date", "price_inc_debt"
        """

        df = df_MT_test.copy()
        df_all = self.df_MT.copy()

        # Fill in the zone column using the zone_func
        df["zone"] = self.zone_func(df)
        df_all["zone"]= self.zone_func(df_all)
        df = fill_in_nan_zones(df, df_all)
        # TODO: Fix this: dropna should not be necessary
        df = df.dropna(subset=["zone"]).reset_index(drop=True)

        # Add the derived columns to the dataframe
        df = add_derived_MT_columns(df, self.period, self.date0)

        # Get the index of the repeated sales
        RS_idx_test = get_repeated_idx(df)
        
        df_ddp = pd.DataFrame()
        line0, line1 = get_RS_idx_lines(df, RS_idx_test, [date_col, "y", "zone"])
        df_ddp["sold_date0"] = line0[date_col]
        df_ddp["sold_date1"] = line1[date_col]
        df_ddp["dp"] = line1["y"] - line0["y"]
        df_ddp["zone"] = line0["zone"]

        # Filter away all transactions that are not in the time period covered by the class instance
        df_ddp = df_ddp[(df_ddp["sold_date0"] >= self.date0) & (df_ddp["sold_date1"] < self.date1)].reset_index(drop=True)

        df_ddp["t0"] = self.convert_to_t_days(df_ddp["sold_date0"])
        df_ddp["t1"] = self.convert_to_t_days(df_ddp["sold_date1"])

        # Needs to interpolate the LORSI: May be sampled at random dates
        LORSI = self.get_LORSI_df()

        # Need to create an interpolation function for each zone
        LORSI_index = self.convert_to_t_days(LORSI.index)
        for zone in LORSI.columns:
            f = interpolate.interp1d(LORSI_index, LORSI[zone], fill_value="extrapolate")

            df_ddp_zone = df_ddp[df_ddp["zone"] == zone]
            pred0 = f(df_ddp_zone["t0"])
            pred1 = f(df_ddp_zone["t1"])

            df_ddp.loc[df_ddp_zone.index, "pred0"] = pred0
            df_ddp.loc[df_ddp_zone.index, "pred1"] = pred1
            df_ddp.loc[df_ddp_zone.index, "dp_est"] = pred1 - pred0

        df_ddp["dp_e"] = df_ddp["dp"] - df_ddp["dp_est"]

        return df_ddp


    def add_HPF_part_from_LORSI(self, other, window_size=5):
        """
        Does a count-weighted LPF-filtering of self, and adds the HPF part of the LORSI instance other to it.
        other does not need to be sampled at the same dates as self, as a linear interpolation is done.
        Inputs:
            - other: LORSI_class instance. Only the first zone is used. The HPF part of this zone is added to every LORSI zone of self.
            - window_size: int. Size of the window for the LPF filtering.

        NOTE: The information in count does not change, only the LORSI values.
        """
        
        # Convert other to the same period as self
        other = other.convert_to_period(self.period)
        
        # Extract the LORSI and count dataframes
        x = other.get_LORSI_df()
        y = self.get_LORSI_df()
        y_c = self.get_count_df()

        # Create LPF as a dataframe of size y
        LPF_y = pd.DataFrame(index=y.index, columns=y.columns)
        HPF_x = pd.DataFrame(index=y.index, columns=y.columns)

        w_b = window_size // 2
        for zone in y.columns:
            weights = y_c[zone]
            LPF_y[zone] = smooth_w(y[zone], weights, w_b)
            HPF_x[zone] = x[0] - smooth_w(x[0], weights, w_b)

        res = self.copy()
        res.LORSI = LPF_y + HPF_x

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


df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[gr_krets] = df_MT[gr_krets]
df_MT[postcode] = df_MT[postcode].astype(int)

# Split df_MT into train and test sets, but only the repeated sales
df_MT = add_derived_MT_columns(df_MT)
RS_idx = get_repeated_idx(df_MT)
RS_idx_train, RS_idx_test = train_test_split(RS_idx, test_size=0.2, random_state=42)

# Let train_I be I0 and I1 in RS_idx_train
I_train = pd.concat([RS_idx_train["I0"], RS_idx_train["I1"]]).drop_duplicates().sort_values().reset_index(drop=True)
I_test = pd.concat([RS_idx_test["I0"], RS_idx_test["I1"]]).drop_duplicates().sort_values().reset_index(drop=True)

# Extract indices
df_MT_train = df_MT.loc[I_train].reset_index(drop=True)
df_MT_test = df_MT.loc[I_test].reset_index(drop=True)

# Create the LORSI class instances for Oslo weekly, and zones monthly
# These are the raw datas, without any filtering
all_LORSI_w = LORSI_zone_class(df_MT_train, date0, date1, "weekly", df_MT_all=df_MT)
zone_LORSI_m = LORSI_zone_class(df_MT_train, date0, date1, "monthly", zone_func=zone_func_div100, df_MT_all=df_MT)

# Augment the zone RSI with_ with missing zones
zones_arr, zones_neighbors = get_zones_and_neighbors(zone_div=100)
zone_LORSI_m = zone_LORSI_m.insert_missing_zones(zones_arr)

# Filter zone in space
zone_LORSI_m_s = zone_LORSI_m.filter_LORSI_in_space(zones_neighbors)
zone_LORSI_m_s2 = zone_LORSI_m_s.filter_LORSI_in_space(zones_neighbors)
zone_LORSI_m_s3 = zone_LORSI_m_s2.filter_LORSI_in_space(zones_neighbors)

# Resample zone LORSI to weekly
zone_LORSI_w_s = zone_LORSI_m_s3.convert_to_period("weekly", kind="linear")

# Filter all of Oslo in time, with a 5 week window size
all_LORSI_w_f = all_LORSI_w.filter_LORSI_in_time(window_size=5)

# Combine x = all_LORSI_w_f and y = zone_LORSI_w_s into one dataframe, with the LPF component of y and HPF component of x
zone_comb_LORSI = zone_LORSI_w_s.add_HPF_part_from_LORSI(all_LORSI_w_f, window_size=13)  # 6 weeks is used, as (2*6+1) weeks is approximately 3 months

# Shift to zero mean
all_LORSI_w.set_LORSI_to_zero_mean()
all_LORSI_w_f.set_LORSI_to_zero_mean()
zone_LORSI_m.set_LORSI_to_zero_mean()
zone_LORSI_m_s.set_LORSI_to_zero_mean()
zone_LORSI_m_s2.set_LORSI_to_zero_mean()
zone_LORSI_m_s3.set_LORSI_to_zero_mean()
zone_LORSI_w_s.set_LORSI_to_zero_mean()
zone_comb_LORSI.set_LORSI_to_zero_mean()


all_LORSI_w.score_LORSI(df_MT_test, df_MT)["dp_e"].abs().mean()
all_LORSI_w_f.score_LORSI(df_MT_test, df_MT)["dp_e"].abs().mean()
zone_LORSI_m.score_LORSI(df_MT_test, df_MT)["dp_e"].abs().mean()
zone_LORSI_m_s.score_LORSI(df_MT_test, df_MT)["dp_e"].abs().mean()
zone_LORSI_m_s2.score_LORSI(df_MT_test, df_MT)["dp_e"].abs().mean()
zone_LORSI_m_s3.score_LORSI(df_MT_test, df_MT)["dp_e"].abs().mean()
zone_LORSI_w_s.score_LORSI(df_MT_test, df_MT)["dp_e"].abs().mean()
zone_comb_LORSI.score_LORSI(df_MT_test, df_MT)["dp_e"].abs().mean()


"""
LORSI
"""
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig = all_LORSI_w.add_scatter(fig, mode="markers")
fig = all_LORSI_w_f.add_scatter(fig, desc="Filtered in time", mode="markers")

zones_arr2 = [2, 11, 12, 42, 44]
for zone in zones_arr2:
    # fig = zone_LORSI_m.add_scatter(fig, zone=zone, desc="Raw monthly zone data")
    fig = zone_LORSI_m_s.add_scatter(fig, zone=zone, desc="Spatially filtered")
    # fig = zone_LORSI_m_s2.add_scatter(fig, zone=zone, desc="Spatially filtered x2")
    fig = zone_LORSI_w_s.add_scatter(fig, zone=zone, desc="Resampled from spatally filtered x3")
    fig = zone_comb_LORSI.add_scatter(fig, zone=zone, desc="LFP + HPF from all_LORSI_w_f")
    
fig.show()




import sys
def get_total_size(obj):
    size = sys.getsizeof(obj)
    
    if hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            size += get_total_size(attr_value)
    
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum(get_total_size(item) for item in obj)
    
    return size


get_total_size(all_LORSI_w) / 1e6
get_total_size(all_LORSI_w_f)