# Standard Python packages
import datetime
import pandas as pd
import numpy as np

# Solgt packages
from solgt.priceindex.repeatsales import get_RSI, add_derived_MT_columns, get_repeated_idx, get_RS_idx_lines, get_df_ttp_from_RS_idx, create_and_solve_OLS_problem
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import smooth_w
from solgt.db.MT_parquet import get_parquet_as_df, update_MT_parquet_file

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt
from scipy import interpolate

from zone_analysis import get_zones_and_neighbors
from compute_LORSI_for_zones import load_MT_data, score_RSI_split, get_LORSI_and_count, get_LORSI_and_count_for_zones, compute_zone_LORSI_weighted

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

zone_div = 100

gr_krets = "grunnkrets_id"
# period = "quarterly"
period = "monthly"

key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"


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




class zone_RSI_class:
    def __init__(self, df_MT=None, date0=None, date1=None, period="monthly", zone_div=int(1e9)):
        """
        Create a zone_RSI_class object. This object contains the LORSI for each zone, and the count of sales for each zone. 
        The matched transactions are loaded from df_MT, and only data between date0 and date1 is used.
        The default value of zone_div is 1e9, resulting in a single zone: hence all transactions are treated as a single zone.
        """
        if df_MT is not None:
            self.date0 = date0
            self.date1 = date1
            self.period = period
            self.zone_div = zone_div 

            df_MT = add_derived_MT_columns(df_MT, period, date0)
            df_MT["zone"] = df_MT[gr_krets] // zone_div
            
            [self.t0, self.t1] = convert_date_to_t([date0, date1], period)
            LORSI, count = get_LORSI_and_count_for_zones(df_MT, self.t0, self.t1) # Create OLS and count for the zones
            self.LORSI = LORSI.values
            self.count = count.values
            self.zones = LORSI.columns.values
            self.t = LORSI.index.values
        else:
            self.LORSI = None
            self.count = None
            self.zones = None
            self.zone_div = None
            self.t = None


    def copy(self):
        # Create a copy of the object
        res = zone_RSI_class()
        res.LORSI = self.LORSI.copy()
        res.count = self.count.copy()
        res.zones = self.zones.copy()
        res.t = self.t.copy()
        res.zone_div = self.zone_div
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

        RS_idx_test = get_repeated_idx(df_MT_test)
        df_MT_test = add_derived_MT_columns(df_MT_test)
        df_MT_test["zone"] = df_MT_test[gr_krets] // self.zone_div

        df_ddp = pd.DataFrame()
        line0, line1 = get_RS_idx_lines(df_MT_test, RS_idx_test, [date_col, "y", "zone"])
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







"""
MAIN PROGRAM
"""

# Define time period
date0 = datetime.date(2014, 1, 6)
date1 = datetime.date(2022, 1, 1)

df_MT = load_MT_data()

# Create the LORSI class instances for Oslo weekly, and zones monthly
all_RSI_w = zone_RSI_class(df_MT, date0, date1, "weekly")
zone_RSI_m = zone_RSI_class(df_MT, date0, date1, "monthly", zone_div=100)

# Augment with missing zones
zones_arr, zones_neighbors = get_zones_and_neighbors(zone_div=100)
zone_RSI_m = zone_RSI_m.insert_missing_zones(zones_arr)

# Create all filtered versions
all_RSI_w_f = all_RSI_w.filter_LORSI_in_time(window_size=5)
zone_RSI_m_f = zone_RSI_m.filter_LORSI_in_space(zones_neighbors)
zone_RSI_m_f2 = zone_RSI_m_f.filter_LORSI_in_time(window_size=5)

# Interpolated to weekly
zone_RSI_w = zone_RSI_m.convert_to_period("weekly", kind="linear")
zone_RSI_w_f2 = zone_RSI_m_f2.convert_to_period("weekly", kind="linear")


# Set all to zero mean for plotting
all_RSI_w.set_LORSI_to_zero_mean()
all_RSI_w_f.set_LORSI_to_zero_mean()
zone_RSI_m.set_LORSI_to_zero_mean()
zone_RSI_m_f.set_LORSI_to_zero_mean()
zone_RSI_m_f2.set_LORSI_to_zero_mean()
zone_RSI_w.set_LORSI_to_zero_mean()
zone_RSI_w_f2.set_LORSI_to_zero_mean()

# Extract data as dataframes
all_w = all_RSI_w.get_LORSI_df()
all_w_f = all_RSI_w_f.get_LORSI_df()
zone_m = zone_RSI_m.get_LORSI_df()
zone_m_f = zone_RSI_m_f.get_LORSI_df()
zone_m_f2 = zone_RSI_m_f2.get_LORSI_df()
zone_w = zone_RSI_w.get_LORSI_df()
zone_w_f2 = zone_RSI_w_f2.get_LORSI_df()


"""
LORSI
"""
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig = fig.add_trace(go.Scatter(x=all_w.index, y=all_w[0], mode="lines", name="Weekly: All"), row=1, col=1)
fig = fig.add_trace(go.Scatter(x=all_w_f.index, y=all_w_f[0], mode="lines", name="Weekly: All, filtered in time"), row=1, col=1)
zones_arr2 = [2, 11, 12, 42, 44]
for zone in zones_arr2:
    fig = fig.add_trace(go.Scatter(x=zone_m_f2.index, y=zone_m_f2[zone], mode="lines", name="Zone, Monthly F2: " + str(zone)), row=1, col=1)
    fig = fig.add_trace(go.Scatter(x=zone_w_f2.index, y=zone_w_f2[zone], mode="lines", name="Zone, Weekly F2: " + str(zone)), row=1, col=1)
    # fig = fig.add_trace(go.Scatter(x=zone_m_f.index, y=zone_m_f[zone], mode="lines", name="Zone, Monthly, filtered in space: " + str(zone)), row=1, col=1)
    # fig = fig.add_trace(go.Scatter(x=zone_m_f2.index, y=zone_m_f2[zone], mode="lines", name="Zone, Monthly, filtered in space and time: " + str(zone)), row=1, col=1)

fig.show()




"""
Combining the LORSI class instances with LPF and HPF
"""
# Make copies of the LORSI class instances, complete with filtering
RSI_a_w = all_RSI_w_f.copy()  # Filtered in time
RSI_z_m = zone_RSI_m_f.copy()  # Filtered in space

# We will now filter RSI_z_m in time, and do a reverse filtering on RSI_a_w

LORSI_z = RSI_z_m.get_LORSI_df()
count_z = RSI_z_m.get_count_df()

for zone in 




