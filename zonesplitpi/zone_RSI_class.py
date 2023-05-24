# Standard Python packages
import datetime
import pandas as pd
import numpy as np

# Solgt packages
from solgt.priceindex.repeatsales import get_RSI, add_derived_MT_columns, get_repeated_idx, get_df_ttp_from_RS_idx, create_and_solve_OLS_problem
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
    def __init__(self, df_MT, date0, date1, period="monthly", zone_div=int(1e9)):
        """
        Create a zone_RSI_class object. This object contains the LORSI for each zone, and the count of sales for each zone. 
        The matched transactions are loaded from df_MT, and only data between date0 and date1 is used.
        The default value of zone_div is 1e9, resulting in a single zone: hence all transactions are treated as a single zone.
        """
        self.date0 = date0
        self.date1 = date1
        self.period = period

        [self.t0, self.t1] = convert_date_to_t([date0, date1], period)

        df_MT = add_derived_MT_columns(df_MT, period, date0)

        df_MT["zone"] = df_MT[gr_krets] // zone_div
        LORSI, count = get_LORSI_and_count_for_zones(df_MT, self.t0, self.t1) # Create OLS and count for the zones
        self.LORSI = LORSI.values
        self.count = count.values
        self.zones = LORSI.columns.values
        self.zone_div = zone_div 

        self.t = LORSI.index.values
        self.LORSI_orig = LORSI
        self.count_orig = count


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
        

    def update_with_missing_zones(self, zones_arr):
        # Augments the stored zones with the missing zones in zones_arr

        LORSI = pd.DataFrame(self.LORSI, columns=self.zones)
        count = pd.DataFrame(self.count, columns=self.zones)
        
        for zone in zones_arr:
            if zone not in LORSI.keys():
                LORSI[zone] = np.zeros(len(LORSI))
                count[zone] = np.zeros(len(count)).astype(int)

        # Sort columns
        self.LORSI = LORSI[sorted(LORSI.columns)].values
        self.count = count[sorted(count.columns)].values
        self.zones = zones_arr.values


    def set_LORSI_to_start_at_zero(self):
        # Shifts the LORSI so that it starts at zero for all entries.
        self.LORSI = self.LORSI - self.LORSI.iloc[0, :]


    def set_LORSI_to_zero_mean(self):
        # Shifts the LORSI so that every entry has zero mean.
        self.LORSI = self.LORSI - np.mean(self.LORSI, axis=0)


    def interpolate_to_dates(self, new_dates):
        """
        Interpolates the OLS to the given dates, with extrapolation.
        Does not update the anything internally, but simply returns the interpolated values.
        """
        index_days = convert_date_to_t(self.get_dates(), 1, self.date0)
        new_index_days = convert_date_to_t(new_dates, 1, self.date0)

        res = pd.DataFrame(index=new_dates, columns=self.zones)
        for (i, zone) in enumerate(self.zones):
            res[zone] = interpolate.interp1d(index_days, self.LORSI[:, i], fill_value='extrapolate')(new_index_days)

        return res


    def update_period(self, new_period):
        """
        Resamples the OLS to the new period rmat. A reference ref_date can be given if new_period is an integer.
        NOTE: Count is lost in this process.
        """

        if self.period != new_period:
            [new_t0, new_t1] = convert_date_to_t([self.date0, self.date1], new_period, self.date0)

            new_T_arr = np.arange(new_t0, new_t1)

            new_dates = convert_t_to_date(new_T_arr, new_period, self.date0)
            new_LORSI = self.interpolate_to_dates(new_dates)
            if self.zone_div is None:
                self.LORSI = new_LORSI["pred"].values
            else:
                self.LORSI = new_LORSI.values

            self.t = new_T_arr
            self.period = new_period


    def filter_LORSI_in_time(self, weights=None, window_size=5):
        # Filters the LORSI values in time using the count as weights. Does not set a value, but simply returns the filtered values.
        if weights is None:
            weights = self.count

        res = np.zeros(self.LORSI.shape)
        for (i, zone) in enumerate(self.zones):
            res[:, i] = smooth_w(self.LORSI[:, i], weights[:, i], window_size)

        return pd.DataFrame(res, index=self.get_dates(), columns=self.zones)


    def set_filter_LORSI_in_time(self, weights=None, window_size=5):
        # Sets the LORSI to the filtered values, using filter_LORSI_in_time to do so.
        self.LORSI = self.filter_LORSI_in_time(window_size=window_size).values


    def filter_LORSI_in_space(self, neighbors):
        # Do a spatial filtering: compute the weighted average of the LORSIs of the zone itself and its neighbors.
        LORSI = self.get_LORSI_df()
        count = self.get_count_df()

        res_LORSI = LORSI.copy()
        res_count = count.copy()
        res_LORSI[:] = np.NaN
        res_count[:] = np.NaN

        central_zone_w = 1

        for zone in res_LORSI.columns:
            neighbors = zones_neighbors[zones_neighbors[zone] == 1].index

            neighbors_LORSI = LORSI[neighbors]
            neighbors_count = count[neighbors]
            
            weighted_sum = neighbors_LORSI.multiply(neighbors_count).sum(axis=1) + LORSI[zone] * central_zone_w * count[zone]
            count_sum = neighbors_count.sum(axis=1) + central_zone_w * count[zone]

            res_LORSI[zone] = weighted_sum / count_sum
            res_count[zone] = count_sum.astype(int)

        return res_LORSI, res_count


    def set_filter_LORSI_in_space(self, neighbors):
        # Sets the LORSI to the filtered values, using filter_LORSI_in_space to do so.
        LORSI, count = self.filter_LORSI_in_space(neighbors)
        self.LORSI = LORSI.values
        self.count = count.values

"""
MAIN PROGRAM
"""


# Define time period
date0 = datetime.date(2014, 1, 6)
date1 = datetime.date(2022, 1, 1)

df_MT = load_MT_data()


# Fetch information about the zones
all_RSI_w = zone_RSI_class(df_MT, date0, date1, "weekly")
zone_RSI_q = zone_RSI_class(df_MT, date0, date1, "quarterly", zone_div=100)
zone_RSI_m = zone_RSI_class(df_MT, date0, date1, "monthly", zone_div=100)
zone_RSI_w = zone_RSI_class(df_MT, date0, date1, "weekly", zone_div=100)

# Augment with missing zones
zones_arr, zones_neighbors = get_zones_and_neighbors(zone_div=100)
zone_RSI_w.update_with_missing_zones(zones_arr)
zone_RSI_m.update_with_missing_zones(zones_arr)
zone_RSI_q.update_with_missing_zones(zones_arr)

all_RSI_w.set_LORSI_to_zero_mean()
all_w = all_RSI_w.get_LORSI_df()
all_w_c = all_RSI_w.get_count_df()

all_RSI_w.set_filter_LORSI_in_time(window_size=5)
all_w_f = all_RSI_w.get_LORSI_df()
all_w_c_f = all_RSI_w.get_count_df()




zone = 0
# Plot for selected zones
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig = fig.add_trace(go.Scatter(x=all_w.index, y=all_w[zone], mode="markers", name="Weekly"), row=1, col=1)

fig.show()


zone_RSI_m.set_LORSI_to_zero_mean()
zone_m = zone_RSI_m.get_LORSI_df()
zone_m_c = zone_RSI_m.get_count_df()

zone_RSI_m.set_filter_LORSI_in_space(zones_neighbors)
zone_RSI_m.set_LORSI_to_zero_mean()
zone_m_f = zone_RSI_m.get_LORSI_df()
zone_m_c_f = zone_RSI_m.get_count_df()

zone_RSI_m.set_filter_LORSI_in_time(window_size=3)
zone_RSI_m.set_LORSI_to_zero_mean()
zone_m_f2 = zone_RSI_m.get_LORSI_df()
zone_m_c_f2 = zone_RSI_m.get_count_df()


fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig = fig.add_trace(go.Scatter(x=all_w.index, y=all_w[0], mode="lines", name="Weekly: All"), row=1, col=1)
fig = fig.add_trace(go.Scatter(x=all_w_f.index, y=all_w_f[0], mode="lines", name="Weekly: All, filtered in time"), row=1, col=1)
zones_arr2 = [2, 11, 12, 42, 44]
for zone in zones_arr2:
    fig = fig.add_trace(go.Scatter(x=zone_m.index, y=zone_m[zone], mode="lines", name="Zone, Monthly: " + str(zone)), row=1, col=1)
    fig = fig.add_trace(go.Scatter(x=zone_m_f.index, y=zone_m_f[zone], mode="lines", name="Zone, Monthly, filtered in space: " + str(zone)), row=1, col=1)
    fig = fig.add_trace(go.Scatter(x=zone_m_f2.index, y=zone_m_f2[zone], mode="lines", name="Zone, Monthly, filtered in space and time: " + str(zone)), row=1, col=1)

fig.show()