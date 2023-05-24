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
    def __init__(self, df_MT, date0, date1, period="monthly", zone_div=None):
        self.date0 = date0
        self.date1 = date1
        self.period = period

        [self.t0, self.t1] = convert_date_to_t([date0, date1], period)

        df_MT = add_derived_MT_columns(df_MT, period, date0)

        if zone_div is None: # All zones as one
            LORSI, count = get_LORSI_and_count(df_MT, self.t0, self.t1)
            self.LORSI = LORSI["pred"].values
            self.count = count["count"].values
            self.zone_div = None
        else:
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
        if self.zone_div is None: # All zones as one
            return pd.DataFrame(self.LORSI, index=self.get_dates(), columns=["pred"])
        else:
            return pd.DataFrame(self.LORSI, index=self.get_dates(), columns=self.zones)


    def get_LORSI_count_df(self):
        if zone_div is None: # All zones as one
            return pd.DataFrame(self.count, index=self.t, columns=["count"])
        else:
            return pd.DataFrame(self.count, index=self.t, columns=self.zones)
        

    def update_with_missing_zones(self, zones_arr):
        # Augments the stored zones with the missing zones. Must only be used if the zones are split

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
        """
        Shifts the LORSI so that it starts at zero for all entries.
        """
        if self.zone_div is None:
            self.LORSI = self.LORSI - self.LORSI[0]
        else:
            self.LORSI = self.LORSI - self.LORSI.iloc[0, :]


    def interpolate_to_dates(self, new_dates):
        """
        Interpolates the OLS to the given dates, with extrapolation.
        Does not update the anything internally, but simply returns the interpolated values.
        """
        index_days = convert_date_to_t(self.get_dates(), 1, self.date0)
        new_index_days = convert_date_to_t(new_dates, 1, self.date0)

        if self.zone_div is None:
            return interpolate.interp1d(index_days, self.LORSI, fill_value='extrapolate')(new_index_days)
        else:
            res = pd.DataFrame(index=new_dates, columns=self.zones)
            for (i, zone) in enumerate(self.zones):
                res[zone] = interpolate.interp1d(index_days, self.LORSI[:, i], fill_value='extrapolate')(new_index_days)

            return res


    def update_period(self, new_period):
        """
        Resamples the OLS to the new period format. A reference ref_date can be given if new_period is an integer.
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


    def filter_count_weighted_in_time(self):
        """
        Filters the OLS values in time using the count as weights.
        """
        a = 0


"""
MAIN PROGRAM
"""


# Define time period
date0 = datetime.date(2014, 1, 1)
date1 = datetime.date(2022, 1, 1)

df_MT = load_MT_data()

# Fetch information about the zones
all_RSI_m = zone_RSI_class(df_MT, date0, date1, "monthly", zone_div=None)
zone_RSI_q = zone_RSI_class(df_MT, date0, date1, "quarterly", zone_div=100)
zone_RSI_m = zone_RSI_class(df_MT, date0, date1, "monthly", zone_div=100)

# Augment with missing zones
zones_arr, zones_neighbors = get_zones_and_neighbors(zone_div=100)
zone_RSI_q.update_with_missing_zones(zones_arr)
zone_RSI_m.update_with_missing_zones(zones_arr)


zone = 11
# Plot for selected zones
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig = fig.add_trace(go.Scatter(x=ROSLI_q.index, y=ROSLI_q[zone], mode="lines", name="Quarterly: " + str(zone)), row=1, col=1)
fig = fig.add_trace(go.Scatter(x=ROSLI_m.index, y=ROSLI_m[zone], mode="markers", name="Monthly: " + str(zone)), row=1, col=1)
fig = fig.add_trace(go.Scatter(x=ROSLI_q2.index, y=ROSLI_q2[zone], mode="markers", name="Monthly: " + str(zone)), row=1, col=1)

fig.show()


