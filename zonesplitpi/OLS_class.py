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

gr_krets = "grunnkrets_id"
# period = "quarterly"
period = "monthly"



def get_zones_and_neighbors(zone_div=100):
    """
    Get the zones and their neighbors.
    """
    zones_geometry = get_zone_geometry(zone_div)
    zones_arr = zones_geometry["zone"]
    zones_neighbors = get_zone_neighbors(zones_geometry)

    return zones_arr, zones_neighbors



class OLS_class:
    def __init__(self, df_MT, date0, date1, period="monthly", zone_div=None):
        self.date0 = date0
        self.date1 = date1
        self.period = period
        self.zone_div = zone_div # Is None if not split into zones

        [self.t0, self.t1] = convert_date_to_t([date0, date1], period)

        df_MT = add_derived_MT_columns(df_MT, period, date0)

        if zone_div is None: # All zones as one
            OLS, count = get_OLS_and_count(df_MT, self.t0, self.t1)
            self.OLS = OLS["pred"].values
            self.count = count["count"].values
        else:
            df_MT["zone"] = df_MT[gr_krets] // zone_div
            OLS, count = get_zone_OLS_and_count(df_MT, self.t0, self.t1) # Create OLS and count for the zones
            self.OLS = OLS.values
            self.count = count.values
            self.zones = OLS.columns.values

        self.t = OLS.index.values
        self.OLS_orig = OLS
        self.count_orig = count


    def get_dates(self):
        return convert_t_to_date(self.t, self.period, self.date0)
    

    def fill_in_missing_zones(self, zones_arr):
        # Augments the stored zones with the missing zones. Must only be used if the zones are split

        OLS = pd.DataFrame(self.OLS, columns=self.zones)
        count = pd.DataFrame(self.count, columns=self.zones)
        
        for zone in zones_arr:
            if zone not in OLS.keys():
                OLS[zone] = np.zeros(len(OLS))
                count[zone] = np.zeros(len(count)).astype(int)

        # Sort columns in OLS_z_m and OLS_z_count_m
        self.OLS = OLS[sorted(OLS.columns)]
        self.count = count[sorted(count.columns)]
        self.zones = zones_arr


    def resample_to_period(new_period, ref_date=None):
        """
        Resamples the OLS to the new period format. A reference ref_date can be given if new_period is an integer.
        """
        a = 0

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

OLS_a = OLS_class(df_MT, date0, date1, period, zone_div=100)

# Fetch information about the zones
zones_arr, zones_neighbors = get_zones_and_neighbors(zone_div=100)


OLS_a.fill_in_missing_zones(zones_arr)


OLS_a.t
OLS_a.OLS


