# Python packages
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import interpolate
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df, update_MT_parquet_file
from solgt.priceindex.repeatsales import get_derived_MT_columns, add_derived_MT_columns, get_repeated_idx, get_RS_idx_lines, get_df_ttp_from_RS_idx, create_and_solve_LORSI_OLS_problem
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import smooth_w

# Local packages
from grk_zone_geometry import get_grk_zones_and_neighbors


def zone_func_div(df, zone_div):
    # Zone function that returns the values of df integer divided by zone_div
    return df // zone_div

def zone_func_div100(df):
    # Zone function that returns the first two digits of the values of df
    return zone_func_div(df=df, zone_div=100)


zones_neighbors = get_grk_zones_and_neighbors(zone_func_div100)


# Here, zones_neighbors is a matrix with 1 if two zones are neighbors, 0 otherwise. It is also 0 on the diagonal. 
# Create a function that computes the distance between two zones, using zones_neighbors only



from scipy.sparse.csgraph import floyd_warshall

distances = floyd_warshall(zones_neighbors)

