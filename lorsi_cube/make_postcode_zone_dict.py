# Python packages
import datetime
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
import solgt.db.dss


# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
postcode = "postcode"
prom_code = "PROM"


def default_zone_func(df):
    # Default zone function. Returns a all-zero numpy array of same length as df
    return np.zeros(len(df)).astype(int)

def zone_func_div(df, zone_div):
    # Zone function that returns the values of df integer divided by zone_div
    return df // zone_div

def zone_func_div100(df):
    # Zone function that returns the first two digits of the values of df
    return zone_func_div(df=df, zone_div=100)


df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[postcode] = df_MT[postcode].astype(int)

zone_func = zone_func_div100

df = df_MT.copy()
df["zone"] = zone_func(df[gr_krets])
df = df[~df["zone"].isna()][["postcode", "zone"]].reset_index(drop=True)
df["zone"] = df["zone"].astype(int)

# Make a list of all unique postcodes and the corresponding most frequent zone, and its count
df_postcode_zone = df.groupby(postcode)["zone"].agg(lambda x: x.value_counts().index[0]).reset_index()

# Now upload to Mongo DB

prodDB = solgt.db.dss.get_prodDB()

collection = prodDB["postcode_zone"]
collection.drop()
collection.insert_many(df_postcode_zone[["postcode", "zone"]].T.to_dict().values())
