# Python packages
import datetime
import time
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df, update_MT_parquet_file
from solgt.priceindex.repeatsales import get_RSI
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import conv_smoother


# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
postcode = "postcode"
prom_code = "PROM"


# Define time period
date0 = datetime.date(2012, 1, 1)
date1 = datetime.date(2023, 1, 1)

period = "weekly"

df_MT = get_parquet_as_df("C:\Code\py\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[postcode] = df_MT[postcode].astype(int)


df = df_MT.copy()


df0 = df[df["PROM"] < 60].reset_index(drop=True)
df1 = df[(df["PROM"] >= 60) & (df["PROM"] < 90)].reset_index(drop=True)
df2 = df[df["PROM"] >= 90].reset_index(drop=True)


RSI0 = get_RSI(df0, date0, date1, period=period, interpolate=True)
RSI1 = get_RSI(df1, date0, date1, period=period, interpolate=True)
RSI2 = get_RSI(df2, date0, date1, period=period, interpolate=True)


comb = pd.DataFrame()
comb["date"] = RSI0["date"]
comb["RSI60"] = RSI0["price"]
comb["RSI60-90"] = RSI1["price"]
comb["RSI90"] = RSI2["price"]

RSI0.to_csv("C:\Code\py\data\RSI60.csv")
RSI1.to_csv("C:\Code\py\data\RSI6090.csv")
RSI2.to_csv("C:\Code\py\data\RSI90.csv")
comb.to_csv("C:\Code\py\data\RSI_comb.csv")