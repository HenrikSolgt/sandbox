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
from solgt.timeseries.filter import smooth_w


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

df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[postcode] = df_MT[postcode].astype(int)


df = df_MT.copy()
# Only those with PROM >= 100
df = df[df[prom_code] >= 100].reset_index(drop=True)

RSI = get_RSI(df, date0, date1, period=period, interpolate=True)

MA_price = smooth_w(RSI["price"], RSI["count"], 5)

# Create a scatterplot with this RSI
fig = go.Figure()
fig = fig.add_trace(go.Scatter(x=RSI["date"], y=RSI["price"], mode="lines", name="RSI"))
fig = fig.add_trace(go.Scatter(x=RSI["date"], y=MA_price, mode="lines", name="RSI 5 Week MA"))
fig = fig.update_layout(title="RSI for all dwellings with PROM >= 100", xaxis_title="Time", yaxis_title="RSI")
fig.show()

# Store as html
fig.write_html("C:\Code\plots\RSI.html")