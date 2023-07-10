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

period = "monthly"

df_MT = get_parquet_as_df("C:\Code\py\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[postcode] = df_MT[postcode].astype(int)


df = df_MT.copy()

df["housingtype"].unique()

housing_type = "Rekkehus"

housing_types = ["Leilighet", "Enebolig", "Rekkehus"]
fig = go.Figure()
for housing_type in housing_types:
    df_B = df[df["housingtype"] == housing_type].reset_index(drop=True)
    RSI_B = get_RSI(df_B, date0, date1, period=period, interpolate=True)
    RSI_B_MA, _ = conv_smoother(RSI_B["price"], RSI_B["count"], w_L=3, window_type="flat")

    # Create a scatterplot with this RSI
    fig = fig.add_trace(go.Scatter(x=RSI_B["date"], y=RSI_B["price"], mode="lines", name="RSI, type " + housing_type))
    fig = fig.add_trace(go.Scatter(x=RSI_B["date"], y=RSI_B_MA, mode="lines", name="RSI, type " + housing_type + ", smoothed"))


fig = fig.update_layout(title="RSI for different dwelling types", xaxis_title="Time", yaxis_title="RSI")
fig.show()
# Store as html
fig.write_html("C:\Code\plots\RSI_dwelling_types.html")