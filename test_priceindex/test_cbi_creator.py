"""
A script to create a CBI price index from.

Extract CBI-scripts from dataprocessing.dataprocessing

A CBI is a combined price index, created by combining 
"""

# Standard packages
import datetime

# Installed packages
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df

# Local packages
from compute_CBI import get_CBI

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
postcode = "postcode"
prom_code = "PROM"



t_switch1_diff = 28  # Four weeks
t_switch0_diff = 28 * 2  # Eight weeks
AD_diff_days = 365  # Only use data from last year for the HMI



"""
MAIN PROGRAM
"""


period = "monthly"

df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[postcode] = df_MT[postcode].astype(int)

# Convert sold_date to datetime.date
df_MT["sold_date"] = df_MT["sold_date"].apply(lambda x: datetime.date(x.year, x.month, x.day))  



CBI = get_CBI(df_MT)


# Plot CBI using graph object
fig = go.Figure()
fig = fig.add_trace(go.Scatter(x=CBI["date"], y=CBI["price"], mode="lines", name="HMI1"))
# fig = fig.add_trace(go.Scatter(x=CBI["date"], y=CBI["HMI1_weight"], mode="lines", name="HMI1 W"))
# fig = fig.add_trace(go.Scatter(x=CBI["date"], y=CBI["HMI2_weight"], mode="lines", name="HMI2 W"))
fig.show()

