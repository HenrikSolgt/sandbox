# Standard Python packages
import datetime
import pandas as pd
import numpy as np

# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df
from solgt.priceindex.repeatsales import get_RSI, get_RS_idx, get_df_ttp_from_RS_idx, create_and_solve_RSI_OLS_problem
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date

from plotly.subplots import make_subplots
import plotly.graph_objects as go

from zone_analysis import get_zone_geometry, get_zone_neighbors, get_zone_controid_distances

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

# Constants
price_col = "price_inc_debt"
date_col = "sold_date"
key_col = "unitkey"
period = "monthly"
gr_krets = "grunnkrets_id"


# Load raw data
df_raw = get_parquet_as_df("C:\Code\data\MT.parquet")

# Copy and preprocess
df = df_raw.copy()
    # Remove entries without a valid grunnkrets
df = df[~df[gr_krets].isna()].reset_index(drop=True)
    # Typecast to required types
df[date_col] = df[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df[gr_krets] = df[gr_krets].astype(int)

# Create zones
df["zone"] = df[gr_krets] // 100

zones = pd.Series(df["zone"].unique()).sort_values().reset_index(drop=True)

# Loop all zones and create RSI for them all
t0 = datetime.date(2018, 1, 1)
t1 = datetime.date(2023, 1, 1)

rsi_all = get_RSI(df, t0, t1, period)

rsi_list = list()

rs_count = [None] * len(zones)

for (i, zone_no) in enumerate(zones):
    df_zone = df[df["zone"] == zone_no].reset_index(drop=True)
    print("Zone number: " + str(zone_no) + ". Størrelse: " + str(len(df_zone)))

    # What follows are segments from get_RSI:
    # Insert a hard limit here, since data before 2010 is not too sparse
    df_zone = df_zone[df_zone[date_col] >= datetime.date(2010, 1, 1)]
    df_zone.reset_index(drop=True, inplace=True)

    try:
        rsi = get_RSI(df_zone, t0, t1, period)
    except Exception as e:
        rsi = pd.DataFrame(columns=["count", "date", "price"])
        print("Exception: " + str(e))

    rs_count[i] = rsi["count"].sum()
    print(rs_count[i])
    
    rsi_list.append(rsi)


d = rsi_list[-1]
plt.plot(d["date"], d["price"])
plt.show()

# Plot it all:
fig1 = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)
fig1.append_trace(
    go.Scatter(x=rsi_all["date"], y=rsi_all["price"], name="Price, all"),
    row=1,
    col=1,
)


for (i, rsi) in enumerate(rsi_list):
    print(i)

    # Plot
    fig1.append_trace(
        go.Scatter(x=rsi["date"], y=rsi["price"], name="Price, " + str(i)),
        row=1,
        col=1,
    )

    
fig1.show()


rsi_list2 = rsi_list.copy()

zones2 = get_zone_geometry()
zone2_neighbors = get_zone_neighbors(zones2)