# Standard Python packages
import datetime
import pandas as pd
import numpy as np

# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df
from solgt.priceindex.repeatsales import get_RSI, convert_MT_data_to_ttp, create_and_solve_OLS_problem
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import smooth_w

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import matplotlib.pyplot as plt


from zone_analysis import get_zone_geometry, get_zone_neighbors, get_zone_controid_distances

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

# Constants
price_col = "price_inc_debt"
date_col = "sold_date"
key_col = "unitkey"
period = "monthly"
gr_krets = "grunnkrets_id"
unitkey_col = "unitkey"


# Load raw data
df_raw = get_parquet_as_df("C:\Code\data\MT.parquet")

# Copy and preprocess
df_a = df_raw.copy()
    # Remove entries without a valid grunnkrets
df_a = df_a[~df_a[gr_krets].isna()].reset_index(drop=True)
    # Typecast to required types
df_a[date_col] = df_a[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_a[gr_krets] = df_a[gr_krets].astype(int)

df_a["t"] = convert_date_to_t(df_a[date_col], period)
df_a["y"] = np.log(df_a[price_col]) # The measured value
df_a["id"] = df_a[unitkey_col] # The dwelling identifier


# Create zones
df_a["zone"] = df_a[gr_krets] // 1000

zones = pd.Series(df_a["zone"].unique()).sort_values().reset_index(drop=True)

# Loop all zones and create RSI for them all
date0 = datetime.date(2018, 1, 1)
date1 = datetime.date(2023, 1, 1)

rsi_all = get_RSI(df_a, date0, date1, period)
rsi_all["pred"] = np.log(rsi_all["price"])
rsi_all["pred"] = rsi_all["pred"] - rsi_all["pred"].mean()
rsi_all["price"] = rsi_all["pred"].apply(lambda x: np.exp(x))


rsi_list = list()
max_t = 0
min_t = 1e9

# Create a matrix of RSIs from date0 to date1
[t0, t1] = convert_date_to_t(pd.Series([date0, date1]), period)
zone_pred = pd.DataFrame(index=np.arange(t0, t1, 1), columns=zones)
zone_count = pd.DataFrame(index=np.arange(t0, t1, 1), columns=zones)

for (i, zone_no) in enumerate(zones):
    df_zone = df_a[df_a["zone"] == zone_no].reset_index(drop=True)
    print("Zone number: " + str(zone_no) + ". Størrelse: " + str(len(df_zone)))

    try:
        df_ttp = convert_MT_data_to_ttp(df_zone, period, date0)
        OLS_res = create_and_solve_OLS_problem(df_ttp)

        OLS_res.set_index("t", inplace=True)
        zone_pred[zone_no] = OLS_res["pred"]
        zone_count[zone_no] = OLS_res["count"]
    except Exception as e:
        rsi = pd.DataFrame(columns=["count", "date", "price"])
        print("Exception: " + str(e))


# Normalize all columns in zone_pred to have mean 1
zone_pred = zone_pred.apply(lambda x: x - x.mean())

zone_price = zone_pred.copy()
for col in zone_price.columns:
    zone_price[col] = zone_price[col].apply(lambda x: np.exp(x))


zone_count.fillna(0, inplace=True)
# zone_price.fillna(0, inplace=True)

# # Plot heatmap of counts:
# fig, ax = plt.subplots()
# im = ax.imshow(zone_count.values, cmap="hot", interpolation="nearest")
# plt.show(block=False)




# Volume-weight price index based on neighboring zones

rsi_all["pred_w"] = smooth_w(rsi_all["pred"], rsi_all["count"], n = 5)
rsi_all["price_w"] = np.exp(rsi_all["pred_w"])

zone_neighbors = get_zone_neighbors(get_zone_geometry(1000))
np.fill_diagonal(zone_neighbors, 0)

zone_pred.fillna(0, inplace=True)

zone_pred_w = pd.DataFrame(index=zone_pred.index, columns=zone_pred.columns)


for (i, col) in enumerate(zones):
    neighbors = zone_neighbors[i, :]
    dummy = np.zeros(len(zone_pred))
    count = np.zeros(len(zone_pred))
    for j in range(len(zones)):
        if neighbors[j] == 1:
            count = count + zone_count.iloc[:, j]
            dummy = dummy + zone_pred.iloc[:, j] * count

    dummy = dummy / count
    zone_pred_w[col] = dummy

    
    
zone_price_w = zone_pred_w.copy()
for col in zone_price_w.columns:
    zone_price_w[col] = zone_price_w[col].apply(lambda x: np.exp(x))


# Print all price indexes for all zones
fig1 = make_subplots(rows=1, cols=2, shared_xaxes=True, vertical_spacing=0.02)
fig1.append_trace(
    go.Scatter(x=rsi_all["date"], y=rsi_all["price"], name="Price, all"),
    row=1,
    col=1,
)

dates = convert_t_to_date(pd.Series(zone_price.index), period, date0)
for col in zone_price.columns:
    print(col)
    dummy = zone_price[col].copy()
    fig1.append_trace(
        go.Scatter(x=dates, y=dummy, name="Price, zone " + str(col)),
        row=1,
        col=1,
    )


fig1.append_trace(
    go.Scatter(x=rsi_all["date"], y=rsi_all["price"], name="Price, all"),
    row=1,
    col=2,
)

dates = convert_t_to_date(pd.Series(zone_pred_w.index), period, date0)
for col in zone_pred_w.columns:
    print(col)
    dummy = zone_pred_w[col].copy()
    fig1.append_trace(
        go.Scatter(x=dates, y=dummy, name="Price, zone " + str(col)),
        row=1,
        col=2,
    )



fig1.show()



