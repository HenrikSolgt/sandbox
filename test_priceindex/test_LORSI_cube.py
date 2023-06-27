import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solgt.priceindex.grk_zones import zone_func_div100
from solgt.db.MT_parquet import get_parquet_as_df
from solgt.priceindex.lorsi_cube import LORSI_cube_class, train_test_split_rep_sales
from solgt.timeseries.filter import conv_smoother
from compute_CBI import create_CBI_from_HMI_RSI_HMI_AD, get_CBI_HMII_monthly, get_CBI_RSI_weekly, get_CBI_HMI_AD_weekly


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
date1 = datetime.date.today()

period = "monthly"

df_MT = get_parquet_as_df("C:\Code\py\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[postcode] = df_MT[postcode].astype(int)


df_MT_train, df_MT_test = train_test_split_rep_sales(df_MT, test_size=0.2)

# Create initial LORSI cube for all transactions treated as one zone and one PROM
all_LORSI = LORSI_cube_class(df_MT_train, date0, date1, period="weekly")
# Filter in time, with a 5-week flat window
all_LORSI_f = all_LORSI.filter_in_time(w_L=4, window_type="gaussian")


# Set up PROM bins and zone function
PROM_bins = [0, 60, 90]
zone_func = zone_func_div100
# Create initial LORSI cube
split_LORSI = LORSI_cube_class(df_MT_train, date0, date1, period="monthly", zone_func=zone_func, PROM_bins=PROM_bins)


# Filter by zone
split_LORSI_z = split_LORSI.filter_by_zone(w_L=3)
# Filter by PROM
split_LORSI_z_p = split_LORSI_z.filter_by_PROM(w_L=1, window_type="gaussian")
# Convert to weekly
split_LORSI_z_p_w = split_LORSI_z_p.convert_to_period("weekly")
# Add HPF part from all zones and PROMs
split_LORSI_z_p_comb = split_LORSI_z_p_w.add_HPF_part_from_LORSI(all_LORSI_f, other_zone=0, other_PROM=0, w_L=12, window_type="gaussian")


# Score the LORSI cubes

all_LORSI.score_LORSI(df_MT_test)
all_LORSI_f.score_LORSI(df_MT_test)

split_LORSI.score_LORSI(df_MT_test)
split_LORSI_z.score_LORSI(df_MT_test)
split_LORSI_z_p.score_LORSI(df_MT_test)
split_LORSI_z_p_w.score_LORSI(df_MT_test)
split_LORSI_z_p_comb.score_LORSI(df_MT_test)


from plotly.subplots import make_subplots

# Plot figure
fig = make_subplots(rows=1, cols=1)
fig = all_LORSI.add_LORSI_scatter(fig, desc="all", row=1, col=1)
fig = all_LORSI_f.add_LORSI_scatter(fig, desc="all, f", row=1, col=1)
for (i, _) in enumerate(split_LORSI.PROM_bins):
    fig = split_LORSI.add_LORSI_scatter(fig, desc="prom", row=1, col=1, zone_i=14, PROM_i=i)
    fig = split_LORSI_z_p.add_LORSI_scatter(fig, desc="prom_z_p", row=1, col=1, zone_i=14, PROM_i=i)
    fig = split_LORSI_z_p_comb.add_LORSI_scatter(fig, desc="prom_z_p_comb", row=1, col=1, zone_i=14, PROM_i=i)

fig.show()




""""
Add Added Data to LORSI cube, using compute_CBI.create_CBI_from_HMI_RSI_AD
"""

# Create HMI and RSI Price Indeces, as well as HMI_AD using Added Data
HMI_monthly = get_CBI_HMII_monthly(df_MT)
HMI_AD_weekly = get_CBI_HMI_AD_weekly()

HMI_AD_weekly["price_orig"] = HMI_AD_weekly["price"]
HMI_AD_weekly["count_orig"] = HMI_AD_weekly["count"]

# Smooth using something more sophisticated than smooth_w
HMI_AD_weekly["price"], HMI_AD_weekly["count"] = conv_smoother(HMI_AD_weekly["price"], HMI_AD_weekly["count"], w_L=3, window_type="gaussian")

d = all_LORSI_f.count.sum(axis=1).sum(axis=1)

# Find last index where count is above 10 for the numpy array d, which is the filtered, weekly count for all of Oslo. This will be the cut off date for RSI
ind = np.where(d > 10)[0][-1]
RSI_stop_date = all_LORSI_f.get_dates()[ind]


CBI_dfs = []

for (i, _) in enumerate(split_LORSI_z_p_comb.zones_arr):
    for (j, _) in enumerate(split_LORSI.PROM_bins):
        RSI_weekly = pd.DataFrame()
        RSI_weekly["date"] = split_LORSI_z_p_comb.get_dates()
        RSI_weekly["price"] = np.exp(split_LORSI_z_p_comb.LORSI[:, i, j])

        CBI_i_j = create_CBI_from_HMI_RSI_HMI_AD(HMI_monthly, RSI_weekly, HMI_AD_weekly, RSI_stop_date)
        # Normalize CBI so that the log-mean is 0
        CBI_i_j["price"] = np.log(CBI_i_j["price"])
        CBI_i_j["price"] = CBI_i_j["price"] - CBI_i_j["price"].mean()
        CBI_i_j["price"] = np.exp(CBI_i_j["price"])
        CBI_dfs.append(CBI_i_j)
        
        
CBI = np.zeros([len(CBI_dfs[0]), len(split_LORSI_z_p_comb.zones_arr), len(split_LORSI.PROM_bins)])
counter = 0
for (i, _) in enumerate(split_LORSI_z_p_comb.zones_arr):
    for (j, _) in enumerate(split_LORSI.PROM_bins):
        CBI[:, i, j] = CBI_dfs[counter]["price"]
        counter = counter + 1

CBI_dates = CBI_dfs[0]["date"]



from plotly.subplots import make_subplots
import plotly.graph_objects as go
fig = make_subplots(rows=1, cols=1)

i = 14
for (i, _) in enumerate(split_LORSI_z_p_comb.zones_arr):
    for (j, _) in enumerate(split_LORSI.PROM_bins):
        fig = fig.add_trace(go.Scatter(x=CBI_dates, y=CBI[:, i, j]), row=1, col=1)

fig.show()