import datetime
import pandas as pd
import matplotlib.pyplot as plt

from solgt.priceindex.grk_zones import zone_func_div100
from solgt.db.MT_parquet import get_parquet_as_df
from solgt.priceindex.lorsi_cube import LORSI_cube_class, train_test_split_rep_sales


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
date1 = datetime.date(2022, 1, 1)

period = "monthly"

df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")
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
fig = all_LORSI.add_scatter(fig, desc="all", row=1, col=1)
fig = all_LORSI_f.add_scatter(fig, desc="all, f", row=1, col=1)
for PROM in split_LORSI.PROM_arr:
    fig = split_LORSI.add_scatter(fig, desc="prom", row=1, col=1, PROM=PROM)
    fig = split_LORSI_z_p.add_scatter(fig, desc="prom_z_p", row=1, col=1, PROM=PROM)
    fig = split_LORSI_z_p_comb.add_scatter(fig, desc="prom_z_p_comb", row=1, col=1, PROM=PROM)

fig.show()



all_LORSI_f2 = all_LORSI.filter_in_time(w_L=2, window_type="flat")
all_LORSI_f3 = all_LORSI.filter_in_time(w_L=3, window_type="gaussian")
all_LORSI_f4 = all_LORSI.filter_in_time(w_L=2, window_type="gaussian")

fig = make_subplots(rows=1, cols=1)
fig = all_LORSI.add_scatter(fig, desc="all", row=1, col=1)
fig = all_LORSI_f.add_scatter(fig, desc="all, f, gauss, w_L=4", row=1, col=1)
fig = all_LORSI_f2.add_scatter(fig, desc="all, f, flat, w_L=2", row=1, col=1)
fig = all_LORSI_f3.add_scatter(fig, desc="all, f, gauss, w_L=3", row=1, col=1)
fig = all_LORSI_f4.add_scatter(fig, desc="all, f, gauss, w_L=2", row=1, col=1)

fig.show()


