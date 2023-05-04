import datetime
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Solgt packages
from solgt.priceindex.repeatsales import get_RSI
from solgt.priceindex.hedonic import get_HMI
from solgt.db.MT_parquet import get_parquet_as_df

from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date

import geopandas as gpd

fp = "../../data/dataprocessing/geodata/Basisdata_03_Oslo_25832_Grunnkretser_FGDB.gdb"
grunnkretser = gpd.read_file(fp, layer = 'grunnkretser_omrade')
grunnkretser.plot(column = 'grunnkretsnavn', figsize=(6, 6))
plt.show()

# Use WGS 84 (epsg:4326) as the geographic coordinate system
grunnkretser = grunnkretser.to_crs(epsg=4326)





# Load the transactions data form parquet file and convert relevant types
df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")
df_MT["sold_date"] = pd.to_datetime(df_MT["sold_date"]).dt.date
df_MT["area_id"] = df_MT["area_id"].astype(int)

t0 = datetime.date(2014, 1, 1)
t1 = datetime.date(2023, 1, 1)

df0 = df_MT[df_MT["area_id"] == 0].reset_index(drop=True)

df_MT[df_MT["grunnkrets_id"] == 101]["address"]



fig1 = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02)


area_ids = range(1, 17)
for area_id in area_ids:
    df = df_MT[df_MT["area_id"] == area_id].reset_index(drop=True)

    rep_sales = get_RSI(df, t0, t1, period="monthly")

    # Plot
    fig1.append_trace(
        go.Scatter(x=rep_sales["date"], y=rep_sales["count"], name="Count: " + str(area_id)),
        row=1,
        col=1,
    )
    fig1.update_yaxes(type="linear", row=1, col=1)



fig1.show()