import datetime
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import Point

# Solgt packages
from solgt.priceindex.repeatsales import get_RSI
from solgt.priceindex.hedonic import get_HMI
from solgt.db.MT_parquet import get_parquet_as_df

from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date

# Constants
price_col = "price_inc_debt"
date_col = "sold_date"
key_col = "unitkey"
period = "monthly"
gr_krets = "grunnkrets_id"

fp = "C:/Code/data/dataprocessing/geodata/Basisdata_03_Oslo_25832_Grunnkretser_FGDB.gdb"
grunnkretser = gpd.read_file(fp, layer = 'grunnkretser_omrade')

# Union of grunnkretser by zone:

grunnkretser["grunnkretsnummer"] = grunnkretser["grunnkretsnummer"].apply(lambda x: int(x[-4:]))
grunnkretser["zone"] = grunnkretser["grunnkretsnummer"] // 100

# grunnkretser.plot(column = 'grunnkretsnummer', figsize=(6, 6), legend = True)
# plt.show(block=False)




# # Load raw data
# df_raw = get_parquet_as_df("C:\Code\data\MT.parquet")

# # Copy and preprocess
# df = df_raw.copy()
#     # Remove entries without a valid grunnkrets
# df = df[~df[gr_krets].isna()].reset_index(drop=True)
#     # Typecast to required types
# df[date_col] = df[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
# df[gr_krets] = df[gr_krets].astype(int)

# # Create zones
# df["zone"] = df[gr_krets] // 100

# MT_zones = pd.Series(df["zone"].unique()).sort_values().reset_index(drop=True)


# df.sort_values(by=gr_krets, inplace = True)


# Dissolve by zone, but keep zone as a column
zones = grunnkretser.dissolve(by = 'zone', as_index = False)[["zone", "geometry"]]
zones.plot(column="zone", figsize=(6, 6), legend = True)
plt.show(block=False)



"""
Find all neighbours of a zone using GeoPandas' intersects method
"""
N = len(zones)
neighbor_matrix = np.zeros((N, N))

for (i, zone) in enumerate(zones["zone"]):
    neighbor_matrix[:, i] = zones["geometry"][i].intersects(zones["geometry"]).astype(int)
    
np.fill_diagonal(neighbor_matrix, 0)


"""
Find distances to centroids of all zones
"""
dist_matrix = np.zeros((N, N))
zones["centroid"] = zones["geometry"].centroid
for (i, zone) in enumerate(zones["zone"]):
    dist_matrix[:, i] = zones["centroid"][i].distance(zones["centroid"])
    