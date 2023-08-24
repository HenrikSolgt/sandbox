import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering 

pd.options.mode.chained_assignment = None

from load_data import load_fylker_map, load_fylker_geodata, load_data

gr_div_number = 100


# Load data
df = load_data()
df["zone"] = df["grunnkrets"] // gr_div_number
df["gr_div_number"] = gr_div_number

# Load geographical fylkesdata (geodata)
geodata = load_fylker_geodata()
geodata["zone"] =  geodata["grunnkretsnummer"] // gr_div_number
geodata["gr_div_number"] = gr_div_number

# Dissolve by zone and compute count
geodata = geodata.dissolve(by="zone")
geodata["count"] = df.groupby("zone").size()
geodata["count"].fillna(0, inplace=True)
geodata.reset_index(inplace=True)

# Keep only the columns we need
geodata = geodata[["zone", "geometry", "grunnkretsnummer", "kommunenummer", "gr_div_number", "count"]]

# Compute centroid
geodata["centroid"] = geodata["geometry"].centroid



# Sort by count
geodata.sort_values(by="count", inplace=True, ascending=True)


tresh = 1000

"""
Loop trough the clusters and for every entry with count < thresh, merge it with the closest other entry also with thresh < 1000
"""

geodata_sub = geodata[geodata["count"] < tresh]
geodata_sub.sort_values(by="count", inplace=True, ascending=True)

# geodata_sub.plot(column="count", legend=True, figsize=(10, 10))
# plt.show(block=False)

geodata_sub["cluster"] = None

count_col_idx = geodata_sub.columns.get_loc("count")
cluster_col_idx = geodata_sub.columns.get_loc("cluster")


for idx in geodata_sub.index:
    print(geodata_sub.loc[idx])

