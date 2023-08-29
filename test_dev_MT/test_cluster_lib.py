from solgt.priceindex.zone_cluster_algorithm import cluster_zones_by_neighbors

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import geopandas as gpd
import time

pd.options.mode.chained_assignment = None

from load_data import load_fylker_map, load_fylker_geodata, load_data, load_zone_geodata_gpkg

"""
This modified method will make sure every zone is in one connected piece.
It will only connec two zones that are adjacent to each other, and 
Once a zone is connected so that it has more than tresh_U entries, it will be considered complete, and no more entries will be added to it.

We will also usea lower treshold tresh_L, used if an isolated sone, with no more adjacent zones with count < tresh_U, has more than tresh_L entries, if will be left alone. 
If the isolated zone has less than tresh_L entries, it will be merged with the closest adjacent zone, even if that zone has more than tresh_U entries.

"""


def count_by_zone(df, geodata):
    # Count the number of entries in each zone in df, and store in count
    geodata.set_index("zone", inplace=True)
    geodata["count"] = df.groupby("zone").size()
    geodata["count"].fillna(0, inplace=True)
    geodata.reset_index(inplace=True)
    return geodata


def create_zones_by_fylke(geodata, tresh_L=500, tresh_U=1000):
        
    # Do this per fylke
    fylker = geodata["fylke"].unique()

    agg_geodata = gpd.GeoDataFrame()

    # Iterate all fylker
    for fylke in fylker:
        print("Fylke: " + fylke)
        geo = geodata[geodata["fylke"] == fylke]

        res = cluster_zones_by_neighbors(geo, tresh_L=tresh_L, tresh_U=tresh_U)

        agg_geodata = pd.concat([agg_geodata, res], ignore_index=True)

    return agg_geodata




gr_div_number_start = 100


# Load data
df = load_data()
df["zone"] = df["grunnkrets"] // gr_div_number_start

# Load from file
geodata = load_zone_geodata_gpkg()

# Count the number of entries in each zone in df, and store in count
geodata = count_by_zone(df, geodata)

# We only need the geometry and count columns for the clustering
geodata = geodata[["zone", "geometry", "count", "fylke"]]



agg_geodata = create_zones_by_fylke(geodata, tresh_L=500, tresh_U=1000)


""" PLOTTING """
agg_geodata.reset_index(inplace=True, drop=True)

# Sort by count
agg_geodata.sort_values(by="count", ascending=True, inplace=True)


# Plot
A = agg_geodata[agg_geodata["count"] < 1000]
B = agg_geodata[agg_geodata["count"] >= 1000]

# Plot in same axis
fig, ax = plt.subplots(figsize=(10, 10))
A.plot(ax=ax, color="white", edgecolor="black")
B.plot(ax=ax, color="red", edgecolor="black")
plt.show()



# Count zones per fylke
agg_geodata.groupby("fylke").size()


import time
start = time.time()
geo = load_fylker_geodata()
print(time.time() - start)

# Open gpkg file
start = time.time()
geo2 = gpd.read_file("../../py/data/dataprocessing/geodata/geodata.gpkg")
print(time.time() - start)


geo = geo[["grunnkretsnummer", "kommunenummer", "fylke", "geometry"]]

# Save geo as gpkg
start = time.time()
geo.to_file("../../py/data/dataprocessing/geodata/geodata_all.gpkg", driver="GPKG")
print(time.time() - start)


# Loadas geo3
start = time.time()
geo3 = gpd.read_file("../../py/data/dataprocessing/geodata/geodata_all.gpkg")
print(time.time() - start)