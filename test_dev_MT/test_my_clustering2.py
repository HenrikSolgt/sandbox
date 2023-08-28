import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering 
import geopandas as gpd
import shapely.geometry
from scipy.spatial import cKDTree

pd.options.mode.chained_assignment = None

from load_data import load_fylker_map, load_fylker_geodata, load_data, load_zone_geodata_gpkg

"""
This modified method will make sure every sone is in one piece.
It will only connect zones that are adjacent to each other, and 
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


def cluster_by_neighbor_and_distance(geo):
    """
    This function assumes that geodata has columns geometry and count. 
    It clusters the entries in geo based on distance to other entries in , but only if they are adjacent to each other.

    IF: it is does not have any adjacent, unclustered entries, one of the following things will happen:
        1) If the entry has more than tresh_U entries, it will be considered complete, and no more entries will be added to it.
        2) If the entry has less than tresh_U entries, it will be merged with the closest adjacent entry, even if that entry has more than tresh_U entries.
    """
    # Sort by count, so that the entries with the lowest counts are processed first
    geo.sort_values(by="count", inplace=True, ascending=True)

    # Compute centroid
    geo["centroid"] = geo["geometry"].centroid

    # Create a blank cluster indexer
    geo["cluster"] = None

    cluster_count = 0

    # Loop trough all entries in geo, and for every entry: merge with the closest other entry. Mark both as clustered.
    for (i, idx) in enumerate(geo.index):
        # print(i / len(geo))
        if geo.loc[idx, "cluster"] is None:
            # Set this entry's cluster number to the current cluster count
            geo.loc[idx, "cluster"] = cluster_count
        
            # Extract only the entries that are not clustered, which then excludes the current entry
            geo_sub = geo[geo["cluster"].isnull()]

            # Only proceed if there now are entries left
            if len(geo_sub) > 0:
                # Find the closest entry
                distances = geo_sub.distance(geo.loc[idx, "centroid"]).sort_values()

                # The closest entry is the first in the distances series
                closest_idx = distances.index[0]

                # Set the closest entry's cluster number to the current cluster count
                geo.loc[closest_idx, "cluster"] = cluster_count

            cluster_count += 1

    # If there is as single e are left unclustered, they are left on their own
    geo["cluster"].fillna(cluster_count, inplace=True)

    # Drop the centroid column
    geo.drop(columns=["centroid"], inplace=True)

    return geo


gr_div_number_start = 100


# Load data
df = load_data()
df["zone"] = df["grunnkrets"] // gr_div_number_start

# Load from file
geodata = load_zone_geodata_gpkg()

# Count the number of entries in each zone in df, and store in count
geodata = count_by_zone(df, geodata)

# We only need the geometry and count columns for the clustering
geodata = geodata[["geometry", "count", "fylke"]]


centroids = geodata["geometry"].centroid



tresh = 500

geodata_sub = geodata.copy()
agg_geodata = gpd.GeoDataFrame()

# Do this per fylke
fylker = geodata["fylke"].unique()

for fylke in fylker:
    print("Fylke: " + fylke)

    geodata_fylke = geodata[geodata["fylke"] == fylke].copy()
    geodata_sub = geodata_fylke.copy()

    done = False

    while not done:
        done = True

        # Rewrite above line using pandas concat instead
        clustered_geodata = geodata_sub[geodata_sub["count"] >= tresh]

        agg_geodata = pd.concat([agg_geodata, clustered_geodata], ignore_index=True)

        # All remaining entries with count < thresh will be processed further
        geodata_sub = geodata_sub[geodata_sub["count"] < tresh]

        if len(geodata_sub) >= 2:
            done = False

            # Create a clustering based on distance. Returns a column "cluster" with the cluster number
            geodata_sub_clustered = cluster_by_neighbor_and_distance(geodata_sub)

            # Merge the geodata by the computed clustering
            geodata_sub_clustered_dissolved = geodata_sub_clustered.dissolve(by="cluster", aggfunc="sum", numeric_only=True)

            # Keep fylke
            if "fylke" in geodata_sub.columns:
                geodata_sub_clustered_dissolved["fylke"] = geodata_sub["fylke"].iloc[0]

            # Reset the index
            geodata_sub_clustered_dissolved.reset_index(inplace=True, drop=True)

            # Store the result in geodata_sub
            geodata_sub = geodata_sub_clustered_dissolved
    else:
        # Store just the last one
        agg_geodata = pd.concat([agg_geodata, geodata_sub], ignore_index=True)


