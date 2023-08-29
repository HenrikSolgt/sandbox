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


def cluster_by_neighbors(geo, tresh_L=500, tresh_U=1000):
    """
    This function assumes that geodata in geo has columns "zone", "geometry" and "count". 
    It clusters the entries in geo based on distance to other entries in, but only if they are adjacent to each other.

    When two zones are clustered, the original zone numbers, as in the column "zone", will be kept in a list called "zonelist".

    Every entry that has at least tresh_U entries will be considered complete, and will not be considered for clustering, but other, smaller entries may be clustered with it.

    For every entry, with count < tresh_U, we consider all neighbors, and cluster the current entry with the closest, available unclustered neighbor.

    If and entry does not have any adjacent, unclustered neighbors, one of the following things will happen:
        1) If the entry has more than tresh_L entries, it will be considered complete, and no more entries will be added to it.
        2) If the entry has less than tresh_L entries, it will be merged with the closest adjacent entry, even if that entry has more than tresh_U entries.

    If the entry has no neigbors at all, it will be considered complete, and no more entries will be added to it.
    """

    geodata_sub = geo.copy()

    # Create the list of zones. First off, all zones only contains one entry, and are not considered complete
    geodata_sub["zonelist"] = geodata_sub["zone"].apply(lambda x: [x])

    # Create a column that indicates if the entry is an isolated zone, that is: A zone with no unclustered adjacent zone, and with count satistfying tresh_L <= count < tresh_U
    geodata_sub["isolated_zone"] = False

    # A variable used to break the while loop
    done = False

    while not done:
        done = True

        # Identify all entries that are already considered a zone on its all. Either because count >= tresh_U, or because it is an isolated zone
        clustered_mask = (geodata_sub["count"] >= tresh_U) | geodata_sub["isolated_zone"]

        # Run the algorithm if there are at least two entries, and the clustered_mask is not all true
        if (len(geodata_sub) >= 2) & (not clustered_mask.all()):
            done = False

            # Compute centroids
            geodata_sub["centroid"] = geodata_sub["geometry"].centroid  

            # Also take out the already clustered entries, for later use
            clustered = geodata_sub[clustered_mask]

            # We now will work on just clustering the entries that have too small count
            unclustered = geodata_sub[~clustered_mask]
            print("Unclustered count:" + str(len(unclustered)))
            
            # The clustered will have their own cluster numbers, starting from 0
            clustered["cluster"] = np.arange(len(clustered))

            # Create a blank cluster column for unclustered
            unclustered["cluster"] = None

            # Start the cluster count at the number of entries with count >= tresh_U
            cluster_count = len(clustered)

            # Loop through all entries in unclustered
            for idx in unclustered.index:
                # Only do something if the entry has not been clustered yet in the current loop
                if unclustered.loc[idx, "cluster"] is None:
                    # Find all neighbors to the current entry, both from clustered and unclustered
                    u_nbhd_idx = unclustered[unclustered.geometry.touches(unclustered.loc[idx, "geometry"])].index

                    # Now only consider the neighbors in u_nbhd_idx that has not yet been clustered in the current loop
                    u_nbhd_idx_2 = u_nbhd_idx[unclustered.loc[u_nbhd_idx, "cluster"].isnull()]

                    # If there are any unclustered neighbors: cluster the current index with the closest, unclustered neighbor
                    if len(u_nbhd_idx_2) > 0:
                        # Find the distances to the centroids of all unclustered neighbors
                        distances = unclustered.loc[u_nbhd_idx_2, "centroid"].distance(unclustered.loc[idx, "centroid"]).sort_values()

                        # Find the index of the closest neighbor
                        closest_idx = distances.index[0]

                        # Set the cluster numbers of the current entry and closest neighbor to the current cluster count, and increment the cluster count
                        unclustered.loc[idx, "cluster"] = cluster_count
                        unclustered.loc[closest_idx, "cluster"] = cluster_count
                        cluster_count += 1
                    else:
                        # Find all neighbors to the current entry, both from clustered and unclustered. 
                        # We already know that there are no unclustered neighbors, meaning that the union of the clustered and unclustered neighbors are all clustered neighbors
                        c_nbhd_idx = clustered[clustered.geometry.touches(unclustered.loc[idx, "geometry"])].index

                        if ((len(u_nbhd_idx) > 0) | (len(c_nbhd_idx) > 0)) & (unclustered.loc[idx, "count"] < tresh_L):
                            # Find the distances to the centroids of all clustered neighbors
                            u_distances = unclustered.loc[u_nbhd_idx, "centroid"].distance(unclustered.loc[idx, "centroid"]).sort_values()
                            c_distances = clustered.loc[c_nbhd_idx, "centroid"].distance(unclustered.loc[idx, "centroid"]).sort_values()

                            # Join these two, and keep the index, and sort by distance
                            distances = pd.concat([u_distances, c_distances]).sort_values()

                            # Find the index of the closest neighbor
                            closest_idx = distances.index[0]

                            # Set the cluster number 
                            # Check which of the two dataframes the closest neighbor belongs to, and set the cluster number of the current entry to the same cluster number
                            if closest_idx in u_nbhd_idx:
                                unclustered.loc[idx, "cluster"] = unclustered.loc[closest_idx, "cluster"]
                            else:
                                unclustered.loc[idx, "cluster"] = clustered.loc[closest_idx, "cluster"]
                        else:
                            # If there are no neighbors at all, or the entry has more than tresh_L entries, set the cluster number to the current cluster count, and increment the cluster count
                            unclustered.loc[idx, "cluster"] = cluster_count
                            # Set isolated_zone to True, indicating that this entry will be defined a zone, even though it does not have count >= tresh_U

                            # If there are no unclustered neighbors at all, it means this zone is isolated, and we should define it as a zone for later use
                            if len(u_nbhd_idx) == 0: 
                                unclustered.loc[idx, "isolated_zone"] = True
                            cluster_count += 1

            # Store all cluster info in geodata_sub
            geodata_sub.loc[clustered.index, "cluster"] = clustered["cluster"]
            geodata_sub.loc[unclustered.index, "cluster"] = unclustered["cluster"]

            # Also store the isolated_zone info
            geodata_sub.loc[unclustered.index, "isolated_zone"] = unclustered["isolated_zone"]

            # Dissolve the geodata_sub by cluster
            geodata_sub_dissolved = geodata_sub.dissolve(
                by="cluster", 
                aggfunc={
                    "count": "sum", 
                    "isolated_zone": "max",
                    "fylke": "first",
                    "zonelist": "sum"
                }
            )

            # Reset the index
            geodata_sub_dissolved.reset_index(inplace=True, drop=True)

            # Store the result in geodata_sub
            geodata_sub = geodata_sub_dissolved

    # Drop the isolated_zone column
    geodata_sub.drop(columns=["isolated_zone"], inplace=True)

    return geodata_sub




def create_zones_by_fylke(geodata, tresh_L=500, tresh_U=1000):
        
    # Do this per fylke
    fylker = geodata["fylke"].unique()

    agg_geodata = gpd.GeoDataFrame()

    # Iterate all fylker
    for fylke in fylker:
        print("Fylke: " + fylke)
        geo = geodata[geodata["fylke"] == fylke]

        res = cluster_by_neighbors(geo, tresh_L=tresh_L, tresh_U=tresh_U)

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