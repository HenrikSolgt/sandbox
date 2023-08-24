import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import AgglomerativeClustering 


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

n_clusters = len(geodata) // 10


# Clustering algorithm
centroids = geodata["centroid"]
X = centroids.apply(lambda p: (p.x, p.y)).tolist()

model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
clusters = model.fit_predict(X)

geodata["cluster"] = clusters

while True:
    cluster_sums = geodata.groupby('cluster')['count'].sum()
    underpopulated_clusters = cluster_sums[cluster_sums < 1000].index.tolist()

    if not underpopulated_clusters:
        break

    for cluster in underpopulated_clusters:
        # Find the closest neighboring cluster
        this_cluster = geodata[geodata['cluster'] == cluster]
        neighboring_clusters = geodata[geodata.geometry.touches(this_cluster.unary_union)]

        if not neighboring_clusters.empty:
            closest_cluster = neighboring_clusters.iloc[0]['cluster']
            geodata.loc[geodata['cluster'] == cluster, 'cluster'] = closest_cluster



aggregated_geodata = geodata.dissolve(by="cluster", aggfunc="sum")

# Sort by count
aggregated_geodata.sort_values(by="count", inplace=True)

# Plot
aggregated_geodata.plot(column="count", legend=True, figsize=(10, 10))
plt.show(block=False)