import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_fylker_map, load_fylker_geodata, load_data

# from solgt.priceindex.rsi_score import score_RSI

# MERK: GRUNNKRETS_ID er kun siste 4 siffer i fullstendig grunnkretsidentifikator, hvilket består av kommunenummer (4 siffer) + GRUNNKRETS_ID (4 siffer)


# TODO: Must rewrite score_RSI to use the new repeatedsales code

import plotly.graph_objects as go







"""
TODO: 
- Beregn RSI for hele Norge
- Splitt på fylke, og beregn RSI for hvert fylke
- Sjekk hvor god score vi får for hvert fylke, sammenliknet med å bruke RSI for hele Norge
"""

gr_div_number = 100


# Load data
df = load_data()
df["zone"] = df["grunnkrets"] // gr_div_number
df["gr_div_number"] = gr_div_number

# Load geographical fylkesdata (geodata)
geodata = load_fylker_geodata()

geodata["zone"] = geodata["grunnkretsnummer"] // gr_div_number
geodata["gr_div_number"] = gr_div_number

# Extract columns of interest
geodata = geodata[["zone", "geometry", "gr_div_number"]]

# Dissolve by zone, the first one
geodata = geodata.dissolve(by="zone")

# Compute the number of MT per zone
geodata["count"] = df.groupby("zone").size()
geodata["count"].fillna(0, inplace=True)
geodata.reset_index(inplace=True)


# Sort by count
geodata.sort_values(by="count", ascending=False, inplace=True)

thresh = 1000
zones_div100 = geodata[geodata["count"] > thresh]

geodata = geodata[geodata["count"] < thresh]


"""
LEVEL 1000
"""
df["zone"] = df["zone"] // 10
geodata["zone"] = geodata["zone"] // 10
geodata["gr_div_number"] = geodata["gr_div_number"] * 10


# Dissolve by zone, the first one
geodata = geodata.dissolve(by="zone")

# Compute the number of MT per zone
geodata["count"] = df.groupby("zone").size()
geodata["count"].fillna(0, inplace=True)
geodata.reset_index(inplace=True)

# Sort by count
geodata.sort_values(by="count", ascending=False, inplace=True)

zones_div1000 = geodata[geodata["count"] >= thresh]
geodata = geodata[geodata["count"] < thresh]




"""
LEVEL 10000
"""
df["zone"] = df["zone"] // 10
geodata["zone"] = geodata["zone"] // 10
geodata["gr_div_number"] = geodata["gr_div_number"] * 10


# Dissolve by zone, the first one
geodata = geodata.dissolve(by="zone")

# Compute the number of MT per zone
geodata["count"] = df.groupby("zone").size()
geodata["count"].fillna(0, inplace=True)
geodata.reset_index(inplace=True)

# Sort by count
geodata.sort_values(by="count", ascending=False, inplace=True)

zones_div10000 = geodata[geodata["count"] >= thresh]
geodata = geodata[geodata["count"] < thresh]




"""
LEVEL 100000
"""
df["zone"] = df["zone"] // 10
geodata["zone"] = geodata["zone"] // 10
geodata["gr_div_number"] = geodata["gr_div_number"] * 10


# Dissolve by zone, the first one
geodata = geodata.dissolve(by="zone")

# Compute the number of MT per zone
geodata["count"] = df.groupby("zone").size()
geodata["count"].fillna(0, inplace=True)
geodata.reset_index(inplace=True)

# Sort by count
geodata.sort_values(by="count", ascending=False, inplace=True)

zones_div100000 = geodata[geodata["count"] >= thresh]
geodata = geodata[geodata["count"] < thresh]



"""
LEVEL 1000000
"""
df["zone"] = df["zone"] // 10
geodata["zone"] = geodata["zone"] // 10
geodata["gr_div_number"] = geodata["gr_div_number"] * 10


# Dissolve by zone, the first one
geodata = geodata.dissolve(by="zone")

# Compute the number of MT per zone
geodata["count"] = df.groupby("zone").size()
geodata["count"].fillna(0, inplace=True)
geodata.reset_index(inplace=True)

# Sort by count
geodata.sort_values(by="count", ascending=False, inplace=True)

zones_div1000000 = geodata[geodata["count"] >= thresh]
geodata = geodata[geodata["count"] < thresh]



# Plot the same using go


fig, ax = plt.subplots(figsize=(10, 8))

# Plot each GeoPandas object
zones_div100.plot(ax=ax, color='red', label='A')
zones_div1000.plot(ax=ax, color='blue', label='B')
zones_div10000.plot(ax=ax, color='green', label='C')
zones_div100000.plot(ax=ax, color='purple', label='D')
zones_div1000000.plot(ax=ax, color='black', label='E')
# geodata.plot(ax=ax, color='gray', label='F')

# Show legend
ax.legend()

plt.show()




"""
TESTING OUT BIG ZONES
"""

geodata["zone"] = geodata["grunnkretsnummer"] // 1e6
