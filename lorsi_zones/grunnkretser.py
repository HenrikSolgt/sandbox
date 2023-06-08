import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

"""
    Loads Basisdata Grunnkrets geometry and returns the geometry of the zones. The zones are defined as the first two digits of the grunnkretsnumber.
    Returns:
        zones (GeoDataFrame): GeoDataFrame with zone number as a column, and the zone geometry as a second column.
"""


zone_div = 1

# Load grunnkretser
fp = "C:/Code/data/dataprocessing/geodata/Basisdata_03_Oslo_25832_Grunnkretser_FGDB.gdb"
grunnkretser = gpd.read_file(fp, layer = 'grunnkretser_omrade')

# The grunnkretsnumber for Oslo consists of only the last four digits
grunnkretser["grunnkretsnummer"] = grunnkretser["grunnkretsnummer"].apply(lambda x: int(x[-4:]))

# The zones are the first two digits of the grunnkretsnumber
grunnkretser["zone"] = grunnkretser["grunnkretsnummer"] // zone_div

# Dissolve by zone and merge grunnkretser constituting the same zone, but keep zone as a column
zones = grunnkretser.dissolve(by = 'zone', as_index = False)[["zone", "geometry"]]


# Plot all zones
zones.plot()
plt.show()