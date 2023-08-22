import numpy as np
import pandas as pd
import geopandas as gpd
import sys

import matplotlib.pyplot as plt

gr_krets = "grunnkrets_id"


grk_filenames = [
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_03_Oslo_25832_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_11_Rogaland_25832_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_15_More_og_Romsdal_25832_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_18_Nordland_25833_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_30_Viken_25832_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_34_Innlandet_25832_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_38_Vestfold_og_Telemark_25832_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_42_Agder_25832_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_46_Vestland_25832_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_50_Trondelag_25832_Grunnkretser_FGDB.gdb",
    "C:/Code/py/data/dataprocessing/geodata/Basisdata_54_Troms_og_Finnmark_25833_Grunnkretser_FGDB.gdb"
]


# grunnkretser = gpd.read_file(grk_filename, layer = 'grunnkretser_omrade')

Oslo = gpd.read_file(grk_filenames[0])
Rogaland = gpd.read_file(grk_filenames[1])
More_og_Romsdal = gpd.read_file(grk_filenames[2])
Nordland = gpd.read_file(grk_filenames[3])
Viken = gpd.read_file(grk_filenames[4])
Innlandet = gpd.read_file(grk_filenames[5])
Vestfold_og_Telemark = gpd.read_file(grk_filenames[6])
Agder = gpd.read_file(grk_filenames[7])
Vestland = gpd.read_file(grk_filenames[8])
Trondelag = gpd.read_file(grk_filenames[9])
Troms_og_Finnmark = gpd.read_file(grk_filenames[10])


# Plot Oslo and Viken in the same figure
fig, ax = plt.subplots(figsize=(10,10))
Oslo.plot(ax=ax, color='red')
Rogaland.plot(ax=ax, color='blue')
More_og_Romsdal.plot(ax=ax, color='green')
Nordland.plot(ax=ax, color='yellow')
Viken.plot(ax=ax, color='orange')
Innlandet.plot(ax=ax, color='purple')
Vestfold_og_Telemark.plot(ax=ax, color='pink')
Agder.plot(ax=ax, color='brown')
Vestland.plot(ax=ax, color='black')
Trondelag.plot(ax=ax, color='grey')
Troms_og_Finnmark.plot(ax=ax, color='cyan')

plt.show(block=False)
