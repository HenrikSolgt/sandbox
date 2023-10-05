"""
This file contains functions for loading all raw data used by the partition creation process.
None of the data loaded by this file is generated by this package.
"""

# Python packages
import os
import re
import pandas as pd
import geopandas as gpd

# Local packages
import MT_dev_parquet
import config


def load_fylkesnummer():
    # Load fylker.csv into a DataFrame, and set the index to "fylke". This is a map from fylke to nyfylke.
    fylkesnummer = pd.read_csv(config.fylkesnummer_filename, sep=",")
    fylkesnummer.set_index("Fylke", inplace=True)

    return fylkesnummer


def load_region_to_fylke_map():
    # Load fylker.csv into a DataFrame, and set the index to "Region". This is a map from regi (derived from Postnummer) to fylke and nyfylke.
    region_to_fylke_map = pd.read_csv(config.region_to_fylke_filename, sep=",")
    region_to_fylke_map.set_index("Region", inplace=True)

    return region_to_fylke_map


def load_postcode_kommune():
    # Load postnummer.csv into a DataFrame, and set the index to "postnummer". This is a mapping from postnummer to poststed, kommunenummer and kommunenavn.
    postnummer = pd.read_csv(config.postnummer_kommune_filename, sep=",")
    postnummer.set_index("postnummer", inplace=True)

    return postnummer


def load_MT_data():
    # Load all matched transactions data from dev, and also adds information about region, fylke, nyfylke, kommunenumer, kommune and grunnkrets.

    df = MT_dev_parquet.get_parquet_as_df()

    # Remove geographical outsiders
    df = df[df["lat"] > 55].reset_index(drop=True)

    # Compute region
    df["region"] = df["postcode"] // 100

    # Load fylker map, and map region to fylke and nyfylke
    fylker_map = load_region_to_fylke_map()
    df["oldfylke"] = df["region"].map(fylker_map["Oldfylke"])
    df["fylke"] = df["region"].map(fylker_map["Fylke"])

    # Set kommuneneummer and kommune name
    postcode_kommune = load_postcode_kommune()
    df["kommunenummer"] = df["postcode"].map(postcode_kommune["kommunenummer"])
    df["kommune"] = df["postcode"].map(postcode_kommune["kommunenavn"])

    # Create grunnkrets
    df["grunnkrets"] = df["kommunenummer"] * 10000 + df["grunnkrets_id"]

    return df


def load_fylker_geodata():
    """
    Loads all fylker geodata from the data folder, and returns a single geodatafram with all geodata, and a column "fylke" containing the fylke name.
    """

    # Find all files in directory with ending .gdb
    fylker_filenames = [
        os.path.join(config.geodata_folder, filename)
        for filename in os.listdir(config.geodata_folder)
        if filename.endswith(".gdb")
    ]

    # Pattern to extract the fylke name from the filename
    pattern = r"(\d+)_(\w+)_(\d+)"

    # Put everything in a single geodataframe with a column "fylke" containing the fylke name
    geodata = gpd.GeoDataFrame()

    for (i, filename) in enumerate(fylker_filenames):
        match = re.search(pattern, filename)

        # Find the name by removing the path and the extension, and substituting underscore with space
        # Special treatment of Møre og Romsdal and Trøndelag, which have an o istead of ø
        substring = match.group(2).replace("_", " ")
        if substring == "More og Romsdal":
            substring = "Møre og Romsdal"

        if substring == "Trondelag":
            substring = "Trøndelag"

        # Store in the list
        gpd_df = gpd.read_file(filename)

        # Store the fylke name
        gpd_df["fylke"] = substring

        if i == 0:
            # Create the geodataframe, and keep the crs from gpd_df
            geodata = gpd.GeoDataFrame(gpd_df, crs=gpd_df.crs)
        else:
            gpd_df = gpd_df.to_crs(geodata.crs)
            geodata = gpd.GeoDataFrame(pd.concat([geodata, gpd_df], ignore_index=True))

        # Convert to int
        geodata["grunnkretsnummer"] = geodata["grunnkretsnummer"].apply(
            lambda x: int(x)
        )

    return geodata