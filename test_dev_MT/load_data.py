import pandas as pd
import MT_dev_parquet
import geopandas as gpd
import re



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


geodata_file = "C:/Code/py/data/dataprocessing/geodata/geodata_parquet.gpkg"



def load_postcode_kommune():
    # Load the file
    postcode_kommune = pd.read_csv("../../py/data/dataprocessing/postnummer_kommune.csv", sep=",")
    # Keep only postnummer and kommune
    postcode_kommune = postcode_kommune[["postnummer", "kommunenummer", "kommunenavn"]]
    # Rename columns
    postcode_kommune.columns = ["postcode", "kommunenummer", "kommunenavn"]
    # Set postcode as index
    postcode_kommune.set_index("postcode", inplace=True)

    return postcode_kommune


def load_fylker_map():
    # Load fylker.csv into a DataFrame, and set the index to "Region"
    fylker_map = pd.read_csv("fylker.csv", sep=",")
    fylker_map.set_index("Region", inplace=True)

    # Find unique fylker

    fylker = fylker_map["Fylke"].unique()
    fylker.sort()

    nyfylker_map = pd.DataFrame(data=fylker, columns = ["Fylke"])

    nyfylker_map["Nyfylke"] = [
        "Viken",
        "Agder",
        "Viken",
        "Troms og Finnmark",
        "Innlandet",
        "Vestland",
        "Møre og Romsdal",
        "Trøndelag",
        "Nordland",
        "Innlandet",
        "Oslo",
        "Rogaland",
        "Vestland",
        "Trøndelag",
        "Vestfold og Telemark",
        "Troms og Finnmark",
        "Agder",
        "Vestfold og Telemark",
        "Viken"
    ]

    nyfylker_map.set_index("Fylke", inplace=True)

    return fylker_map, nyfylker_map


def load_data():
    df = MT_dev_parquet.get_parquet_as_df()
        
    # Remove geographical outsiders
    df = df[df["lat"] > 55].reset_index(drop=True)

    # Compute region
    df["region"] = df["postcode"] // 100
    
    fylker_map, nyfylker_map = load_fylker_map()

    # Use fylker to put the correct region to fylke 
    df["fylke"] = df["region"].map(fylker_map["Fylke"])

    df["nyfylke"] = df["fylke"].map(nyfylker_map["Nyfylke"])

    # Set kommuneneummer and kommune name
    postcode_kommune = load_postcode_kommune()
    df["kommunenummer"] = df["postcode"].map(postcode_kommune["kommunenummer"])
    df["kommune"] = df["postcode"].map(postcode_kommune["kommunenavn"])

    # Create grunnkrets
    df["grunnkrets"] = (df["kommunenummer"] * 10000 + df["grunnkrets_id"])

    return df




def load_fylker_geodata():
    """
    Returns a list of tuples with (fylkename, geodataframe), by loading all files in grk_filenames
    """

    pattern = r'(\d+)_(\w+)_(\d+)'

    # Put everything in a single geodataframe with a column "fylke" containing the fylke name
    geodata = gpd.GeoDataFrame()
    
    for (i, filename) in enumerate(grk_filenames):
        match = re.search(pattern, filename)

        # Find the name by removing the path and the extension, and substituting underscore with space
        # Special treatment of Møre og Romsdal, which has an o istead of ø
        substring = match.group(2).replace("_", " ")
        if substring == "More og Romsdal":
            substring = "Møre og Romsdal"
            
        # Store in the list
        gpd_df = gpd.read_file(filename)
        
        # Store the fylke name
        gpd_df["fylke"] = substring

        if i == 0:
            geodata = gpd.GeoDataFrame(gpd_df)
        else:
            gpd_df = gpd_df.to_crs(geodata.crs)
            geodata = gpd.GeoDataFrame(pd.concat([geodata, gpd_df], ignore_index=True))

        # Convert to int
        geodata["grunnkretsnummer"] = geodata["grunnkretsnummer"].apply(lambda x: int(x))

    return geodata



def dissolve_by_zone(geodata):
    # Dissolve the geodata by zone 
    geodata = geodata.dissolve(by="zone")
    geodata.reset_index(inplace=True)
    return geodata


def create_fylker_geodata_gpkg(gr_div_number_start):
    # Load geographical fylkesdata (geodata)
    geodata = load_fylker_geodata()

    # Keep only the columns we need
    geodata = geodata[["geometry", "grunnkretsnummer", "kommunenummer", "fylke"]]

    # Dissolve into the smalles geographical units 
    geodata["zone"] =  geodata["grunnkretsnummer"] // gr_div_number_start
    geodata = dissolve_by_zone(geodata) 

    geodata.to_file("C:/Code/py/data/dataprocessing/geodata/geodata.gpkg", driver="GPKG")



def load_zone_geodata_gpkg():
    geodata = gpd.read_file("C:/Code/py/data/dataprocessing/geodata/geodata.gpkg")
    return geodata


create_fylker_geodata_gpkg(100)