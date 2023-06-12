import numpy as np
import pandas as pd
import geopandas as gpd

gr_krets = "grunnkrets_id"


def get_grk_geometry(fp="C:/Code/data/dataprocessing/geodata/Basisdata_03_Oslo_25832_Grunnkretser_FGDB.gdb"):
    # Loads the grunnkrets geometry from the file fp, converts the grunnkretsnumber to integer and returns the result
    # Load the file
    grunnkretser = gpd.read_file(fp, layer = 'grunnkretser_omrade')

    # The grunnkretsnumber for Oslo consists of only the last four digits
    grunnkretser["grunnkretsnummer"] = grunnkretser["grunnkretsnummer"].apply(lambda x: int(x[-4:]))

    return grunnkretser


def dissolve_grk_by_zone(grunnkretser):
    """
    Dissolve the grunnkrets by zone and merge grunnkretser constituting the same zone, but keep zone as a column 
    """
    # Dissolve by zone and merge grunnkretser constituting the same zone, but keep zone as a column
    zones_geometry = grunnkretser.dissolve(by = 'zone', as_index = False)[["zone", "geometry"]]

    # Return the zone geometry
    return zones_geometry


def get_zone_neighbors(zones_geometry):
    """
    Find all neighbours of a zone using GeoPandas' intersects method. 
    Returns: 
        neighbor_matrix: A Pandas DataFrame with 1 if two zones are neighbors, 0 otherwise. It is also 0 on the diagonal.
    """
    N = len(zones_geometry)
    neighbor_matrix = np.zeros((N, N))

    for i in range(N):
        neighbor_matrix[:, i] = zones_geometry["geometry"][i].intersects(zones_geometry["geometry"])

    np.fill_diagonal(neighbor_matrix, 0)

    res = pd.DataFrame(neighbor_matrix, index = zones_geometry["zone"], columns = zones_geometry["zone"])
    
    return res


def get_zone_controid_distances(zones_geometry):

    """
    Find distances to centroids of all zones
    """

    N = len(zones_geometry)
    dist_matrix = np.zeros((N, N))

    zones_geometry["centroid"] = zones_geometry["geometry"].centroid

    for i in range(N):
        dist_matrix[:, i] = zones_geometry["centroid"][i].distance(zones_geometry["centroid"])

    res = pd.DataFrame(dist_matrix, index = zones_geometry["zone"], columns = zones["zone"])

    return res
    

def get_grk_zones_and_neighbors(zone_func):
    grunnkretser = get_grk_geometry()
    grunnkretser["zone"] = zone_func(grunnkretser["grunnkretsnummer"])
    zones_geometry = dissolve_grk_by_zone(grunnkretser)
    zones_neighbors = get_zone_neighbors(zones_geometry)

    return zones_geometry, zones_neighbors