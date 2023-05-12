import numpy as np
import pandas as pd
import geopandas as gpd


def get_zone_geometry(div_num = 100):
    """
        Loads Basisdata Grunnkrets geometry and returns the geometry of the zones. The zones are defined as the first two digits of the grunnkretsnumber.
        Returns:
            zones (GeoDataFrame): GeoDataFrame with zone number as a column, and the zone geometry as a second column.
    """
    # Load grunnkretser
    fp = "C:/Code/data/dataprocessing/geodata/Basisdata_03_Oslo_25832_Grunnkretser_FGDB.gdb"
    grunnkretser = gpd.read_file(fp, layer = 'grunnkretser_omrade')

    # The grunnkretsnumber for Oslo consists of only the last four digits
    grunnkretser["grunnkretsnummer"] = grunnkretser["grunnkretsnummer"].apply(lambda x: int(x[-4:]))

    # The zones are the first two digits of the grunnkretsnumber
    grunnkretser["zone"] = grunnkretser["grunnkretsnummer"] // div_num

    # Dissolve by zone and merge grunnkretser constituting the same zone, but keep zone as a column
    zones = grunnkretser.dissolve(by = 'zone', as_index = False)[["zone", "geometry"]]

    # Return the zone geometry
    return zones


def get_zone_neighbors(zones):
    """
    Find all neighbours of a zone using GeoPandas' intersects method. 
    Returns: 
        neighbor_matrix: A numpy matrix with 1 if two zones are neighbors, 0 otherwise. It is also 0 on the diagonal.
    """
    N = len(zones)
    neighbor_matrix = np.zeros((N, N))

    for i in range(N):
        neighbor_matrix[:, i] = zones["geometry"][i].intersects(zones["geometry"])

    np.fill_diagonal(neighbor_matrix, 0)

    res = pd.DataFrame(neighbor_matrix, index = zones["zone"], columns = zones["zone"])
    
    return res


def get_zone_controid_distances(zones):

    """
    Find distances to centroids of all zones
    """

    N = len(zones)
    dist_matrix = np.zeros((N, N))

    zones["centroid"] = zones["geometry"].centroid

    for i in range(N):
        dist_matrix[:, i] = zones["centroid"][i].distance(zones["centroid"])

    res = pd.DataFrame(dist_matrix, index = zones["zone"], columns = zones["zone"])

    return dist_matrix
    