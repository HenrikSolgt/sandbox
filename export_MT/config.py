# Rule for converting grunnkrets to zone
gr_div_number = 100

def grunnkrets_to_zone(grunnkrets):
    return grunnkrets // gr_div_number

# Tweakable constants for the partition creation process
tresh_U = 1000
tresh_L = 500

# Tweakable constants for the LORSI cube
PROM_arr = [0, 60, 90]

# Location of raw data
geodata_folder = "../../py/data/raw_data/geodata/"
fylkesnummer_filename = "../../py/data/raw_data/fylkesnummer.csv"
region_to_fylke_filename = "../../py/data/raw_data/region_to_fylke.csv"
postnummer_kommune_filename = "../../py/data/raw_data/postnummer_kommune.csv"

# Partition data
partitions_geodata_filename = "../../py/data/partition_priceindex/partitions.gpkg"
fylker_geodata_filename = "../../py/data/partition_priceindex/fylker.gpkg"

# Maps
zone_partition_map_filename = "../../py/data/partition_priceindex/zone_partition_map.csv"
postcode_zone_map_filename = "../../py/data/partition_priceindex/postcode_zone_map.csv"

# LORSI data (just the prefix - the name with be extended by a strict ruling system)
LORSIs_partition_class_filename = "../../py/data/partition_priceindex/partition_LORSIs"

