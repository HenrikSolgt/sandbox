import load_raw_data

df_MT_raw = load_raw_data.load_MT_data()

# Find all columns starting with "N_"
cols_on_N = [col for col in df_MT_raw if col.startswith("N_")]

# Drop all columns starting with "N_", and some other columns
cols_to_drop = cols_on_N + [
    "oldfylke", 
    "top_floor", 
    "document_date",
    "_id",
    "listing_id", 
    "conveyance_id",
    "region"
]

df_MT = df_MT_raw.drop(columns=cols_to_drop)

csv_file = "../output/MT_dev.csv"

# Export as csv
df_MT.to_csv(csv_file, index=False)