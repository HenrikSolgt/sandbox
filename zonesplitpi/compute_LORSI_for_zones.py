# Standard python packages
import datetime
import numpy as np
import pandas as pd


# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df, update_MT_parquet_file
from solgt.priceindex.repeatsales import add_derived_MT_columns, get_repeated_idx, get_df_ttp_from_RS_idx, create_and_solve_LORSI_OLS_problem


# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
columns_of_interest = [key_col, date_col, price_col, gr_krets]

# update_MT_parquet_file("C:\Code\data\MT.parquet")

def load_MT_data():
    # Load raw data
    df = get_parquet_as_df("C:\Code\data\MT.parquet")

    # Preprocess: manipulation of already existing columns
        # Remove entries without a valid grunnkrets
    df = df[~df[gr_krets].isna()].reset_index(drop=True)
        # Typecast to required types
    df[date_col] = df[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
    df[gr_krets] = df[gr_krets].astype(int)

    return df


# Get LORSI and count for the whole region
def get_LORSI_and_count(df, t0, t1):
    """
    Get Log-RSI (LORSI) and count for a all matched transactions in DataFrame df, for the time period [t0, t1).
    All raw transactions from df is used. The filtering is performed after the transactions have been matched.
    Inputs:
        - df: DataFrame with columns "id", "y", "t"
        - t0: Start time
        - t1: End time
    """

    T_arr = np.arange(t0, t1)

    # Get repeated sales index and create df_ttp
    R_idx = get_repeated_idx(df)
    df_ttp = get_df_ttp_from_RS_idx(df, R_idx)

    if (len(df_ttp) > 0):
        LORSI_res = create_and_solve_LORSI_OLS_problem(df_ttp)
        LORSI_res = LORSI_res[(LORSI_res["t"] >= t0) & (LORSI_res["t"] < t1)].reset_index(drop=True)

        # Split LORSI_res into LORSI and count dataframes
        LORSI = LORSI_res[["t", "pred"]].set_index(["t"]).reindex(T_arr)
        count = LORSI_res[["t", "count"]].set_index(["t"]).reindex(T_arr, fill_value=0)
    else:
        LORSI = pd.DataFrame(index=T_arr, columns=["pred"])
        count = pd.DataFrame(index=T_arr, columns=["count"]).fillna(0)

    # Remove index names
    LORSI.index.name = None
    count.index.name = None

    return LORSI, count


# Create LORSI and count for all zones
def get_LORSI_and_count_for_zones(df, t0, t1):
    """
    Get LORSI and count for all matched transactions in DataFrame df, for the time period [t0, t1).
    All raw transactions from df are used. The filtering on time period [t0, t1) is performed after the transactions have been matched.
    Note that only the zones occuring in df are included in the output.
    Inputs:
        - df: DataFrame with columns "unitkey", "price_inc_debt", "zone", "t"
        - t0: Start time
        - t1: End time
    """

    zones_arr = df["zone"].unique()
    zones_arr.sort()

    T_arr = np.arange(t0, t1)
    # Create empty dataframes
    zone_LORSI = pd.DataFrame(index=T_arr, columns=zones_arr)
    zone_counts = pd.DataFrame(index=T_arr, columns=zones_arr)

    for zone_no in zones_arr:
        # Filter the MT data for the current zone
        df_zone = df[df["zone"] == zone_no].reset_index(drop=True)
        print("Zone number: " + str(zone_no) + ". Number of transactions: " + str(len(df_zone)))

        LORSI, count = get_LORSI_and_count(df_zone, t0, t1)

        zone_LORSI[zone_no] = LORSI
        zone_counts[zone_no] = count

    # Substitute NaN with 0
    zone_LORSI.fillna(0, inplace=True)
    zone_counts.fillna(0, inplace=True)

    # Convert to int
    zone_counts = zone_counts.astype(int)

    return zone_LORSI, zone_counts


