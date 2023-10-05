"""
This file is an ad-hoc implementation of for fetching from the dev database, and storing the data locally as a parquet file.
"""

import datetime
import pandas as pd
import pyarrow.parquet as pq
import pymongo
import re

# Constants
date_col = "sold_date"
postcode = "postcode"

default_MT_dev_parquet_file = "..\..\py\data\MT_dev.parquet"


def adresse2postcode(s):
    # Extracts the postcode from the address line, and returns the postcode as an integer
    try:
        # Find the last occurence of a 4 digit number in the string
        matches = re.findall(r"\d{4}", s)
        if matches:
            c = int(matches[-1])
        else:
            c = None
    except:
        print("Postcode parse failed: ", s)
        c = None
    return c


def get_postcode(df):
    # A function extrating the postcode from the column "address" in dataframe df, and storing it in df, if "address" is present in df.
    if "address" in df.columns:
        df["postcode"] = df["address"].apply(adresse2postcode)
    else:
        df["postcode"] = None

    return df


def localize_timezone_UTC(df):
    # Localize the timezone of the columns in df to UTC
    date_cols = ["sold_date"]
    df[date_cols] = df[date_cols].apply(lambda r: r.dt.tz_localize("UTC"))
    return df


def get_devDB():
    """Return the PropertiesDev Mongo database as pymongo object."""

    CONNECTION_STRING = """mongodb+srv://ulfjakob:UdD02fqO9Rt5gx6V@properties.2qcut.mongodb.net/PropertiesProduction?authSource=admin&replicaSet=atlas-x4ox8n-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true"""

    myclient = pymongo.MongoClient(CONNECTION_STRING)
    devDB = myclient["PropertiesDev"]

    return devDB


def get_matchedtransactions(query={}, limit=int(1e6)):
    """
    Return the matched transactions specified in 'query' from dss as DataFrame.
    Arguments:
        query: dict, query for matched transactions
        limit: integer, max number of returned transactions
    Returns:
        Dataframe of matched transactions
    """
    devDB = get_devDB()
    matched_transactions_collection = devDB["matched_transactions"]
    cursor = matched_transactions_collection.find(query).limit(limit)
    df_mt = pd.DataFrame(list(cursor))

    if df_mt.empty:
        return df_mt

    df_mt = localize_timezone_UTC(df_mt)

    return df_mt


def get_matchedtransactions_timeperiod(t0, t1, limit=1000):
    """
    Return dataframe of matched transactions with 'sold_date' in period [t0, t1].
    Arguments:
        t0: from time
        t1: to time
        limit: integer, max number of returned transactions
    Returns:
        Dataframe of matched transactions
    """
    query = {
        "sold_date": {
            "$gte": t0,
            "$lte": t1,
        }
    }

    return get_matchedtransactions(query, limit)


def get_parquet_as_df(MT_parquet_file=default_MT_dev_parquet_file):
    df = pq.read_table(MT_parquet_file).to_pandas()

    df[date_col] = df[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))

    return df


def update_MT_parquet_file(MT_parquet_file=default_MT_dev_parquet_file):
    """
    Update the locally stored parquet file with the latest transactions data.
    """

    # Fetch raw data from DSS
    data_t0 = datetime.datetime(2000, 1, 1)
    data_t1 = datetime.datetime(3000, 1, 1)
    df = get_matchedtransactions_timeperiod(data_t0, data_t1, limit=int(1e6))

    # Convert _id to string
    df["_id"] = df["_id"].astype(str)

    # Export as parquet
    df.to_parquet(MT_parquet_file, index=False)
