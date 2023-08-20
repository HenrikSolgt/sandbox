import datetime
import os
import pandas as pd
import pyarrow.parquet as pq
import pymongo
import re

import solgt.db.MT_parquet

# Constants
date_col = "sold_date"
postcode = "postcode"

default_MT_dev_parquet_file = "..\..\py\data\MT_dev.parquet"


def adresse2postcode(s):
    # Extracts the postcode from the address line, and returns the postcode as an integer
    try:
        # Find the last occurence of a 4 digit number in the string
        matches = re.findall(r'\d{4}', s)
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


def localize_timezone_UTC(df) :
    date_cols = ["sold_date"]
    df[date_cols] = df[date_cols].apply(lambda r: r.dt.tz_localize("UTC"))
    return df


def get_prodDB():
    """Return the PropertiesProduction Mongo database as pymongo object."""

    CONNECTION_STRING = """mongodb+srv://ulfjakob:UdD02fqO9Rt5gx6V@properties.2qcut.mongodb.net/PropertiesProduction?authSource=admin&replicaSet=atlas-x4ox8n-shard-0&readPreference=primary&appname=MongoDB%20Compass&ssl=true"""

    myclient = pymongo.MongoClient(CONNECTION_STRING)
    prodDB = myclient["PropertiesProduction"]

    return prodDB

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



def set_collection_df(df, collection_name=None):
    """
    Insert dataframe into collection "collection_name" in PropertiesProduction database.
    """
    if collection_name is None:
        raise ValueError("collection_name must be specified")
    
    prodDB = get_prodDB()
    collection = prodDB[collection_name]
    collection.drop()
    collection.insert_many(df.T.to_dict().values())


def get_collection_df(collection_name=None):
    """
    Get collection "collection_name" in PropertiesProduction database, and return result as a dataframe.
    """
    if collection_name is None:
        raise ValueError("collection_name must be specified")
    
    prodDB = get_prodDB()
    collection = prodDB[collection_name]

    cursor = collection.find({}, {"_id": 0})
    res = pd.DataFrame(list(cursor))

    return res



df = get_collection_df("matched_transactions")


