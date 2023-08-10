import datetime
import os
import pymongo
import pandas as pd


def localize_timezone_UTC(df) :
    date_cols = ["sold_date"]
    df[date_cols] = df[date_cols].apply(lambda r: r.dt.tz_localize("UTC"))
    return df


def get_DB():
    """Return the dev or production Mongo database as pymongo object depending on the environment variable 'ENV'."""
    if os.getenv('APP_ENV') == "development":
        return get_devDB()
    else:
        return get_prodDB()

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
    prodDB = get_devDB()
    matched_transactions_collection = prodDB["matched_transactions"]
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





# Fetch raw data from DSS
data_t0 = datetime.datetime(2000, 1, 1)
data_t1 = datetime.datetime(3000, 1, 1)
df = get_matchedtransactions_timeperiod(
    data_t0, data_t1, limit=int(1e6)
)