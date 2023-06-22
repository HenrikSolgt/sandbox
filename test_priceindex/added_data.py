"""
Get added data from Finn.no, by reading .csv files in a passed folder
"""

# Standard packages
import datetime
import os
import re

# Installed packages
import numpy as np
import pandas as pd


def adresse2postcode(s):
    try:
        c = int(s[-9:-4])
    except:
        print("Postcode parse failed: ", s)
        c = None
    return c


def get_added_data(path_AD):

    fn_list = []
    for (_, _, filenames) in os.walk(path_AD):
        fn_list.extend(filenames)
        break

    names = [
        "Adresse",
        "Eierform",
        "Boligtype",
        "PROM",
        "BTA",
        "adcreated",
        "Salgsdato",
        "Omsetningtid",
        "Prisantydning",
        "Pris",
        "Fellesgjeld",
        "m2pris",
        "m2tomt",
        "buildyear",
        "megler",
    ]
    df = pd.DataFrame()
    for fn in fn_list:
        dfc = pd.read_csv(path_AD + fn, names=names)
        df = pd.concat([df, dfc])

    # Drop newbuilds etc.
    df = df.drop_duplicates()
    df = df[
        (~df["Pris"].isna())
        & (~df["Prisantydning"].isna())
        & (~df["PROM"].isna())
        & (df["PROM"].astype(float) < 200)
        & (df["PROM"].astype(float) > 19)
        & (df["Boligtype"] == "Leilighet")
    ]

    df["Pris"] = df["Pris"].apply(lambda x: int(re.sub(" ", "", x)))
    df["Prisantydning"] = df["Prisantydning"].apply(lambda x: int(re.sub(" ", "", x)))

    df = df[
        (df["Prisantydning"] < 2 * df["Pris"])
        & (df["Prisantydning"] > 1 / 2 * df["Pris"])
    ]

    df["Fellesgjeld"] = df["Fellesgjeld"].fillna("0")
    df["Fellesgjeld"] = df["Fellesgjeld"].apply(lambda x: int(re.sub(" ", "", x)))
    df["adcreated"] = df["adcreated"].apply(
        lambda x: datetime.datetime.strptime(x, "%d.%m.%Y")
    )
    df["sold_date"] = df["Salgsdato"].apply(
        lambda x: datetime.datetime.strptime(x, "%d.%m.%Y")
    )
    df = df.drop(["Salgsdato"], axis=1)

    date_cols = ["adcreated", "sold_date"]
    df[date_cols] = df[date_cols].apply(lambda r: r.dt.tz_localize("UTC"))

    df["postcode"] = df["Adresse"].apply(adresse2postcode)

    df_id = pd.read_json("dataprocessing/bydeler.json")
    df_post = pd.read_csv("dataprocessing/postbydel.csv")
    df_post = df_post.join(df_id.set_index("name"), on="BYDEL")
    df_post = df_post.rename({"id": "area_id"}, axis=1)

    df = df.join(df_post[["POSTNR", "area_id"]].set_index("POSTNR"), on="postcode")

    df["buildyear_cat"] = (
        df["buildyear"]
        .apply(
            lambda x: +1 * (x < 2015)
            + 1 * (x < 2010)
            + 1 * (x < 1995)
            + 1 * (x < 1975)
            + 1 * (x < 1960)
            + 1 * (x < 1945)
            + 1 * (x < 1910)
        )
        .fillna(4)
    )
    df["size_cat"] = (
        df["PROM"]
        .astype("float")
        .apply(lambda x: int(np.maximum(np.minimum(np.round(x / 10), 12), 2)))
    )
    df["price_inc_debt"] = df["Pris"] + df["Fellesgjeld"]

    return df
