import datetime
import numpy as np
import pandas as pd

from solgt.db.MT_parquet import get_parquet_as_df

from solgt.priceindex.priceindex import Priceindex
from solgt.priceindex.cbi_cube_api_class import CBI_cube_api_class
import solgt.priceindex.priceindex_utils as pi_utils

kommunenummer_Oslo = 301
kommunenummer_default = kommunenummer_Oslo  # 301 is Oslo


# TODO: Kjør oppslag av PROM og adresse for

# Suppress warning
pd.options.mode.chained_assignment = None  # default='warn'


t0 = "fromdate"
t1 = "todate"
prom = "PROM"
postcode = "postcode"
unitkey = "unitkey"
kommunenummer = "kommunenummer"


PI = Priceindex(return_msg_col=False, print_messages=True)
self = PI

from solgt.db.MT_parquet import get_parquet_as_df

df_MT = get_parquet_as_df("..\..\py\data\MT.parquet")

dates = df_MT["sold_date"]


# Sample some unitkeys
df = pd.DataFrame()
df[["unitkey", "address", "postcode", "PROM"]] = (
    df_MT[["unitkey", "address", "postcode", "PROM"]]
    .sample(1000)
    .reset_index(drop=True)
)

df["fromdate"] = dates.sample(1000).reset_index(drop=True)
df["todate"] = dates.sample(1000).reset_index(drop=True)

# Swap fromdate and todate if fromdate > todate
mask = df["fromdate"] > df["todate"]
df.loc[mask, ["fromdate", "todate"]] = df.loc[mask, ["todate", "fromdate"]].values


df.loc[0:3, "PROM"] = np.nan
df.loc[0:3, "postcode"] = np.nan
df.loc[0, "unitkey"] = "301-149-552-3"
df.loc[1, "unitkey"] = "15125"
df.loc[2, "unitkey"] = "0"
df = df.head(8)

res = PI.reindex_by_unitkey(df)
print(res)
