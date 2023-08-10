import datetime
import numpy as np
import pandas as pd

from solgt.db.MT_parquet import get_parquet_as_df

from solgt.priceindex.priceindex import Priceindex
from solgt.priceindex.cbi_cube_api_class import CBI_cube_api_class
import solgt.priceindex.priceindex_utils as pi_utils

kommunenummer_Oslo = 301
kommunenummer_default = kommunenummer_Oslo  # 301 is Oslo


t0 = "fromdate"
t1 = "todate"
unitkey = "unitkey"
kommunenummer = "kommunenummer"


PI = Priceindex(return_msg_col=True, print_messages=True)
self = PI

from solgt.db.MT_parquet import get_parquet_as_df

df_MT = get_parquet_as_df("..\..\py\data\MT.parquet")

dates = df_MT["sold_date"]


# Sample some unitkeys
uks = pd.DataFrame(data={"unitkey": ["301-149-552-3"]})

df = uks

df["fromdate"] = datetime.date(2020, 1, 1)
df["todate"] = datetime.date(2022, 1, 1)

res = PI.reindex(df)
