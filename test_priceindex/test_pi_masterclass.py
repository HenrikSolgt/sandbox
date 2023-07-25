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



t0='fromdate'
t1='todate'
unitkey='unitkey'
kommunenummer='kommunenummer'


PI = Priceindex(return_msg_col=True, print_messages=True)
self = PI

from solgt.db.MT_parquet import get_parquet_as_df
df_MT = get_parquet_as_df( "..\..\py\data\MT.parquet")

dates = df_MT["sold_date"]



# Sample some unitkeys
uks = pd.DataFrame()
uks["unitkey"] = df_MT["unitkey"].sample(1000).reset_index(drop=True)
df = uks


df["fromdate"] = dates.sample(1000).reset_index(drop=True)
df["todate"] = dates.sample(1000).reset_index(drop=True)


k_idx = [10, 12, 15]
for i in k_idx:
    df.loc[i, "unitkey"] = np.NaN
    df.loc[i, "kommunenummer"] = kommunenummer_Oslo
    df.loc[i+400, "unitkey"] = np.NaN



res = PI.get_priceindex_by_unitkey(df)


for (i, col) in enumerate(res.columns):
    print(i, col)
    sub = res[col]
    print(sub.mean())



res2 = PI.get_priceindex_by_kommune()

import plotly.graph_objects as go
fig = go.Figure()
fig = fig.add_trace(go.Scatter(x=res2.index, y=res2["price"]))
fig.show()

res2.reset_index(inplace=True)

fig = go.Figure()
fig = fig.add_trace(go.Scatter(x=res2.index, y=res2["index"]))
fig.show()