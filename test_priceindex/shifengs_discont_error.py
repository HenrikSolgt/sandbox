import pandas as pd
from solgt.priceindex import priceindex
import datetime

CBI = priceindex.Priceindex(print_messages=True, return_msg_col=True)


dfx = pd.DataFrame()
dfx["fromdate"] = pd.date_range(start="2008-01-01", end="2023-07-01").date
dfx["todate"] = datetime.datetime(2022, 2, 16).date()
dfx["kommunenummer"] = "301"


# Something wrong with the reindex on kommunenummer

# Also: "success" should be FALSE for the first rows outside bounds

A[A["msg"] != ""]


A = CBI.reindex(dfx)
B = A.set_index("fromdate")
C = B["dp"]

A.plot()

# Show figure
import matplotlib.pyplot as plt

plt.show()
