import datetime
import pandas as pd
from solgt.priceindex.priceindex import Priceindex


pi = Priceindex()
dates = pi.date.tolist()
prices = pi.price.tolist()

# Create a list
entries2 = [{"date": dates[i].strftime("%Y-%m-%d"), "price": prices[i]} for i in range(len(dates))]

res = {"entries: ": entries2}

