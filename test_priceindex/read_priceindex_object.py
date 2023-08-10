import datetime
import pandas as pd
from solgt.priceindex.priceindex import Priceindex
import pytz


pi = Priceindex()
dates = pi.date.tolist()
prices = pi.price.tolist()

# Create a list
entries2 = [
    {"date": dates[i].strftime("%Y-%m-%d"), "price": prices[i]}
    for i in range(len(dates))
]

res = {"entries: ": entries2}


d1 = datetime.date(2008, 1, 1)
d2 = datetime.datetime(2019, 1, 2).replace(tzinfo=pytz.utc)

# Initialize d5 as a pd Series of datetime.datetime
d5 = pd.Series([d1, d2])


pi.date

pi.interpolate(d5)
