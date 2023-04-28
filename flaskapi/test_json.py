import json
import datetime
import flask
import pandas as pd
from solgt.priceindex.priceindex import Priceindex

date0_str = "2020-01-02"
date1_str = "2021-07-01"

date0 = datetime.datetime.strptime(date0_str, "%Y-%m-%d").date()
date1 = datetime.datetime.strptime(date1_str, "%Y-%m-%d").date()

date0_2 = date0.strftime("%Y-%m-%d")
date1_2 = date1.strftime("%Y-%m-%d")

pi = Priceindex()

pi0 = float(pi.interpolate(date0))
pi1 = float(pi.interpolate(date1))


res = {
    "date0": date0_2,
    "date1": date1_2,
    "pi0": pi0,
    "pi1": pi1
}

pi = Priceindex()
pi0 = pi.interpolate(date0)
pi1 = pi.interpolate(date1)

dates = pd.Series([date0, date1])

pi01 = pi.interpolate(dates).tolist()

# Create a json string. The type is string
b = json.dumps(res)

# Convert the json string to a dictionary. The type is dict
c = json.loads(b)

# Do manipulations on the dictionary
c["pi1"] = c["pi1"] / c["pi0"]
c["pi0"] = 1.0

# Convert the dictionary to a json string. The type is string
d = json.dumps(c)



# Test list of units
datastr = """
{
    "units": [
        {
            "unitkey": "301-214-58-42",
            "date0": "2020-01-01",
            "date1": "2021-07-01"
        },
        {
            "unitkey": "301-214-58-43",
            "date0": "2019-01-01",
            "date1": "2022-07-01"
        },
        {
            "unitkey": "301-214-58-44",
            "date0": "2019-03-01",
            "date1": "2022-09-01"
        }
    ]
}
"""

# Convert the json string to a dictionary. The type is dict
req = json.loads(datastr)

units = req["units"]

N = len(units)
unitkeys = [None] * N
dates0 = [None] * N
dates1 = [None] * N

for (i, unit) in enumerate(units):
    print(unit["unitkey"])
    unitkeys[i] = unit["unitkey"]
    dates0[i] = datetime.datetime.strptime(unit["date0"], "%Y-%m-%d").date()
    dates1[i] = datetime.datetime.strptime(unit["date1"], "%Y-%m-%d").date()

pi = Priceindex()
pi0 = pi.interpolate(pd.Series(dates0)).tolist()
pi1 = pi.interpolate(pd.Series(dates1)).tolist()
priceindex1div0 = [j / i for i, j in zip(pi0, pi1)]


# Store new information in the dictionary
for (i, unit) in enumerate(units):
    unit["pi0"] = pi0[i]
    unit["pi1"] = pi1[i]
    unit["priceindex1div0"] = priceindex1div0[i]


## Test return of full price index
pi = Priceindex()
pi.date 