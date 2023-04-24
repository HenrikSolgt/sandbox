import json
import datetime
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


b = json.dumps(res)
