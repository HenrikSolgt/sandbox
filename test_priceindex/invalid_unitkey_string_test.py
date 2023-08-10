import datetime
import pandas as pd
from solgt.priceindex.priceindex import Priceindex
from dateutil import parser

# Create instance of the CBI Masterclass
CBI_masterclass = Priceindex(return_msg_col=True, print_messages=True)


unitkey = "0 124 145 15"

res_df = CBI_masterclass.get_priceindex_by_unitkey(
    unitkey
)  # TODO: Fails if unitkey is a string with invalid unitkey. Must handle this case.
