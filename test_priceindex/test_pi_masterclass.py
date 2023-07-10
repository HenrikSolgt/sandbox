import numpy as np
import pandas as pd

from solgt.db.MT_parquet import get_parquet_as_df

from solgt.priceindex.priceindex import Priceindex
from solgt.priceindex.cbi_oslo_class import CBI_Oslo_class
from solgt.priceindex.cbi_cube_api_class import CBI_cube_api_class
import solgt.priceindex.priceindex_utils as pi_utils

kommunenummer_Oslo = 301
kommunenummer_default = kommunenummer_Oslo  # 301 is Oslo


class Priceindex():
    """
    This is a Price index master class: it contains all the price index classes in the solgt package, 
        and will automatically choose the correct one based on the arguments given.
    Internal variables:
    - CBI_cube_api: The CBI Cube API class for Oslo
    - CBI_Oslo: CBI for all of Oslo treated as a single region
    - (More regions to be added as the functionallity is developed)

    NOTE: Currently only Oslo is supported.
    """

    def fetch_priceindexes(self):
        # Fetch Price Indexes
        self.CBI_cube_api = CBI_cube_api_class()
        self.CBI_Oslo = CBI_Oslo_class()
        self.fetched = True

    def __init__(self, do_fetch=True):
        """
        Initialize the Priceindex class. Needs to download all the price index data from the database.
        """
        if do_fetch:
            self.fetch_priceindexes()
        else:
            self.CBI_cube_api = None
            self.CBI_Oslo = None
            self.fetched = False


    def reindex(self, df, t0='fromdate', t1='todate', unitkey='unitkey', kommunenummer='kommunenummer'):
        """
        Reindex price from t0 to t1. Automatically chooses the correct price index class based on the provieded columns in df.
        Input:
            df: DataFrame with the query. Each row is a separate query, and the columns are:
                - 't0': The from dates for the reindexing
                - 't1': The to dates for the reindexing
                A series of optional columns. For each row, the provided column can be empty. If it is empty, the other columns will be used instead.
                    Examples of optional columns:
                    - 'unitkey': The unitkey: This is a unique identifier for each unit, and no other columns will be checked if the unitkey is provided. 
                    - 'kommunenummer': If provided, the kommunenummer will be used to find the correct price index.
        Output:
            The input dataframe df augmented with columns:
            - "dp" column with the reindeces
            - "success": True if the reindexing was successful, False otherwise
            - "msg": Error message if success is False

        This function should decide which price index to use based on the available input data in the input DataFrame df.
        
        Priority list:
        1. unitkey: This is an unique identifier for each unit, and no other columns will be checked if the unitkey is provided.
        2. kommunenummer: If provided, the kommunenummer will be used to find the correct price index.

        It none of the above is provided, the function will return "success" = False and a corresponding error message.
        If for instance a unitkey is provided, but is unsuccessful, no alternativ method will be tried.
        """

        # Add "success" and "msg" columns to the dataframe, if they do not exist
        res = df.copy()
        res = pi_utils.add_success_msg_colums(res)

        # Lookup based on unitkey
        if unitkey in df.columns:
            # Extract the rows with unitkey provided
            df_unitkey = res[res[unitkey].notnull()]
            df_unitkey = self.reindex_by_unitkey(df_unitkey, t0=t0, t1=t1, unitkey=unitkey)  # TODO: Denne resetter index: Det skal den ikke
            res.loc[df_unitkey.index, "dp"] = df_unitkey["dp"]
            # Drop the rows with unitkey provided from the original dataframe, and continue with the rest
            df = df.drop(df_unitkey.index)

        # Lookup based on kommunenummer
        if kommunenummer in df.columns:
            df_kommune = df[df[kommunenummer].notnull()]
            df_kommune = self.reindex_by_kommune(df_kommune, t0=t0, t1=t1, kommunenummer=kommunenummer)
            res.loc[df_kommune.index, "dp"] = df_kommune["dp"]
            # Drop the rows with kommunenummer provided from the original dataframe, and continue with the rest
            df = df.drop(df_kommune.index)

        # The rows not yet treated will be set to "success"=False and "msg"="No price index found".
        res = pi_utils.add_error_msg(res, "No price index found.", df.index)

        return res

    def reindex_by_unitkey(self, df, t0='fromdate', t1='todate', unitkey='unitkey'):
        """
        Reindex price from t0 to t1. Automatically chooses the correct price index class based on the provided columns in df.
        Input:
            df: DataFrame with the query. Each row is a separate query, and the columns are:
            - 't0': The from dates for the reindexing
            - 't1': The to dates for the reindexing
            - 'unitkey': The unitkey: This is a unique identifier for each unit.
        Output:
            The input dataframe df augmented with columns:
            - "dp" column with the reindeces
            - "success": True if the reindexing was successful, False otherwise
            - "msg": Error message if success is False
        """
        
        # Convert t0, t1 and unitkey to the column names used by the CBI_cube_api
        df.rename(columns={t0: "fromdate", t1: "todate", unitkey: "unitkey"}, inplace=True)
        res = self.CBI_cube_api.reindex_by_unitkey(df)  # TODO: Denne resetter index: Det skal den ikke
        res.rename(columns={"fromdate": t0, "todate": t1, "unitkey": unitkey}, inplace=True)

        return res


    def reindex_by_kommune(self, df, t0='fromdate', t1='todate', kommunenummer='kommunenummer'):
        """
        Reindex price from t0 to t1. Automatically chooses the correct price index class based on the provided columns in df.
        Input:
            df: DataFrame with the query. Each row is a separate query, and the columns are:
                - 't0': The from dates for the reindexing
                - 't1': The to dates for the reindexing
                - 'kommunenummer': the kommunenumber to use. E.g 301 is Oslo.
        Output:
            The input dataframe df augmented with columns:
            - "dp" column with the reindeces
            - "success": True if the reindexing was successful, False otherwise
            - "msg": Error message if success is False
        """
        
        # Convert t0, t1 and unitkey to the column names used by the CBI_cube_api
        df.rename(columns={t0: "fromdate", t1: "todate"}, inplace=True)
        res = self.CBI_Oslo.reindex(df)
        res.rename(columns={"fromdate": t0, "todate": t1}, inplace=True)

        return res


    def get_priceindex_by_kommune(self, kommunenummer=kommunenummer_default):
        """
        Returns the price index for the given kommunenummer. All available dates are returned.
        The kommunenummer can be passed as a single integer, or as a Pandas Series of integers. 
            If a Pandas Series is passed, the returned price index will be a Pandas DataFrame.
        """
        res = self.CBI_Oslo.get_priceindex()

        return res

    def get_priceindex_by_unitkey(self, unitkey):
        """
        Returns the price index for a given, single unitkey. All available dates are returned.
        The unitkey can be passed as a single string, or as a Pandas Series of strings. 
           If a Pandas Series is passed, the returned price index will be a Pandas DataFrame.
        """
        res = self.CBI_cube_api.get_priceindex_by_unitkey(unitkey)

        return res
        



t0='fromdate'
t1='todate'
unitkey='unitkey'
kommunenummer='kommunenummer'


PI = Priceindex()
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


df3 = df.copy()

PI.reindex(df)

