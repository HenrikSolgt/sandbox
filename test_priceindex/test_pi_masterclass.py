import datetime
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
        """

        # TODO: This function needs a function for looking up unitkeys, and then decide, based on the location, which price index to use.
        pass

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
        res = self.CBI_cube_api.reindex_by_unitkeys(df)
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

        # Choose sample
        df["kommunenummer"] = np.NaN
        df_s = df.sample(400)
        df_s["kommunenummer"] = kommunenummer_Oslo
        df.loc[df_s.index, :] = df_s

        # Select the correct CBI_class based on the kommunenummer
        df_Oslo = df[df["kommunenummer"] == kommunenummer_Oslo]
        df_Oslo.loc[:, "fromdate"] = datetime.date(2012, 1, 1)
        df_Oslo.loc[:, "todate"] = datetime.date(2015, 1, 1)

        # Convert t0, t1 and unitkey to the column names used by the CBI_cube_api
        df_Oslo.rename(columns={t0: "fromdate", t1: "todate"}, inplace=True)
        res = self.CBI_Oslo.reindex(df_Oslo)
        res.rename(columns={"fromdate": t0, "todate": t1}, inplace=True)



        return res


    def get_priceindex_by_kommune(self, kommunenummer=kommunenummer_default):
        """
        Returns the price index for the given kommunenummer. All available dates are returned.
        The kommunenummer can be passed as a single integer, or as a Pandas Series of integers. 
            If a Pandas Series is passed, the returned price index will be a Pandas DataFrame.
        """
        pass

    def get_priceindex_by_unitkey(self, unitkey):
        """
        Returns the price index for a given, single unitkey. All available dates are returned.
        The unitkey can be passed as a single string, or as a Pandas Series of strings. 
           If a Pandas Series is passed, the returned price index will be a Pandas DataFrame.
        """


from solgt.db.MT_parquet import get_parquet_as_df
df_MT = get_parquet_as_df( "..\..\py\data\MT.parquet")

dates = df_MT["sold_date"]

# Sample some unitkeys
uks = pd.DataFrame()
uks["unitkey"] = df_MT["unitkey"].sample(1000).reset_index(drop=True)
df = uks

PI = Priceindex()
self = PI