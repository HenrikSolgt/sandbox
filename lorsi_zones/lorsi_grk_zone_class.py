# Python packages
import pandas as pd
import numpy as np
from scipy import interpolate
import plotly.graph_objects as go

# Solgt packages
from solgt.priceindex.repeatsales import get_derived_MT_columns, add_derived_MT_columns, get_repeated_idx, get_RS_idx_lines, get_df_ttp_from_RS_idx, create_and_solve_LORSI_OLS_problem
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import smooth_w

# Local packages
from grk_zone_geometry import get_grk_zones_and_neighbors

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
postcode = "postcode"


def default_zone_func(df):
    # Default zone function. Returns a all-zero numpy array of same length as df
    return np.zeros(len(df))

def zone_func_div(df, zone_div):
    # Zone function that returns the values of df integer divided by zone_div
    return df // zone_div

def zone_func_div100(df):
    # Zone function that returns the first two digits of the values of df
    return zone_func_div(df=df, zone_div=100)


def get_LORSI_and_count_for_zones(df, t0, t1):
    """
    Get LORSI and count for all matched transactions in DataFrame df, for the time period [t0, t1).
    All raw transactions from df are used. The filtering on time period [t0, t1) is performed after the transactions have been matched.
    Note that only the zones occuring in df are included in the output.
    Inputs:
        - df: DataFrame with columns "unitkey", "price_inc_debt", "zone", "t"
        - t0: Start time
        - t1: End time
    Returns:
        - LORSI: LORSI for all zones in df, as a Pandas DataFrame
        - count: Number of transactions for all zones in df, as a Pandas DataFrame
        - zones_arr: List of zones, as a Pandas Series
        - t_arr: List of times, as a Pandas Series
    """

    zones_arr = df["zone"].unique()
    zones_arr.sort()

    t_arr = np.arange(t0, t1)
    # Create empty dataframes
    zone_LORSI = pd.DataFrame(index=t_arr, columns=zones_arr)
    zone_counts = pd.DataFrame(index=t_arr, columns=zones_arr).fillna(0)

    for zone_no in zones_arr:
        # Filter the MT data for the current zone
        df_zone = df[df["zone"] == zone_no].reset_index(drop=True)
        print("Zone number: " + str(zone_no) + ". Number of transactions: " + str(len(df_zone)))

        # Get repeated sales index and create df_ttp
        RS_idx = get_repeated_idx(df_zone)
        df_ttp = get_df_ttp_from_RS_idx(df_zone, RS_idx)

        if (len(df_ttp) > 0):
            LORSI_res = create_and_solve_LORSI_OLS_problem(df_ttp)
            LORSI_res = LORSI_res[(LORSI_res["t"] >= t0) & (LORSI_res["t"] < t1)].reset_index(drop=True) # Filters wrongly

            # Split LORSI_res into LORSI and count dataframes
            zone_LORSI[zone_no] = LORSI_res[["t", "pred"]].set_index(["t"]).reindex(t_arr).values
            zone_counts[zone_no] = LORSI_res[["t", "count"]].set_index(["t"]).reindex(t_arr, fill_value=0).values

    # Substitute NaN with 0
    zone_LORSI.fillna(0, inplace=True)
    zone_counts.fillna(0, inplace=True)

    return zone_LORSI, zone_counts


def insert_missing_zones(LORSI, count, new_zones_arr):
    # LORSI and count with missing zones are inserted into LORSI and count, and the new zones are added to zones_arr
    for zone in new_zones_arr:
        if zone not in LORSI.keys():
            LORSI[zone] = np.zeros(len(LORSI))
            count[zone] = np.zeros(len(count)).astype(int)

    # Sort columns
    LORSI = LORSI[sorted(LORSI.columns)]
    count = count[sorted(count.columns)]

    return LORSI, count


class LORSI_grk_zone_class:
    def compute_zones(self, df):
        """
        Computes the zones for the matched transactions in df, using all information that it has available, including the zone information from the original df_MT.
        Takes DataFrame df of matched transactions as input, and returns a column of zones with the same index and length as df
        """ 

        res = df
        res["zone"] = self.zone_func(res[gr_krets])
        res_idx_nan = res[res["zone"].isna()].index

        if self.src_MT is not None:
            src_MT = self.src_MT.copy()
            src_MT["zone"] = self.zone_func(src_MT[gr_krets])
            src_idx_nan = src_MT[src_MT["zone"].isna()].index

            ok_entries = pd.concat([res.drop(res_idx_nan), src_MT.drop(src_idx_nan)])
        else:
            ok_entries = res.drop(res_idx_nan)

        nan_entries = res.loc[res_idx_nan]
        res.loc[res_idx_nan, "zone"] = nan_entries["postcode"].map(ok_entries.groupby("postcode")["zone"].agg(lambda x: x.value_counts().index[0]))

        return res["zone"]
    

    def __init__(self, df_MT=None, date0=None, date1=None, period="monthly", zone_func=default_zone_func):
        """
        Create a LORSI_grk_zone_class object. This object contains the LORSI for each zone, and the count of transactions for each zone. 
        The matched transactions are loaded from df_MT, and only data between date0 and date1 is used.
        The default value of zone_func is a function returning all zeros, producing in a single zone: hence all transactions are treated to belong to the same zone.
        """
        # Original data
        self.src_MT = None  # Will be set later if df_MT is not None
        self.date0 = date0
        self.date1 = date1
        self.period = period
        self.zone_func = zone_func

        if df_MT is not None:
            # Fill in the zone column using the zone_func
            df = df_MT.copy()
            df["zone"] = self.compute_zones(df)

            # If any zones are NaN, remove them (should not be many)
            df = df.dropna(subset=["zone"]).reset_index(drop=True)

            # Add the columns derived from the matched transactions
            df = add_derived_MT_columns(df, period, date0)

            # Convert date0 and date1 to t0 and t1 and get LORSI and count for the zones
            [t0, t1] = convert_date_to_t([date0, date1], period)
            [LORSI, count] = get_LORSI_and_count_for_zones(df, t0, t1) # Create OLS and count for the zones

            [zones_geometry, zones_neighbors] = get_grk_zones_and_neighbors(self.zone_func)
            new_zones_arr = zones_neighbors.index.values
            LORSI, count = insert_missing_zones(LORSI, count, new_zones_arr)

            # Store a reference to the source material
            self.src_MT = df_MT   # Stores a reference to the original df_MT, which will be used for Zone identification later

            # Store the computed material
            self.src_MT_zones = df["zone"]   # A list of zones for each matched transaction in src_MT
            self.LORSI = LORSI.values
            self.count = count.values
            self.zones_arr = new_zones_arr
            self.t_arr = LORSI.index.values
            self.zones_geometry = zones_geometry
            self.zones_neighbors = zones_neighbors.values
        else:
            # Create an empty object
            self.src_MT_zones = None
            self.LORSI = None
            self.count = None
            self.zones_arr = None
            self.t_arr = None
            self.zones_geometry = None
            self.zones_neighbors = None


    def copy(self):
        # Create a copy of the object
        res = LORSI_grk_zone_class()
        # Original data
        res.src_MT = self.src_MT    # Stores a reference to the original df_MT: Hence: NOT a copy
        res.date0 = self.date0
        res.date1 = self.date1
        res.period = self.period
        res.zone_func = self.zone_func

        # Computed data
        res.src_MT_zones = self.src_MT_zones.copy()
        res.LORSI = self.LORSI.copy()
        res.count = self.count.copy()
        res.zones_arr = self.zones_arr.copy()
        res.t_arr = self.t_arr.copy()
        res.zones_neighbors = self.zones_neighbors.copy() # ERROR: Can be NOne

        return res
    

    def get_dates(self):
        # Returns the dates corresponding to the time indices
        # Does not update anything internally
        return convert_t_to_date(self.t_arr, self.period, self.date0)
    

    def get_LORSI_df(self):
        # Returns the LORSI as a dataframe with dates as index and zones as columns
        return pd.DataFrame(self.LORSI, index=self.get_dates(), columns=self.zones_arr)


    def get_count_df(self):
        # Returns the counts as a dataframe with dates as index and zones as columns
        return pd.DataFrame(self.count, index=self.get_dates(), columns=self.zones_arr)
    

    def get_zones_neighbors_df(self):
        # Returns the neighbors as a dataframe with zones as index and columns
        return pd.DataFrame(self.zones_neighbors, index=self.zones_arr, columns=self.zones_arr)


    def convert_to_t_days(self, dates):
        # Converts the dates to time indices
        return convert_date_to_t(dates, 1, self.date0)


    def set_LORSI_to_zero_mean(self):
        # Shifts the LORSI so that every entry has zero mean.
        self.LORSI = self.LORSI - np.mean(self.LORSI, axis=0)


    def interpolate_to_dates(self, new_dates, kind="linear"):
        """
        Returns two dataframes, LORSI_interp and count_interp, with the LORSI and count interpolated to the given dates, with extrapolation.
        The given dates can be arbitrary, and does not have to be within the original time interval. 
        NOTE: The total count is preserved in the process, but estimated at the new dates.
        """

        index_days = self.convert_to_t_days(self.get_dates())
        new_index_days = self.convert_to_t_days(new_dates)

        LORSI_interp = pd.DataFrame(index=new_dates, columns=self.zones_arr)
        count_interp = pd.DataFrame(index=new_dates, columns=self.zones_arr)
        for (i, zone) in enumerate(self.zones_arr):
            LORSI_interp[zone] = interpolate.interp1d(index_days, self.LORSI[:, i], kind=kind, fill_value='extrapolate')(new_index_days)
            count_interp[zone] = interpolate.interp1d(index_days, self.count[:, i], kind=kind, fill_value='extrapolate')(new_index_days)
            # Normalize count so that the total equals the original total, if the total is non-zero
            zone_sum = np.sum(count_interp[zone])
            if zone_sum > 0:
                count_interp[zone] = count_interp[zone] * np.sum(self.count[:, i]) / zone_sum

        return LORSI_interp, count_interp


    def convert_to_period(self, new_period, kind="linear"):
        """
        Converts the LORSI to new period given. date0 is kept the same and used as ref_date if new_period is an integer.
        The information in count is interpolated to create approximate values for the new period. 
        NOTE: The total count is preserved in the process, but estimated at the new dates.
        """

        res = self.copy()

        if self.period != new_period:
            [new_t0, new_t1] = convert_date_to_t([res.date0, res.date1], new_period, res.date0)

            new_t_arr = np.arange(new_t0, new_t1)
            new_dates = convert_t_to_date(new_t_arr, new_period, res.date0)

            new_LORSI, new_count = res.interpolate_to_dates(new_dates, kind=kind)
            res.LORSI = new_LORSI.values
            res.count = new_count.values

            res.t_arr = new_t_arr
            res.period = new_period

        return res


    def filter_LORSI_in_time(self, weights=None, window_size=5):
        # Filters the LORSI values in time using the count as weights, unless weights are explicitly given. Returns the result as a class of the same type.

        if weights is None:
            weights = self.count

        LORSI_w = np.zeros(self.LORSI.shape)
        for (i, _) in enumerate(self.zones_arr):
            LORSI_w[:, i] = smooth_w(self.LORSI[:, i], weights[:, i], window_size // 2)

        res = self.copy()
        res.LORSI = LORSI_w

        return res
    

    def filter_LORSI_by_zone(self):
        # Do a spatial filtering: compute the weighted average of the LORSIs of the zone itself and its neighbors.
        LORSI = self.get_LORSI_df()
        count = self.get_count_df()
        zones_neighbors = self.get_zones_neighbors_df()

        LORSI_w = LORSI.copy()
        count_w = count.copy()

        for zone in LORSI_w.columns:
            neighbors = zones_neighbors[zones_neighbors[zone] == 1].index
            num_of_neighbors = len(neighbors)

            neighbors_LORSI = LORSI[neighbors]
            neighbors_count = count[neighbors]
            
            weighted_sum = neighbors_LORSI.multiply(neighbors_count).sum(axis=1) + LORSI[zone] * (num_of_neighbors + 1) * count[zone]
            count_sum = neighbors_count.sum(axis=1) + (num_of_neighbors + 1) * count[zone]

            LORSI_w[zone] = weighted_sum / count_sum

            # Normalize count so that the total number is representatitive 
            count_sum = count_sum / (2 * num_of_neighbors + 1)
            count_w[zone] = count_sum

            # Insert 0 for nan in case of division by 0
            LORSI_w[zone] = LORSI_w[zone].fillna(0)

        res = self.copy()
        res.LORSI = LORSI_w.values
        res.count = count_w.values

        return res


    def filter_LORSI_by_zone_iterations(self, iterations=2):
        res = self.copy()
        for _ in range(iterations):
            res = res.filter_LORSI_by_zone()

        return res
    

    def evaluate_test_set(self, df_MT_test):
        """
        Compute the LORSI score for the LORSI predictions created by instance self applied to df_MT_test.
        df_MT_test contains the matched transactions, with a time indicator "t". This one is filtered on the period [t0, t1).
        Inputs:
        Computes the score of the LORSI on the test set df_MT_test.
        - df_MT_test: Dataframe with matched transactions. Must contain the following columns: "unitkey", "sold_date", "price_inc_debt"
        """

        df = df_MT_test.copy()

        # Fill in the zone column using the internal compute_zones function
        df["zone"] = self.compute_zones(df)
        # TODO: Fix this: dropna should not be necessary
        df = df.dropna(subset=["zone"]).reset_index(drop=True)

        # Add the derived columns to the dataframe
        df = add_derived_MT_columns(df, self.period, self.date0)

        # Get the index of the repeated sales
        RS_idx_test = get_repeated_idx(df)
        
        df_ddp = pd.DataFrame()
        line0, line1 = get_RS_idx_lines(df, RS_idx_test, [date_col, "y", "zone"])
        df_ddp["sold_date0"] = line0[date_col]
        df_ddp["sold_date1"] = line1[date_col]
        df_ddp["dp"] = line1["y"] - line0["y"]
        df_ddp["zone"] = line1["zone"]

        # Filter away all transactions that are not in the time period covered by the class instance
        df_ddp = df_ddp[(df_ddp["sold_date0"] >= self.date0) & (df_ddp["sold_date1"] < self.date1)].reset_index(drop=True)

        df_ddp["t0"] = self.convert_to_t_days(df_ddp["sold_date0"])
        df_ddp["t1"] = self.convert_to_t_days(df_ddp["sold_date1"])

        # Needs to interpolate the LORSI: May be sampled at random dates
        LORSI = self.get_LORSI_df()

        # Need to create an interpolation function for each zone
        LORSI_index = self.convert_to_t_days(LORSI.index)
        for zone in LORSI.columns:
            f = interpolate.interp1d(LORSI_index, LORSI[zone], fill_value="extrapolate")

            df_ddp_zone = df_ddp[df_ddp["zone"] == zone]
            pred0 = f(df_ddp_zone["t0"])
            pred1 = f(df_ddp_zone["t1"])

            df_ddp.loc[df_ddp_zone.index, "pred0"] = pred0
            df_ddp.loc[df_ddp_zone.index, "pred1"] = pred1
            df_ddp.loc[df_ddp_zone.index, "dp_est"] = pred1 - pred0

        df_ddp["dp_e"] = df_ddp["dp"] - df_ddp["dp_est"]

        return df_ddp


    def score_LORSI(self, df_MT_test):
        # Scores the LORSI on the test set df_MT_test, by calling evaluate_test_set and computing the mean absolute error of the prediction
        df_ddp = self.evaluate_test_set(df_MT_test)
        res = df_ddp["dp_e"].abs().mean()
        return res


    def add_HPF_part_from_LORSI(self, other, window_size=5):
        """
        Does a count-weighted LPF-filtering of self, and adds the HPF part of the LORSI instance other to it.
        other does not need to be sampled at the same dates as self, as a linear interpolation is done.
        Inputs:
            - other: LORSI_class instance. Only the first zone is used. The HPF part of this zone is added to every LORSI zone of self.
            - window_size: int. Size of the window for the LPF filtering.

        NOTE: The information in count does not change, only the LORSI values.
        """
        
        # Convert other to the same period as self
        other = other.convert_to_period(self.period)
        
        # Extract the LORSI and count dataframes
        x = other.get_LORSI_df()
        y = self.get_LORSI_df()
        y_c = self.get_count_df()

        # Create LPF as a dataframe of size y
        LPF_y = pd.DataFrame(index=y.index, columns=y.columns)
        HPF_x = pd.DataFrame(index=y.index, columns=y.columns)

        w_b = window_size // 2
        for zone in y.columns:
            weights = y_c[zone]
            LPF_y[zone] = smooth_w(y[zone], weights, w_b)
            HPF_x[zone] = x[0] - smooth_w(x[0], weights, w_b)

        res = self.copy()
        res.LORSI = LPF_y.values + HPF_x.values

        return res


    def add_scatter(self, fig, desc="", row=1, col=1, zone=0, mode="lines"):
        # Add a scatter plot of the LORSI values
        df = self.get_LORSI_df()
        name = "Period: " + self.period + ", Zone: " + str(zone)
        if desc != "":
            name += ", " + desc
        return fig.add_trace(go.Scatter(x=df.index, y=df[zone], mode=mode, name=name), row=row, col=col)


