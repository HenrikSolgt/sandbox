# Python packages
import datetime
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import interpolate
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df, update_MT_parquet_file
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
prom_code = "PROM"

default_PROM_bins = [0]
PROM_bins_0_60_90 = [0, 60, 90]
PROM_bins_s10 = [10*c for c in range(1, 15, 1)]


def get_window(window_size=5, window_type="flat", beta=14):
    """
    Returns a numpy array of size window_size with the specified window_type.
    """
    match window_type:
        case "hanning":
            window = np.hanning(window_size)
        case "hamming":
            window = np.hamming(window_size)
        case "bartlett":
            window = np.bartlett(window_size)
        case "blackman":
            window = np.blackman(window_size)
        case "kaiser":
            window = np.kaiser(window_size, beta)
        case _: # Use flat window as default
            window = np.ones(window_size)

    window = window / np.sum(window)

    return window



def train_test_split_rep_sales(df_MT, test_size=0.2):
    # Split df_MT into train and test sets, but only the repeated sales
    derived_MT_cols = get_derived_MT_columns(df_MT)
    RS_idx = get_repeated_idx(derived_MT_cols)
    _, RS_idx_test = train_test_split(RS_idx, test_size=test_size)

    # Let train_I be I0 and I1 in RS_idx_train
    # I_train = pd.concat([RS_idx_train["I0"], RS_idx_train["I1"]]).drop_duplicates().sort_values()
    I_test = pd.concat([RS_idx_test["I0"], RS_idx_test["I1"]]).drop_duplicates().sort_values()

    # Extract indices
    df_MT_test = df_MT.loc[I_test].reset_index(drop=True)
    df_MT_train = df_MT.drop(I_test).reset_index(drop=True)

    return df_MT_train, df_MT_test


def default_zone_func(df):
    # Default zone function. Returns a all-zero numpy array of same length as df
    return np.zeros(len(df)).astype(int)


def zone_func_div(df, zone_div):
    # Zone function that returns the values of df integer divided by zone_div
    return df // zone_div


def zone_func_div100(df):
    # Zone function that returns the first two digits of the values of df
    return zone_func_div(df=df, zone_div=100)


def zone_func_div1000(df):
    # Zone function that returns the first two digits of the values of df
    return zone_func_div(df=df, zone_div=1000)



class LORSI_cube_class:
    """
    Class for computing and storing LORSI cubes. 
    It is a multi-dimensional numpy array with dimensions (time, zone, PROM), where PROM is the PROM size group.
    Passed variables:
    - df_scr: DataFrame with matched transactions, used as a source for computing the LORSI cube
    - date0: Start date for the LORSI cube
    - date1: End date for the LORSI cube
    - period: Period for the LORSI cube (weekly, monthly, quarterly, or an integer for a custom period length in days)
    - zone_func: Function used to compute the zones for the matched transactions. Default is default_zone_func, which returns a all-zero numpy array of same length as df
    - PROM_bins: List of PROM bin levels. Default is [0], which means that all PROMs are included in the same bin

    Internal variables:
    - LORSI: LORSI numpy array with dimensions (time, zone, PROM)
    - count: Number of transactions numpy array with dimensions (time, zone, PROM)
    - t_arr: List of times, in the t-format. Can be used to convert to date format using date0, date1 and 
    - zones_arr: List of zones
    - PROM_bins: List of PROM bins
    - N_t: Number of time periods
    - N_z: Number of zones
    - N_p: Number of PROM bins
    """

    def compute_zones(self, df):
        """
        Computes the zones for the matched transactions in df, using all information that it has available, including the zone information from the original df_MT.
        Takes DataFrame df of matched transactions as input, and returns a column of zones with the same index and length as df
        """ 

        res = df
        res["zone"] = self.zone_func(res[gr_krets])
        res_idx_nan = res[res["zone"].isna()].index

        if self.computed:
            src_MT = self.src_MT.copy()
            src_MT["zone"] = self.zone_func(src_MT[gr_krets])
            src_idx_nan = src_MT[src_MT["zone"].isna()].index

            ok_entries = pd.concat([res.drop(res_idx_nan), src_MT.drop(src_idx_nan)])
        else:
            ok_entries = res.drop(res_idx_nan)

        nan_entries = res.loc[res_idx_nan]
        res.loc[res_idx_nan, "zone"] = nan_entries["postcode"].map(ok_entries.groupby("postcode")["zone"].agg(lambda x: x.value_counts().index[0]))

        return res["zone"]
    

    def compute_PROM_groups(self, df):
        """
        Computes the PROM groups for the matched transactions in df, using all information that it has available, including the PROM group information from the original df_MT.
        """
        res = df[prom_code].apply(lambda x: np.digitize(x, self.PROM_bins) - 1)  # Group into size groups
        return res
    

    def __init__(self, df_MT=None, date0=None, date1=None, period="monthly", zone_func=default_zone_func, PROM_bins=default_PROM_bins, do_compute=True):
        """
        Initializes the LORSI_cube_class object, but does not compute the LORSI cube.
        """
        self.src_MT = df_MT  # Will be set later if df_MT is not None
        self.date0 = date0
        self.date1 = date1
        self.period = period
        self.zone_func = zone_func
        self.PROM_bins = PROM_bins
        self.computed = False  # Set to False until the LORSI cube has been computed

        if do_compute:
            self.compute()
        else:
            # Set all other variables to None
            self.LORSI = None
            self.count = None
            self.t_arr = None
            self.zones_arr = None
            self.zones_geom = None
            self.zones_dist = None
            self.PROM_arr = None
            self.N_t = None
            self.N_z = None
            self.N_p = None


    def copy(self):
        # Creates a copy of itself
        res = LORSI_cube_class(self.src_MT, self.date0, self.date1, self.period, self.zone_func, self.PROM_bins, do_compute=False)

        # Computed data
        if self.computed:
            res.computed = True
            res.LORSI = self.LORSI.copy()
            res.count = self.count.copy()
            res.t_arr = self.t_arr.copy()
            res.zones_arr = self.zones_arr.copy()
            res.zones_geom = self.zones_geom.copy()
            res.zones_dist = self.zones_dist.copy()
            res.PROM_arr = self.PROM_arr.copy()
            res.N_t = self.N_t
            res.N_z = self.N_z
            res.N_p = self.N_p

        return res
    
    
    def compute_LORSI_and_count(self):
        """ 
        Computes the LORSI and count cubes, using all repeated sales in src_MT, split into the bins and groups as described by self.zones_arr and self.PROM_arr    
        """
        # Set up a DataFrame with the matched transactions, with the zone and PROM_group columns added, and additional columns needed by the OLS algorithm
        df = self.src_MT.copy()
        df["zone"] = self.compute_zones(df)
        df["PROM_group"] = self.compute_PROM_groups(df)

        # If there are any zones of PROM_group that are NaN, then remove them (should not be many)
        df = df.dropna(subset=["zone", "PROM_group"])

        # Convert to integers
        df["zone"] = df["zone"].astype(int)
        df["PROM_group"] = df["PROM_group"].astype(int)

        # Add the columns derived from the matched transactions
        df = add_derived_MT_columns(df, self.period, self.date0)

        # Initialize LORSI and count cubes
        LORSI = np.zeros((self.N_t, self.N_z, self.N_p))
        count = np.zeros((self.N_t, self.N_z, self.N_p))

        # Iterate all zones and PROM groups, and compute the LORSI and count
        for (i, zone) in enumerate(self.zones_arr):
            for (j, PROM_group) in enumerate(self.PROM_arr):
                df_sub = df[(df["zone"] == zone) & (df["PROM_group"] == PROM_group)].reset_index(drop=True)
                print("Zone: " + str(zone) + ", PROM group: " + str(PROM_group) + ". Transactions: " + str(len(df_sub)))

                # Get repeated sales index and create df_ttp
                RS_idx = get_repeated_idx(df_sub)
                df_ttp = get_df_ttp_from_RS_idx(df_sub, RS_idx)

                # Compute LORSI and count if the number of repeated sales is greater than 0
                if (len(df_ttp) > 0):
                    LORSI_res = create_and_solve_LORSI_OLS_problem(df_ttp)
                    LORSI_res = LORSI_res[LORSI_res["t"].isin(self.t_arr)].reset_index(drop=True)

                    # Split LORSI_res into LORSI and count dataframes, and reindex to also include the missing t values
                    LORSI[:, i, j] = LORSI_res[["t", "pred"]].set_index(["t"]).reindex(self.t_arr, fill_value=0)["pred"].values
                    count[:, i, j] = LORSI_res[["t", "count"]].set_index(["t"]).reindex(self.t_arr, fill_value=0)["count"].values

        # Fill LORSI and count Nans to zero
        LORSI = np.nan_to_num(LORSI)
        count = np.nan_to_num(count)

        # Store the results
        self.LORSI = LORSI
        self.count = count


    def compute(self):
        """
        Computes the LORSI and count, and sets self.computed to True. This is the main computation function.
        """
        # Convert date0 and date1 to t0 and t1
        [t0, t1] = convert_date_to_t([self.date0, self.date1], self.period)

        # Compute the t_arr, zones_arr and PROM_arr
        self.t_arr = np.arange(t0, t1)
        [zones_geom, zones_dist] = get_grk_zones_and_neighbors(self.zone_func)
        self.zones_arr = zones_dist.index.values
        self.zones_dist = zones_dist.astype(int).values
        self.zones_geom = zones_geom["geometry"].values
        self.PROM_arr = np.arange(len(self.PROM_bins))

        # Compute the lenghts of the arrays
        self.N_t = len(self.t_arr)
        self.N_z = len(self.zones_arr)
        self.N_p = len(self.PROM_arr)

        # Compute LORSI and count for all zones and PROM groups
        self.compute_LORSI_and_count()
        self.set_LORSI_to_zero_mean()  # Purely for visualization purposes

        # Set computed flag to True
        self.computed = True
        

    def get_dates(self):
        # Returns the dates corresponding to the time indices
        return convert_t_to_date(self.t_arr, self.period, self.date0)
    
    
    def convert_to_t_days(self, dates):
        # Converts the dates to time indices
        return convert_date_to_t(dates, 1, self.date0)


    def set_LORSI_to_zero_mean(self):
        # Shifts the LORSI so that every entry has zero mean. This is purely for visualization purposes.
        self.LORSI = self.LORSI - np.mean(self.LORSI, axis=0)


    def get_LORSI_PROM_df(self, zone=0):
        # Returns the LORSI for the given zone as a dataframe with dates as index and PROM as columns
        return pd.DataFrame(self.LORSI[:, zone, :], index=self.get_dates(), columns=self.PROM_arr)


    def get_count_PROM_df(self, zone=0):
        # Returns the LORSI for the given zone as a dataframe with dates as index and PROM as columns
        return pd.DataFrame(self.count[:, zone, :], index=self.get_dates(), columns=self.PROM_arr)
    

    def get_LORSI_zone_df(self, PROM_group=0):
        # Returns the LORSI for the given zone as a dataframe with dates as index and zones_arr as columns
        return pd.DataFrame(self.LORSI[:, :, PROM_group], index=self.get_dates(), columns=self.zones_arr)


    def get_count_zone_df(self, PROM_group=0):
        # Returns the LORSI for the given zone as a dataframe with dates as index and zones_arr as columns
        return pd.DataFrame(self.count[:, :, PROM_group], index=self.get_dates(), columns=self.zones_arr)


    def interpolate_to_dates(self, new_dates, kind="linear"):
        """
        Returns two numpy arrays of the same size as self.LORSI and self.count. Also returns zones_arr and PROM_arr.
        The given dates can be arbitrary, and does not have to be within the original time interval. 
        NOTE: The total count is preserved in the process, but estimated at the new dates.
        """

        index_days = self.convert_to_t_days(self.get_dates())
        new_index_days = self.convert_to_t_days(new_dates)

        new_N_t = len(new_dates)
        LORSI_interp = np.zeros((new_N_t, self.N_z, self.N_p))
        count_interp = np.zeros((new_N_t, self.N_z, self.N_p))

        for (i, _) in enumerate(self.zones_arr):
            for (j, _) in enumerate(self.PROM_arr):
                LORSI_interp[:, i, j] = interpolate.interp1d(index_days, self.LORSI[:, i, j], kind=kind, fill_value='extrapolate')(new_index_days)
                count_interp[:, i, j] = interpolate.interp1d(index_days, self.count[:, i, j], kind=kind, fill_value='extrapolate')(new_index_days)
                # Normalize count so that the total equals the original total, if the total is non-zero
                zone_prom_sum = np.sum(count_interp[:, i, j])
                if zone_prom_sum > 0:
                    count_interp[:, i, j] = count_interp[:, i, j] * np.sum(self.count[:, i, j]) / zone_prom_sum

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

            res.LORSI, res.count = res.interpolate_to_dates(new_dates, kind=kind)

            res.t_arr = new_t_arr
            res.N_t = len(res.t_arr)
            res.period = new_period

            res.set_LORSI_to_zero_mean()

        return res


    def filter_in_time(self, weights=None, window_size=5, window_type="flat"):
        # Filters the LORSI values in time using the count as weights, unless weights are explicitly given. 
        # Window_type can be "flat", "hanning", "hamming", "bartlett", "blackman", "kaiser". Default is flat. If kaiser, the beta parameter is set to 14.
        # Returns the result as a class of the same type.

        if weights is None:
            weights = self.count
            
        window = get_window(window_size, window_type)

        LORSI_w = np.zeros(self.LORSI.shape)
        count_w = np.zeros(self.LORSI.shape)
        for (i, _) in enumerate(self.zones_arr):
            for (j, _) in enumerate(self.PROM_arr):
                LORSI_w[:, i, j] = np.convolve(self.LORSI[:, i, j] * weights[:, i, j], window, mode="same") / np.convolve(weights[:, i, j], window, mode="same")
                count_w[:, i, j] = np.convolve(self.count[:, i, j], window, mode="same")

        res = self.copy()
        res.LORSI = LORSI_w
        res.count = count_w

        res.set_LORSI_to_zero_mean()

        return res


    def filter_by_zone(self, w_L=2):
        """
        This function filters the LORSI spatially using zone neighbors. w_L is the (maximum) distance to consider.
        Note that the number of neighbours has to be taken into account, as the volume weighting can be biased when the number of neighbours is high
        """

        LORSI = self.LORSI
        count = self.count
        dist = self.zones_dist
        zones_arr = self.zones_arr

        LORSI_w = np.zeros(LORSI.shape)
        count_w = np.zeros(count.shape)

        # Create zone filtering window
        window = np.zeros(w_L + 1)
        for d in range(w_L+1):
            avg_d = (dist == d).sum(axis=0).mean()
            window[d] = 1 / avg_d
        window = window / np.sum(window)

        # Maximum distance to consider. Is set to the minimum of w_L and the maximum distance between two zones
        d_max = min([w_L, dist.max()])

        for (i, zone) in enumerate(zones_arr):
            d_LORSI = np.zeros([zone_LORSI.N_t, d_max+1])
            d_count = np.zeros([zone_LORSI.N_t, d_max+1])

            # Extract the LORSI and count of the actual zone
            d_LORSI[:, 0] = LORSI[:, i, 0]
            d_count[:, 0] = count[:, i, 0]

            # Iterate all d_max closest neighbors
            for d in range(1, d_max+1):
                # Get index of the nearest neighbors (that is: distance = 1)
                neighbor_idx = np.where(dist[i, :] == d)[0]

                # Extract the LORSI and count of the neighbors
                LORSI_neighbors = LORSI[:, neighbor_idx, 0]
                count_neighbors = count[:, neighbor_idx, 0]

                # Compute the count sum of the neighbors
                d_sum = np.sum(count_neighbors, axis=1)

                # Store the count of the neighbors
                    # Can do division by len(neighbor_idx) is to preserve the count scale. Can be removed
                # d_count[:, d] = d_sum / len(neighbor_idx) 
                d_count[:, d] = d_sum

                # Handle division by a zero: Set the relevant d_sum element to 1 to avoid division by zero
                d_sum_is_zero = (d_sum == 0)
                d_sum[d_sum_is_zero] = 1

                # Compute the weighted average LORSI
                d_LORSI[:, d] = np.sum(LORSI_neighbors * count_neighbors, axis=1) / d_sum

            # Compute the weighted average count sum
            count_w_sum = np.sum(d_count * window, axis=1).reshape(-1, 1)
            count_w[:, i] = count_w_sum

            # Handle division by a zero: Set the relevant count_w_sum element to 1 to avoid division by zero
            count_w_sum_is_zero = (count_w_sum == 0)
            count_w_sum[count_w_sum_is_zero] = 1

            # Compute the weighted average LORSI
            LORSI_w[:, i] = np.sum(d_LORSI * d_count * window, axis=1).reshape(-1, 1) / count_w_sum

        # Can scale the count to the original scale
        # count_w = count_w * self.count.sum() / count_w.sum()

        res = self.copy()
        res.LORSI = LORSI_w
        res.count = count_w

        res.set_LORSI_to_zero_mean()

        return res


    def filter_by_PROM(self, window_size=3, window_type="flat"):
        # Filter the LORSI by PROM, where a get_window function is used to create a window

        weights = self.count  # Can made more flexible later if needed
            
        window = get_window(window_size, window_type)

        LORSI_w = np.zeros(self.LORSI.shape)
        count_w = np.zeros(self.LORSI.shape)
        for (i, _) in enumerate(self.zones_arr):
            for (j, _) in enumerate(self.t_arr):
                LORSI_w[j, i, :] = np.convolve(self.LORSI[j, i, :] * weights[j, i, :], window, mode="same") / np.convolve(weights[j, i, :], window, mode="same")
                count_w[j, i, :] = np.convolve(self.count[j, i, :], window, mode="same")

        res = self.copy()
        res.LORSI = LORSI_w
        res.count = count_w

        res.set_LORSI_to_zero_mean()

        return res
    

    def evaluate_test_set(self, df_MT_test):
        """
        This function evaluates the test set using the LORSI model
        """
        df = df_MT_test.copy()

        # Fill in the zone column using the internal compute_zones function
        df["zone"] = self.compute_zones(df)
        df["PROM_group"] = self.compute_PROM_groups(df)

        # If there are any zones of PROM_group that are NaN, then remove them (should not be many)
        df = df.dropna(subset=["zone", "PROM_group"])

        # Convert to integers
        df["zone"] = df["zone"].astype(int)
        df["PROM_group"] = df["PROM_group"].astype(int)

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
        LORSI = self.LORSI
        zones_arr = self.zones_arr
        PROM_arr = self.PROM_arr

        # Need to create an interpolation function for each zone and PROM group
        LORSI_index = self.convert_to_t_days(self.get_dates())
        for (i, zone) in enumerate(zones_arr):
            for (j, PROM_group) in enumerate(PROM_arr):
                f = interpolate.interp1d(LORSI_index, LORSI[:, i, j], fill_value="extrapolate")

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
        print("LORSI score: ", str(res))
        return res
    

    def add_scatter(self, fig, desc="", row=1, col=1, zone=0, PROM=0, mode="lines"):
        df = self.get_LORSI_PROM_df(zone=zone)
        name = "Period: " + self.period + ", Zone: " + str(zone) + ", PROM: " + str(PROM)
        if desc != "":
            name += ", " + desc
        return fig.add_trace(go.Scatter(x=df.index, y=df[PROM], mode=mode, name=name), row=row, col=col)



# Define time period
date0 = datetime.date(2012, 1, 1)
date1 = datetime.date(2022, 1, 1)

period = "monthly"

df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[postcode] = df_MT[postcode].astype(int)


train_MT, test_MT = train_test_split_rep_sales(df_MT, test_size=0.2)


# # Uniform spacing of 10
# PROM_bins_s10 = np.arange(0, 200, 10)
all_LORSI = LORSI_cube_class(train_MT, date0, date1, period)
zone_LORSI = LORSI_cube_class(train_MT, date0, date1, period, zone_func=zone_func_div100)
PROM_LORSI = LORSI_cube_class(train_MT, date0, date1, period, PROM_bins=PROM_bins_0_60_90)

split = LORSI_cube_class(train_MT, date0, date1, period, zone_func=zone_func_div1000, PROM_bins=PROM_bins_0_60_90)
split_z = split.filter_by_zone(w_L=2)
split_z_p = split_z.filter_by_PROM(window_size=3)
split_z_p_f = split_z_p.filter_in_time(window_size=3)

zone_LORSI_z = zone_LORSI.filter_by_zone(w_L=2)
zone_LORSI_z_f = zone_LORSI_z.filter_in_time(window_size=3, window_type="flat")
zone_LORSI_z_f2 = zone_LORSI_z.filter_in_time(window_size=7, window_type="hamming")

PROM_LORSI_p = PROM_LORSI.filter_by_PROM(window_size=3)
PROM_LORSI_f = PROM_LORSI_p.filter_in_time(window_size=3, window_type="flat")

# Plot
fig = make_subplots(rows=1, cols=1)
# Add LORSI_w
for (i, zone) in enumerate(zone_LORSI.zones_arr):
    fig = zone_LORSI.add_scatter(fig, desc="All, Monthly", row=1, col=1, zone=i, PROM=0, mode="lines")    

fig.show()



print("Score: All")
res = all_LORSI.score_LORSI(test_MT)
res = zone_LORSI.score_LORSI(test_MT)
res = zone_LORSI_z.score_LORSI(test_MT)
res = zone_LORSI_z_f.score_LORSI(test_MT)
res = zone_LORSI_z_f2.score_LORSI(test_MT)
res = PROM_LORSI.score_LORSI(test_MT)
res = PROM_LORSI_p.score_LORSI(test_MT)
res = PROM_LORSI_f.score_LORSI(test_MT)

res = split.score_LORSI(test_MT)
res = split_z.score_LORSI(test_MT)
res = split_z_p.score_LORSI(test_MT)
res = split_z_p_f.score_LORSI(test_MT)
