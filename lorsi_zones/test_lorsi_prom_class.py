# Python packages
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from plotly.subplots import make_subplots

# Solgt packages
from solgt.db.MT_parquet import get_parquet_as_df, update_MT_parquet_file
from solgt.priceindex.repeatsales import get_derived_MT_columns, add_derived_MT_columns, get_repeated_idx, get_RS_idx_lines, get_df_ttp_from_RS_idx, create_and_solve_LORSI_OLS_problem
from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date
from solgt.timeseries.filter import smooth_w

# Local packages
from lorsi_grk_zone_class import LORSI_grk_zone_class, default_zone_func, zone_func_div100

# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
postcode = "postcode"
prom_code = "PROM"


def get_neighbors_to_PROM_group(N_prom):
    """
    Find all neighbours of a PROM group.
    Returns: 
        neighbor_matrix: A pandas DataFrame with 1 if two zones are neighbors, 0 otherwise. It is also 0 on the diagonal.
    """
    N_arr = np.arange(N_prom)
    d = np.ones(N_prom-1)
    neighbor_matrix = np.diag(d, -1) + np.diag(d, 1)

    res = pd.DataFrame(neighbor_matrix, index=N_arr, columns=N_arr)

    return res


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


default_PROM_bins = [0]
PROM_bins_0_60_90 = [0, 60, 90]


class LORSI_PROM_class:

    def compute_PROM_group(self, df):
        res = df[prom_code].apply(lambda x: np.digitize(x, self.PROM_bins) - 1)  # Group into size groups
        return res


    def __init__(self, df_MT=None, date0=None, date1=None, period="monthly", PROM_bins=default_PROM_bins, zone_func=default_zone_func):
        """
        Create a LORSI_PROM_class object. This object contains the LORSI for each PROM bin, and the count of transactions for each bin. 
        The matched transactions are loaded from df_MT, and only data between date0 and date1 is used.
        The default value of PROM_bins is a single bin, which means that the LORSI is computed for all transactions.
        """
        self.src_MT = df_MT
        self.date0 = date0
        self.date1 = date1
        self.period = period
        self.N_PROM = len(PROM_bins)
        self.PROM_bins = PROM_bins
        self.PROM_groups = np.arange(self.N_PROM)

        LORSI_PROM = []
        PROM_neighbors = None

        if df_MT is not None:
            # Fill in the PROM column using the PROM_bins
            df = df_MT.copy()
            df["PROM_group"] = self.compute_PROM_group(df)

            # Remove those with PROM_group < 0, as these are outside valid PROM range
            df = df[df["PROM_group"] >= 0].reset_index(drop=True)
            
            # If any PROM_group are NaN, remove them (should not be many)
            df = df.dropna(subset=["PROM_group"]).reset_index(drop=True)

            # Now, call LORSI_grk_zone_class to create the sub-classes for each PROM group
            for i in self.PROM_groups:
                LORSI_PROM.append(LORSI_grk_zone_class(df[df["PROM_group"] == i], date0, date1, period, zone_func))

            # Compute the neighbors
            PROM_neighbors = get_neighbors_to_PROM_group(self.N_PROM)

        # Store the results
        self.LORSI_PROM = LORSI_PROM
        self.PROM_neighbors = PROM_neighbors


    def copy(self):
        # Create a copy of the object
        res = LORSI_PROM_class()

        # Copy all attributes
        res.src_MT = self.src_MT    # Stores a reference to the original df_MT: Hence: NOT a copy
        res.date0 = self.date0
        res.date1 = self.date1
        res.period = self.period
        res.N_PROM = self.N_PROM
        res.PROM_bins = self.PROM_bins
        res.PROM_groups = self.PROM_groups
        for i in self.PROM_groups:
            res.LORSI_PROM.append(self.LORSI_PROM[i].copy())

        res.PROM_neighbors = self.PROM_neighbors

        return res
    

    def get_LORSI_dfs(self):
        # Returns a list of LORSI dataframes, one for each PROM group
        LORSI_dfs = []
        for i in self.PROM_groups:
            LORSI_dfs.append(self.LORSI_PROM[i].get_LORSI_df())

        return LORSI_dfs
    

    def get_LORSI_and_count_arr(self):
        # Returns the LORSI values as an array of size (dates, zones, PROM_groups)
        t_arr = self.LORSI_PROM[0].t_arr
        zones_arr = self.LORSI_PROM[0].zones_arr
        N_t = len(t_arr)
        N_zones = len(zones_arr)
        N_PROM = self.N_PROM

        LORSI_arr = np.zeros((N_t, N_zones, N_PROM))
        count_arr = np.zeros((N_t, N_zones, N_PROM))

        for i in range(N_PROM):
            LORSI_arr[:, :, i] = self.LORSI_PROM[i].get_LORSI_df().values
            count_arr[:, :, i] = self.LORSI_PROM[i].get_count_df().values
        
        return LORSI_arr, count_arr, t_arr, zones_arr, self.PROM_groups
    

    def get_count_dfs(self):
        # Returns a list of count dataframes, one for each PROM group
        count_dfs = []
        for i in self.PROM_groups:
            count_dfs.append(self.LORSI_PROM[i].get_count_df())

        return count_dfs


    def set_LORSI_to_zero_mean(self):
        # Shifts the LORSI so that every entry has zero mean.
        for i in range(self.N_PROM):
            self.LORSI_PROM[i].set_LORSI_to_zero_mean()


    def convert_to_period(self, new_period, kind="linear"):
        """
        Converts the LORSIs to the new given period. date0 is kept the same and used as ref_date if new_period is an integer.
        The information in count is interpolated to create approximate values for the new period. 
        NOTE: The total count is preserved in the process, but estimated at the new dates.
        """

        res = self.copy()
        for i in range(self.N_PROM):
            res.LORSI_PROM[i] = res.LORSI_PROM[i].convert_to_period(new_period, kind)
        
        return res



    def filter_LORSI_in_time(self, weights=None, window_size=5):
        # Filters the all sub-class-LORSIs using its count as weights, unless weights are explicitly given.
        res = self.copy()
        for i in res.PROM_groups:
            res.LORSI_PROM[i] = res.LORSI_PROM[i].filter_LORSI_in_time(weights, window_size)

        return res
    

    def filter_LORSI_by_zone(self):
        # Filters the all sub-class-LORSIs using its count as weights, unless weights are explicitly given.
        res = self.copy()
        for i in self.PROM_groups:
            res.LORSI_PROM[i] = res.LORSI_PROM[i].filter_LORSI_by_zone()

        return res
    

    def filter_LORSI_by_zone_iterations(self, iterations=2):
        res = self.copy()
        for i in self.PROM_groups:
            res.LORSI_PROM[i] = res.LORSI_PROM[i].filter_LORSI_by_zone_iterations(iterations=iterations)

        return res
    

    def filter_LORSI_by_PROM(self):
        """
        Performs a spatial filtering across PROM values, using the count as weights.
        """

        LORSI_arr, count_arr, t_arr, zones_arr, PROM_groups = self.get_LORSI_and_count_arr()
        PROM_groups = PROM_groups.astype(int)  # For some reason, this is needed to avoid problems with indexing

        LORSI_arr_w = LORSI_arr.copy()
        count_arr_w = count_arr.copy()

        neighbors_matrix = self.PROM_neighbors

        for (j, _) in enumerate(zones_arr):
            LORSI = pd.DataFrame(LORSI_arr[:, j, :], index=t_arr, columns=PROM_groups)
            count = pd.DataFrame(count_arr[:, j, :], index=t_arr, columns=PROM_groups)

            LORSI_w = LORSI.copy()
            count_w = count.copy()

            for PROM in PROM_groups:
                neighbors = neighbors_matrix[neighbors_matrix[PROM] == 1].index
                num_of_neighbors = len(neighbors)

                neighbors_LORSI = LORSI[neighbors]
                neighbors_count = count[neighbors]

                weighted_sum = neighbors_LORSI.multiply(neighbors_count).sum(axis=1) + LORSI[PROM] * (num_of_neighbors + 1) * count[PROM]
                count_sum = neighbors_count.sum(axis=1) + (num_of_neighbors + 1) * count[PROM]

                LORSI_w[PROM] = weighted_sum / count_sum

                # Normalize count so that the total number is representatitive 
                count_sum = count_sum / (2 * num_of_neighbors + 1)
                count_w[PROM] = count_sum

                # Insert 0 for nan in case of division by 0
                LORSI_w[PROM] = LORSI_w[PROM].fillna(0)

            LORSI_arr_w[:, j, :] = LORSI_w.values
            count_arr_w[:, j, :] = count_w.values

        # Need to copy the filtered results back into the LORSI_PROM objects
        
        res = self.copy()
        for i in PROM_groups:
            res.LORSI_PROM[i].LORSI = LORSI_arr_w[:, :, i]
            res.LORSI_PROM[i].count = count_arr_w[:, :, i]

        return res

        
    def score_LORSI(self, df_MT_test):
        # Returns a list of score infos, one for each PROM group
        df = df_MT_test.copy()

        # Fill in the PROM group column using the internal compute_PROM_group function
        df["PROM_group"] = self.compute_PROM_group(df)
        df = df.dropna(subset=["PROM_group"]).reset_index(drop=True)  # If any PROM_group are NaN, remove them (should not be many)
        
        dp_e_list = pd.DataFrame()
        for i in self.PROM_groups:
            d = self.LORSI_PROM[i].evaluate_test_set(df[df["PROM_group"] == i])
            print(d["dp_e"].abs().mean())
            dp_e_list = pd.concat([dp_e_list, d], axis=0)
            
        res = dp_e_list["dp_e"].abs().mean()

        return res


    def add_HPF_part_from_LORSI(self, other, window_size=5):
        """
        Does a count-weighted LPF-filtering of self, and adds the HPF part of the LORSI instance other to it.
        Class object other is assumed to have the same PROM bins as self.
        """

        res = self.copy()
        for i in self.PROM_groups:
            res.LORSI_PROM[i] = res.LORSI_PROM[i].add_HPF_part_from_LORSI(other.LORSI_PROM[i], window_size)
        
        return res


    def add_scatter(self, fig, desc="", row=1, col=1, zone=0, PROM_ind=0, mode="lines"):
        # Use self.LORSI_PROM to add a scatter plot to fig, by the PROM_ind index
        if (PROM_ind >= 0):
            lower_PROM = str(self.PROM_bins[PROM_ind])
        else:
            lower_PROM = ""

        if (PROM_ind < self.N_PROM-1):
            upper_PROM = str(self.PROM_bins[PROM_ind+1])
        else:
            upper_PROM = ""

        desc = "PROM group " + str(PROM_ind) + " (" + lower_PROM + "-" + upper_PROM + ") " + desc
        fig = self.LORSI_PROM[PROM_ind].add_scatter(fig, desc=desc, row=row, col=col, zone=zone, mode=mode)

        return fig
        



# Remove warning
pd.options.mode.chained_assignment = None  # default='warn'

# Constants
key_col = "unitkey"
date_col = "sold_date"
price_col = "price_inc_debt"
gr_krets = "grunnkrets_id"
postcode = "postcode"

"""
MAIN PROGRAM
"""


# Define time period
date0 = datetime.date(2012, 1, 1)
date1 = datetime.date(2022, 1, 1)

period = "quarterly"


df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")
df_MT[date_col] = df_MT[date_col].apply(lambda x: datetime.date(x.year, x.month, x.day))
df_MT[postcode] = df_MT[postcode].astype(int)


train_MT, test_MT = train_test_split_rep_sales(df_MT, test_size=0.2)

all_raw = LORSI_PROM_class(train_MT, date0, date1, "weekly", default_PROM_bins, default_zone_func)
all_raw_f = all_raw.filter_LORSI_in_time(window_size=5)
PROM_raw = LORSI_PROM_class(train_MT, date0, date1, "weekly", PROM_bins_0_60_90, default_zone_func)
PROM_f_prom = PROM_raw.filter_LORSI_by_PROM()
PROM_f_prom_time = PROM_f_prom.filter_LORSI_in_time(window_size=5)
# PROM_f_prom = PROM_raw.filter_LORSI_by_PROM()
# PROM_f_prom_time = PROM_f_prom.filter_LORSI_in_time(window_size=5)
# PROM_f_time_prom = PROM_raw.filter_LORSI_in_time(window_size=5).filter_LORSI_by_PROM()


# PROM and zone splitted
prom_zone = LORSI_PROM_class(train_MT, date0, date1, "monthly", PROM_bins_0_60_90, zone_func_div100)
# Filter by PROM
prom_zone_f_prom0 = prom_zone.filter_LORSI_by_PROM()
prom_zone_f_prom1 = prom_zone_f_prom0.filter_LORSI_by_PROM()
prom_zone_f_prom = prom_zone_f_prom1.filter_LORSI_by_PROM()
# Filter by zone
prom_zone_f_prom_zone = prom_zone_f_prom.filter_LORSI_by_zone_iterations(iterations=3)
# Filter in time
prom_zone_f_prom_zone_f_time = prom_zone_f_prom_zone.filter_LORSI_in_time(window_size=3)
# Resample to weekly
prom_zone_f_prom_zone_w = prom_zone_f_prom_zone.convert_to_period("weekly")

# Add HPF and LPF parts
prom_zone_prom_comb = prom_zone_f_prom_zone_w.add_HPF_part_from_LORSI(PROM_f_prom_time, window_size=13)

all_raw.set_LORSI_to_zero_mean()
all_raw_f.set_LORSI_to_zero_mean()
PROM_raw.set_LORSI_to_zero_mean()
PROM_f_prom.set_LORSI_to_zero_mean()
PROM_f_prom_time.set_LORSI_to_zero_mean()
prom_zone.set_LORSI_to_zero_mean()
prom_zone_f_prom.set_LORSI_to_zero_mean()
prom_zone_f_prom_zone.set_LORSI_to_zero_mean()
prom_zone_f_prom_zone_f_time.set_LORSI_to_zero_mean()
prom_zone_f_prom_zone_w.set_LORSI_to_zero_mean()
prom_zone_prom_comb.set_LORSI_to_zero_mean()



# Plot the LORSI for all PROM groups
fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
# fig = all_raw.add_scatter(fig, desc="All", row=1, col=1, zone=0, PROM_ind=0, mode="lines")
# fig = all_raw_f.add_scatter(fig, desc="All, filtered", row=1, col=1, zone=0, PROM_ind=0, mode="lines")
for i in prom_zone.PROM_groups:
    zone = 6
    # fig = PROM_raw.add_scatter(fig, desc="", row=1, col=1, zone=0, PROM_ind=i, mode="lines")
    fig = PROM_f_prom.add_scatter(fig, desc="Filtered in PROM", row=1, col=1, zone=0, PROM_ind=i, mode="lines")
    fig = PROM_f_prom_time.add_scatter(fig, desc="Filtered in PROM and time", row=1, col=1, zone=0, PROM_ind=i, mode="lines")
    
    # fig = prom_zone.add_scatter(fig, desc="", row=1, col=1, zone=zone, PROM_ind=i, mode="lines")
    # fig = prom_zone_f_prom.add_scatter(fig, desc="Filtered by PROM", row=1, col=1, zone=zone, PROM_ind=i, mode="lines")
    fig = prom_zone_f_prom_zone_w.add_scatter(fig, desc="Filtered in PROM and zone", row=1, col=1, zone=zone, PROM_ind=i, mode="lines")
    fig = prom_zone_prom_comb.add_scatter(fig, desc="Combined", row=1, col=1, zone=zone, PROM_ind=i, mode="lines")

fig.show()


## Scoring the LORSIs

# all_raw.score_LORSI(test_MT)
all_raw_f.score_LORSI(test_MT)
prom_zone.score_LORSI(test_MT)
prom_zone_f_prom0.score_LORSI(test_MT)
prom_zone_f_prom1.score_LORSI(test_MT)
prom_zone_f_prom.score_LORSI(test_MT)
prom_zone_f_prom_zone.score_LORSI(test_MT)
prom_zone_f_prom_zone_w.score_LORSI(test_MT)
prom_zone_f_prom_zone_f_time.score_LORSI(test_MT)
prom_zone_prom_comb.score_LORSI(test_MT)


df = test_MT.copy()
df["PROM_group"] = prom_zone_f_prom_zone_w.compute_PROM_group(df)
prom_zone_prom_comb.LORSI_PROM[0].score_LORSI(df[df["PROM_group"] == 0])
prom_zone_prom_comb.LORSI_PROM[1].score_LORSI(df[df["PROM_group"] == 0])
prom_zone_prom_comb.LORSI_PROM[2].score_LORSI(df[df["PROM_group"] == 0])