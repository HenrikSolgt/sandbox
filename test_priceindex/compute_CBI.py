# Python packages
import datetime
import numpy as np
import pandas as pd

# Solgt packages
from solgt.timeseries.filter import smooth_w
import solgt.priceindex.hedonic as hmi
import solgt.priceindex.repeatsales as rsi
from added_data import get_added_data

t_switch1_diff = 28  # Four weeks
t_switch0_diff = 28 * 2  # Eight weeks
AD_diff_days = 365  # Only use data from last year for the HMI

# Hard coded dates
date0 = datetime.date(2000, 1, 4)
HMI1_start_date = datetime.date(2008, 1, 1)
RSI_start_date = datetime.date(
    2010, 1, 1
)  # This is where RSI_weekly starts getting sufficiently populated
HMI1_stop_date = datetime.date(
    2011, 1, 1
)  # A transition period between HMI1 and RSI stops here

# Dynamic date
todaydate = datetime.date.today()


def get_CBI_HMII_monthly(df_MT):
    return hmi.get_HMI(df_MT, HMI1_start_date, todaydate, "monthly")


def get_CBI_RSI_weekly(df_MT):
    return rsi.get_RSI(df_MT, date0, todaydate, "weekly")


def get_CBI_HMI_AD_weekly():
    """
    This function computes the HMI from added data and returns a weekly HMI based on that data.
    """

    # Get (locally stored) added data
    path_AD = "C:/Code/py/data/dataprocessing/added_data/"
    df_AD = get_added_data(path_AD)

    # Filter HMI data and create monthly and weekly HMI and save as a figure
    t1 = datetime.datetime.now().astimezone()
    t0 = datetime.datetime.now().astimezone() - datetime.timedelta(days=AD_diff_days)
    df_AD = df_AD[(df_AD["sold_date"] > t0) & (df_AD["sold_date"] <= t1)]

    # Create date categorical variable
    t1 = min(df_AD["sold_date"])
    days = (df_AD["sold_date"] - t1).apply(lambda x: x.days)
    n_days = 7
    df_AD["date_cat"] = (days / n_days).astype(int)

    # Declare all categorical and numerical columns, as well as the target column
    categorical_columns = ["date_cat", "buildyear_cat", "area_id", "size_cat"]
    numerical_columns = ["PROM"]
    y_col = "price_inc_debt"

    # Create date vector
    date_vec = np.sort(df_AD["date_cat"].unique())

    # Compute HMI model
    i_res = hmi.compute_model_in_period(
        df_AD, date_vec, y_col, categorical_columns, numerical_columns
    )
    i_res = i_res.sort_values("date_cat")

    # Normalize from price_pred
    i_res["price"] = i_res["price_pred"] / i_res["price_pred"].mean()

    # Create date column
    d = i_res["date_cat"].apply(lambda x: datetime.timedelta(days=x * 7)) + t1
    i_res["date"] = d.apply(lambda x: x.date())

    # Keep only relevant columns
    HMI_AD_weekly = i_res[["date", "price", "count"]]

    return HMI_AD_weekly


def create_CBI_from_HMI_RSI_HMI_AD(
    HMI_monthly, RSI_weekly, HMI_AD_weekly, RSI_stop_date=None
):
    """
    Stitches together a CBI price index from three different price indices, using the following logic:
    1. Use HMI_monthly for the period before RSI_weekly is sufficiently populated
    2. Use RSI_weekly for the main period, where it is sufficiently populated
    3. Use HMI_AD_weekly, which is a HMI computed from Added Data, for the period after RSI_weekly is no longer sufficiently populated
    Returns: CBI: a DataFrame with date and price columns, as well as additional information on how the index was created
    """

    # Create start and stop dates for the different indices
    CBI_start_date = HMI_monthly["date"].min()

    # If RSI_stop_date is not provided, the use the "old" rule, using a fixed number of days
    if RSI_stop_date is None:
        RSI_stop_date = todaydate - datetime.timedelta(days=t_switch1_diff)

    t_switch1 = RSI_stop_date
    t_switch0 = RSI_stop_date - datetime.timedelta(days=t_switch0_diff)

    # Filter out data before RSI_start_date, since it is too sparse for the smooth_w to work
    RSI_weekly = RSI_weekly[
        (RSI_weekly["date"] >= RSI_start_date) & (RSI_weekly["date"] <= RSI_stop_date)
    ]

    # Create a new date series for CBI, starting on the first Monday after the first date in RSI_weekly
    dummy = (
        CBI_start_date - datetime.date(2000, 1, 3)
    ).days % 7  # The given date is a Monday
    if dummy > 0:
        CBI_start_date = CBI_start_date + datetime.timedelta(days=7 - dummy)
    CBI_date_stop = HMI_AD_weekly["date"].max()
    CBI = pd.DataFrame()
    CBI["date"] = [
        CBI_start_date + datetime.timedelta(days=7 * x)
        for x in range(0, (CBI_date_stop - CBI_start_date).days // 7 + 1)
    ]

    # Create resampled dataseries for HMI an
    # d RSI, based on the CBI dates
    CBI["HMI1"] = np.interp(
        CBI["date"].apply(lambda x: (x - CBI_start_date).days),
        HMI_monthly["date"].apply(lambda x: (x - CBI_start_date).days),
        HMI_monthly["price"],
    )
    CBI["RSI"] = np.interp(
        CBI["date"].apply(lambda x: (x - CBI_start_date).days),
        RSI_weekly["date"].apply(lambda x: (x - CBI_start_date).days),
        RSI_weekly["price"],
    )
    CBI["HMI2"] = np.interp(
        CBI["date"].apply(lambda x: (x - CBI_start_date).days),
        HMI_AD_weekly["date"].apply(lambda x: (x - CBI_start_date).days),
        HMI_AD_weekly["price"],
    )

    # Normalize HMI and RSI to the same level
    # First, start with RSI and HMI2
    mask = (CBI["date"] > t_switch0) & (CBI["date"] < t_switch1)
    ref = CBI[mask][["RSI", "HMI2"]].mean()
    CBI["HMI2"] = CBI["HMI2"] / ref["HMI2"]
    CBI["RSI"] = CBI["RSI"] / ref["RSI"]
    # Then, normalize HMI1
    mask = (CBI["date"] > RSI_start_date) & (CBI["date"] < HMI1_stop_date)
    ref = CBI[mask][["HMI1", "RSI"]].mean()
    CBI["HMI1"] = ref["RSI"] * CBI["HMI1"] / ref["HMI1"]

    # Create weights for HMI
    CBI["HMI1_weight"] = 1  # Create a new column to store the weight of HMI1
    CBI["HMI2_weight"] = 0  # Create a new column to store the weight of HMI2
    for i in range(len(CBI)):  # Loop through all rows
        if CBI["date"][i] >= RSI_start_date:
            if CBI["date"][i] <= HMI1_stop_date:
                CBI["HMI1_weight"][i] = (HMI1_stop_date - CBI["date"][i]).days / (
                    HMI1_stop_date - RSI_start_date
                ).days  # Linearly interpolate
            else:
                CBI["HMI1_weight"][i] = 0  # After t_switch1, use HMI only

        if CBI["date"][i] >= t_switch0:
            if CBI["date"][i] <= t_switch1:
                CBI["HMI2_weight"][i] = (CBI["date"][i] - t_switch0).days / (
                    t_switch1 - t_switch0
                ).days  # Linearly interpolate
            else:
                CBI["HMI2_weight"][i] = 1  # After t_switch1, use HMI only

    # Create the CBI. First merge HMI and RSI, then merge with HMI2
    CBI["price"] = (
        CBI["HMI1_weight"] * CBI["HMI1"] + (1 - CBI["HMI1_weight"]) * CBI["RSI"]
    )
    CBI["price"] = (
        CBI["price"] * (1 - CBI["HMI2_weight"]) + CBI["HMI2"] * CBI["HMI2_weight"]
    )

    # Add future date for interpolation purposes
    CBI = pd.concat(
        [
            CBI,
            pd.DataFrame(
                {
                    "date": datetime.date.today() + datetime.timedelta(days=60),
                    "price": CBI["price"].iloc[-1],
                },
                [0],
            ),
        ],
        ignore_index=True,
    )

    return CBI


def get_CBI(df_MT):
    """
    Creates a CBI using create_CBI_from_HMI_RSI_AD, by first fetching all necessary data from the MT database, as well as Added Data.
    Returns a CBI DataFrame with only the date and price columns.
    """

    # Create HMI and RSI Price Indeces, as well as HMI_AD using Added Data
    HMI_monthly = get_CBI_HMII_monthly(df_MT)
    RSI_weekly = get_CBI_RSI_weekly(df_MT)
    HMI_AD_weekly = get_CBI_HMI_AD_weekly()

    # Weighted smoothing of the weekly datasets RSI and HMI_AD
    RSI_weekly["price_orig"] = RSI_weekly["price"]
    HMI_AD_weekly["price_orig"] = HMI_AD_weekly["price"]

    # Smooth using something more sophisticated than smooth_w
    RSI_weekly["price"] = smooth_w(RSI_weekly["price"], RSI_weekly["count"], 2)
    HMI_AD_weekly["price"] = smooth_w(HMI_AD_weekly["price"], HMI_AD_weekly["count"], 2)

    # Create CBI by stitching together HMI, RSI and HMI_AD
    CBI = create_CBI_from_HMI_RSI_HMI_AD(HMI_monthly, RSI_weekly, HMI_AD_weekly)

    return CBI
