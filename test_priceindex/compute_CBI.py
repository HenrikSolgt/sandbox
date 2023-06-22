
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


def normalize_index(df_I, t_switch0, t_switch1):
    # Normalize so that the mean value is 1 in the period [t_switch0, t_switch1]

    # If series is datetime, make it into date
    if isinstance(df_I["date"][0], datetime.datetime):
        df_I["date"] = df_I["date"].apply(lambda x: x.date())

    # dt_obj_wo_tz = dt_obj_w_tz.replace(tzinfo=None)
    ref_I = df_I[(df_I["date"] > t_switch0) & (df_I["date"] < t_switch1)][
        "price"
    ].mean()

    df_I["price"] = df_I["price"] / ref_I

    return df_I


def get_HMI_AD_weekly():
    # Get (locally stored) added data
    path_AD = "C:/Code/data/dataprocessing/added_data/"
    df_AD = get_added_data(path_AD)

    # Filter HMI data and create monthly and weekly HMI and save as a figure
    t1 = datetime.datetime.now().astimezone()
    t0 = datetime.datetime.now().astimezone() - datetime.timedelta(days=AD_diff_days)
    df_AD = df_AD[ (df_AD['sold_date'] > t0) & (df_AD['sold_date'] <= t1) ]

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




def stitch_CBI(HMI_monthly, RSI_weekly, HMI_AD_weekly):
    """
    Stitches a CBI from three different price indices.
    """
    HMI1_start_date = HMI_monthly["date"].min()
    RSI_start_date = datetime.date(2010, 1, 1)
    HMI1_stop_date = datetime.date(2011, 1, 1)

    # Create RSI stop date when RSI_weekly count is below 10. This date is also the date of t_switch1
    RSI_stop_date = RSI_weekly[RSI_weekly["count"] > 10]["date"].max()
    t_switch1 = RSI_stop_date
    t_switch0 = RSI_stop_date - datetime.timedelta(days=t_switch0_diff)

    # Filter out data before RSI_start_date, since it is too sparse for the smooth_w to work
    RSI_weekly = RSI_weekly[(RSI_weekly["date"] >= RSI_start_date) & (RSI_weekly["date"] <= RSI_stop_date)]

    # Weighted smoothing of the inputs
    RSI_weekly["price_smooth"] = smooth_w(RSI_weekly["price"], RSI_weekly["count"], 2)
    HMI_AD_weekly["price_smooth"] = smooth_w(
        HMI_AD_weekly["price"], HMI_AD_weekly["count"], 2
    )

    # Create a new date series for CBI, starting on the first Monday after the first date in RSI_weekly
    dummy = (
        HMI1_start_date - datetime.date(2000, 1, 3)
    ).days % 7  # The given date is a Monday
    if dummy > 0:
        HMI1_start_date = HMI1_start_date + datetime.timedelta(days=7 - dummy)
    CBI_date_stop = HMI_AD_weekly["date"].max()
    CBI = pd.DataFrame()
    CBI["date"] = [
        HMI1_start_date + datetime.timedelta(days=7 * x)
        for x in range(0, (CBI_date_stop - HMI1_start_date).days // 7 + 1)
    ]

    # Create resampled dataseries for HMI an
    # d RSI, based on the CBI dates
    CBI["HMI1"] = np.interp(
        CBI["date"].apply(lambda x: (x - HMI1_start_date).days),
        HMI_monthly["date"].apply(lambda x: (x - HMI1_start_date).days),
        HMI_monthly["price"],
    )
    CBI["RSI"] = np.interp(
        CBI["date"].apply(lambda x: (x - HMI1_start_date).days),
        RSI_weekly["date"].apply(lambda x: (x - HMI1_start_date).days),
        RSI_weekly["price_smooth"],
    )
    CBI["HMI2"] = np.interp(
        CBI["date"].apply(lambda x: (x - HMI1_start_date).days),
        HMI_AD_weekly["date"].apply(lambda x: (x - HMI1_start_date).days),
        HMI_AD_weekly["price_smooth"],
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

    # Define time period
    date0 = datetime.date(2000, 1, 4)
    HMI1_start_date = datetime.date(2008, 1, 1)
    todaydate = datetime.date.today() 

    # Create HMI and RSI Price Indeces
    HMI_monthly = hmi.get_HMI(df_MT, HMI1_start_date, todaydate, "monthly")
    RSI_weekly = rsi.get_RSI(df_MT, date0, todaydate, "weekly")
    HMI_AD_weekly = get_HMI_AD_weekly()

    CBI = stitch_CBI(HMI_monthly, RSI_weekly, HMI_AD_weekly)
    
    # Keep only the date and price columns
    CBI = CBI[["date", "price"]]

    return CBI
