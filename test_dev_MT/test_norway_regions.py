import pandas as pd
import datetime
import MT_dev_parquet
import matplotlib.pyplot as plt
from solgt.priceindex.repeatsales import get_RSI
from solgt.priceindex.rsi_score import score_RSI

# TODO: Must rewrite score_RSI to use the new repeatedsales code


import plotly.graph_objects as go


def load_fylker_map():
    # Load fylker.csv into a DataFrame, and set the index to "Region"
    fylker_map = pd.read_csv("fylker.csv", sep=",")
    fylker_map.set_index("Region", inplace=True)

    # Find unique fylker

    fylker = fylker_map["Fylke"].unique()
    fylker.sort()

    nyfylker_map = pd.DataFrame(data=fylker, columns = ["Fylke"])

    nyfylker_map["Nyfylke"] = [
        "Viken",
        "Agder",
        "Viken",
        "Troms og Finnmark",
        "Innlandet",
        "Vestland",
        "Møre og Romsdal",
        "Trøndelag",
        "Nordland",
        "Innlandet",
        "Oslo",
        "Rogaland",
        "Vestland",
        "Trøndelag",
        "Vestfold og Telemark",
        "Troms og Finnmark",
        "Agder",
        "Vestfold og Telemark",
        "Viken"
    ]

    nyfylker_map.set_index("Fylke", inplace=True)

    return fylker_map, nyfylker_map


def load_data():
    df = MT_dev_parquet.get_parquet_as_df()
        
    # Remove geographical outsiders
    df = df[df["lat"] > 55].reset_index(drop=True)

    # Compute region
    df["region"] = df["postcode"] // 100
    
    fylker_map, nyfylker_map = load_fylker_map()

    # Use fylker to put the correct region to fylke 
    df["fylke"] = df["region"].map(fylker_map["Fylke"])

    df["nyfylke"] = df["fylke"].map(nyfylker_map["Nyfylke"])

    return df






"""
TODO: 
- Beregn RSI for hele Norge
- Splitt på fylke, og beregn RSI for hvert fylke
- Sjekk hvor god score vi får for hvert fylke, sammenliknet med å bruke RSI for hele Norge
"""


# Load data
df = load_data()

# Load fylker and nyfylker
fylker_map, nyfylker_map = load_fylker_map()
fylker = fylker_map["Fylke"].unique()
fylker.sort()
nyfylker = nyfylker_map["Nyfylke"].unique()
nyfylker.sort()

df_Oslo = df[df["fylke"] == "Oslo"].reset_index(drop=True)

df_Oslo_train = df_Oslo[df_Oslo["date"] < datetime.date(2020, 1, 1)].reset_index(drop=True)
df_Oslo_test = df_Oslo[df_Oslo["date"] >= datetime.date(2020, 1, 1)].reset_index(drop=True)







df.groupby("fylke").count()["_id"]



df.groupby("nyfylke").count()["_id"]

plt.show()

# Create RSI

date0 = datetime.date(2008, 1, 1)
date1 = datetime.date(2023, 1, 1)

period = "quarterly"

rsi_Norway = get_RSI(df, date0, date1, period=period, interpolate=True)

# Iterate all fylker and create RSI
rsi_fylker = {}
for fylke in fylker:
    print(fylke)
    df_fylke = df[df["fylke"] == fylke].reset_index(drop=True)
    rsi_fylker[fylke] = get_RSI(df_fylke, date0, date1, period=period, interpolate=True)

# Iterate all fylker and create RSI
rsi_nyfylker = {}
for nyfylke in nyfylker:
    print(nyfylke)
    df_nyfylke = df[df["nyfylke"] == nyfylke].reset_index(drop=True)
    rsi_nyfylker[nyfylke] = get_RSI(df_nyfylke, date0, date1, period=period, interpolate=True)



"""
SCORING
"""

# Scoring
n_bins_array = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
score_Norway = score_RSI(df, n_bins_array, all_test_data=False, max_bins=1024)



"""
PLOTTING
"""
# plot using graph objects
fig1 = go.Figure()
print("Plotting:")
fig1 = fig1.add_trace(go.Scatter(x=rsi_Norway["date"], y=rsi_Norway["price"], name="Norway"))
for fylke in fylker:
    print(fylke)
    fig1 = fig1.add_trace(go.Scatter(x=rsi_fylker[fylke]["date"], y=rsi_fylker[fylke]["price"], name=fylke))
fig1.show()
# fig1.write_html("../output/rsi.html")

# Plot count
fig2 = go.Figure()
fig2 = fig2.add_trace(go.Scatter(x=rsi_Norway["date"], y=rsi_Norway["count"], name="Norway"))
for fylke in fylker:
    print(fylke)
    fig2 = fig2.add_trace(go.Scatter(x=rsi_fylker[fylke]["date"], y=rsi_fylker[fylke]["count"], name=fylke))
fig2.show()
# fig2.write_html("../output/rsi_count.html")



# plot using graph objects
fig3 = go.Figure()
print("Plotting:")
fig3 = fig3.add_trace(go.Scatter(x=rsi_Norway["date"], y=rsi_Norway["price"], name="Norway"))
for nyfylke in nyfylker:
    print(nyfylke)
    fig3 = fig3.add_trace(go.Scatter(x=rsi_nyfylker[nyfylke]["date"], y=rsi_nyfylker[nyfylke]["price"], name=nyfylke))
fig3.show()
# fig3.write_html("../output/rsi_nyfylke.html")


# Plot count
fig4 = go.Figure()
fig4 = fig4.add_trace(go.Scatter(x=rsi_Norway["date"], y=rsi_Norway["count"], name="Norway"))
for nyfylke in nyfylker:
    print(nyfylke)
    fig4 = fig4.add_trace(go.Scatter(x=rsi_nyfylker[nyfylke]["date"], y=rsi_nyfylker[nyfylke]["count"], name=nyfylke))
fig4.show()
# fig4.write_html("../output/rsi_nyfylke_count.html")




