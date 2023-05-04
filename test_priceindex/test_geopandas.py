import datetime
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Solgt packages
from solgt.priceindex.repeatsales import get_RSI
from solgt.priceindex.hedonic import get_HMI
from solgt.db.MT_parquet import get_parquet_as_df

from solgt.timeseries.date_t_converter import convert_date_to_t, convert_t_to_date

import geopandas as gpd

fp = "../../data/dataprocessing/geodata/Basisdata_03_Oslo_25832_Grunnkretser_FGDB.gdb"
grunnkretser = gpd.read_file(fp, layer = 'grunnkretser_omrade')
grunnkretser.plot(column = 'grunnkretsnavn', figsize=(6, 6))
plt.show(block=False)

# Use WGS 84 (epsg:4326) as the geographic coordinate system
grunnkretser = grunnkretser.to_crs(epsg=4326)




# Plot GeoPandas Frame
from shapely.ops import unary_union

soner = gpd.GeoDataFrame()

sone_nrs = grunnkretser['grunnkretsnummer'].apply(lambda x: x[-4:-2] ).unique()
sone_nrs.sort()
boundaries = []
index = 1
for nr in sone_nrs :
    c_gk = grunnkretser[ grunnkretser['grunnkretsnummer'].apply(lambda x: x[-4:-2] == nr ) ]
    
    polygons = c_gk['geometry']
    boundary = gpd.GeoSeries(unary_union(polygons)).values
    
    soner = pd.concat([soner, gpd.GeoDataFrame({'sone':nr},geometry=boundary, index = [index])])
    index = index+1


soner.plot(column = 'sone', figsize=(6, 6))
plt.show(block=False)



# Plot priceindex
from solgt.priceindex import priceindex

PI = priceindex.Priceindex()

from solgt.db.MT_parquet import get_parquet_as_df



import pandas as pd
import numpy as np
import datetime

df_MT = get_parquet_as_df("C:\Code\data\MT.parquet")

dft = df_MT[
        #(df['housingtype'] == 'Leilighet') &
        (df_MT['adcreated_date'] > pd.to_datetime( datetime.datetime(2016,1,1), utc=True))  &
        (df_MT['PROM'] > 14) &
        (df_MT['PROM'] < 300)
    ][['price_inc_debt','adcreated_date','buildyear','PROM','housingtype','lat','lng','floor']].copy()
# Convert to datetime.date
dft["adcreated_date"] = dft["adcreated_date"].dt.date

dft['price_indexed'] = dft['price_inc_debt'] * PI.reindex(datetime.date.today(), dft['adcreated_date'])

dft['buildyear_cat'] = dft["buildyear"].apply(lambda x:  + 1*(x< 2015)+ 1*(x<2000) + 1*(x<1980)
                                        + 1*(x<1950) ).fillna(4)
dft['floor_cat'] = dft['floor'].apply(lambda x:  + 1*(x< 1)+ 1*(x<2) + 1*(x<4)
                                        + 1*(x<8) ).fillna(1)
dft['size_cat'] = dft['PROM'].astype('float').apply(lambda x: 
                        int( np.maximum(np.minimum(np.round(x/10),11),2) ) )

len(dft)




"""
# HEDONIC MODEL
"""
import solgt.priceindex.hedonic as hm

categorical_columns = ['buildyear_cat','size_cat']
numerical_columns = ['PROM']
features = categorical_columns + numerical_columns
y_col = 'price_indexed'

df2 = pd.DataFrame()

dfs = dft.copy()
pl,X,y = hm.linear_HM(dfs, y_col, categorical_columns, numerical_columns)
dfs['price_res'] = np.log(y/pl.predict(X))
df2 = pd.concat([df2, dfs])


#df2 = df2[['lng','lat','price_res']]
df2['price_res'] = np.maximum(df2['price_res'].values, -1)
df2['price_res'] = np.minimum(df2['price_res'].values, 1)



"""
# Function 
"""
def coords2distance(X0, X1) :
    # X0: scalars [lng,lat].
    # X1: vectors [lng,lat], 
    R = 6373.0    # radius of the Earth

    lng0 = np.radians(X0[0])
    lat0 = np.radians(X0[1])
    lng1 = np.radians(X1[:,0])
    lat1 = np.radians(X1[:,1])
    
    dlng = lng1 - lng0
    dlat = lat1 - lat0
    a = np.sin(dlat/2)**2 + np.cos(lat0) * np.cos(lat1) * np.sin(dlng/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = R * c *1000  # Distance in meters
    return distance
    



"""
Create a list of all dwellings in the city and plot them to a map
"""
ind = df2.reset_index().index.to_list()
X = df2[['lng','lat','price_res']].to_numpy()
pre_l = []
lng_l = []
lat_l = []

dist_m = 50

while len(ind) > 0 :
    print(len(ind))
    ii = ind.pop()
    i_d = np.where(coords2distance(X[ii,:2],X[ind,:2] ) < dist_m )[0]
    i_d = [ind[i] for i in i_d ]
    Ni = len(i_d)
    lng_l.append( (np.sum(X[i_d,0])+X[ii,0])/(Ni+1) )
    lat_l.append( (np.sum(X[i_d,1])+X[ii,1])/(Ni+1) )
    pre_l.append( (np.sum(X[i_d,2])+X[ii,2])/(Ni+1) )
    for i in i_d :
        ind.remove(i)
    #print(len(ind))
    
df_ds = pd.DataFrame({'lng':lng_l,
                      'lat':lat_l,
                      'price_res':pre_l,})




# Create map
import plotly.express as px
fig = px.scatter_mapbox(
                    #df2, 
                    df_ds,
                    lat="lat", 
                    lon="lng", 
                   # color_discrete_sequence=["fuchsia"],
                    color = 'price_res',
                    zoom=11, height=600)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

fig.write_html("output/Kart.html")



"""
POINTS
"""
df_points = gpd.GeoDataFrame(df2, geometry=gpd.points_from_xy(df2['lng'], df2['lat']))

df_points['grunnkrets_id'] = ''
for ii in grunnkretser.index :
    ind = df_points[ df_points['geometry'].within( grunnkretser['geometry'].loc[ii] ) ].index
    df_points.loc[ind,'grunnkrets_id'] = ii
    if ii%50 == 0 : 
        print(ii)
        
df_points = df_points.join(grunnkretser[['grunnkretsnummer','grunnkretsnavn']], on='grunnkrets_id' )



"""
Smooth functions:
"""
def smooth(y,n=5) :
    y = np.array(y)
    ySum = np.cumsum(y)
    ySum = np.insert(ySum, 0, 0, axis=0)
    N = len(y)
    nn = np.cumsum(np.ones(np.shape(y))).astype(int)
    nM = np.minimum(nn+n,N)
    nP = np.maximum(nn-n-1,0)

    ys = (ySum[nM] - ySum[nP])/(nM-nP)
    return ys

def smooth2(y,n=5) :
    y = np.array(y)
    ySum = np.cumsum(y)
    ySum = np.insert(ySum, 0, 0, axis=0)
    N = len(y)
    nn = np.cumsum(np.ones(np.shape(y))).astype(int)
    nM = np.minimum( np.minimum(nn+n,2*nn-1)   ,N)
    nP = np.maximum( np.maximum(nn-n-1, 2*nn-N-1 ) ,0)

    ys = (ySum[nM] - ySum[nP])/(nM-nP)
    return ys




"""
Machine learning:
"""
from sklearn import linear_model

mean_res_price = np.empty(len(grunnkretser))
rate_res_price = np.empty(len(grunnkretser))

date_interp = pd.Series(pd.date_range(start='2016-1-01', end='2021-6-30', freq='Q' ).date)
sparklines_mat = np.empty((len(grunnkretser), len(date_interp)))
sparklines_mat[:] = np.NaN

for ii, gk_ind in enumerate( grunnkretser.index ):
#for gk_ind in grunnkretser.index :
#for gk_ind in range(125,150) :
#for gk_ind in [438] :
    ind = df_points[df_points['grunnkrets_id'] == gk_ind].index
    #print(len(ind))
    if len(ind) < 50:
        mean_res_price[gk_ind] = float('NaN')
        rate_res_price[gk_ind] = float('NaN')
        continue
    
    c_price = df2.loc[ ind ][['price_res','adcreated_date']].sort_values('adcreated_date')
    c_price['price_res_exp'] = np.exp( c_price['price_res'] )
    c_price['price_res_smooth_exp'] = np.exp(smooth(c_price['price_res'],20))
    
    d0 = date_interp.min()

    sparklines_mat[gk_ind,:] = np.interp( (date_interp - d0).apply(lambda x: x.days),
                                          (c_price['adcreated_date'] - d0).apply(lambda x: x.days),
                                          c_price['price_res_smooth_exp']  )


    a = (c_price['adcreated_date'] - d0).apply(lambda x: x.days)
    b = a.values
    c = b.reshape(-1, 1)

    regr = linear_model.LinearRegression()
    lm = regr.fit(c, c_price['price_res'].values.reshape(-1, 1))
    y_pred = regr.predict(c)
    c_price['pred_exp'] = np.exp(y_pred)

    
    mean_res_price[gk_ind] = np.exp(c_price['price_res'].mean())
    rate_res_price[gk_ind] = lm.coef_*(1e9*60*60*24*365)*100 # Average percent change per year
    # Hvorfor i alle dager ganges det med et så stort tall? Det er ca 3*10^18
    # Plot
    #c_price.plot(x='adcreated',y=['price_res_exp','price_res_smooth_exp','pred_exp'])


rate_res_price = np.minimum( np.maximum(rate_res_price,-4), 4)
grunnkretser['mean_res_price'] = mean_res_price
grunnkretser['rate_res_price'] = rate_res_price
#grunnkretser.T


"""
Folium
"""

import folium
m = folium.Map(location=[59.92, 10.78], 
              zoom_start=12,)

folium.Choropleth(
    geo_data = grunnkretser['geometry'].to_json(),
    data = grunnkretser[['objid','rate_res_price']],
    columns = ['objid', 'rate_res_price'],
    key_on = "feature.id",
    fill_color = 'YlGnBu',
    bins = 7,
    fill_opacity = 0.7,
    line_opacity = 0.3,
    legend_name = "% Rate of change p.a compared to nominal",
).add_to(m)
folium.LayerControl().add_to(m)

m.save('output/osm_rate.html')
m



"""
Plot choropleth
"""
#gks = grunnkretser.set_index('grunnkretsnavn')
soner_t = soner.set_index('sone')
fig = px.choropleth(soner_t,
                   geojson=soner_t['geometry'],
                   locations=soner_t.index,
                #    color="rate_res_price",
                   projection="mercator")
_ = fig.update_geos(fitbounds="locations", visible=False)
_ = fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()

"""
FUNGERER IKKE: Da "rate_res_price" ikke er en del av soner_t
"""





"""

"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import base64

from io import BytesIO

def rate_sparkline(data, figsize=(4, 1.2), **kwags):
    """
    Returns a HTML image tag containing a base64 encoded sparkline style plot
    """
    data = list((data-1)*100)

    fig, ax = plt.subplots(1, 1, figsize=figsize, **kwags)
    ax.plot(data)
    for k,v in ax.spines.items():
        v.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.plot(len(data) - 1, data[len(data) - 1], 'r.')
    ax.set_ylim((max(data)+min(data))/2-10, (max(data)+min(data))/2+10)

    ax.fill_between(range(len(data)), data, len(data)*[0], alpha=0.1)

    img = BytesIO()
    plt.savefig(img, transparent=True, bbox_inches='tight')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.read()).decode("UTF-8")

