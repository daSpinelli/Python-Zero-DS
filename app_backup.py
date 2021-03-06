import pandas as pd
import streamlit as st
import numpy as np
import folium
import geopandas
import plotly.express as px
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from datetime import datetime

# page config
st.set_page_config(layout='wide')


@st.cache(allow_output_mutation=True)
def get_data(path):
    return pd.read_csv(path)


@st.cache(allow_output_mutation=True)
def get_geofile(_url_):
    _geofile = geopandas.read_file(_url_)
    return _geofile


# get data
file_path = 'database/kc_house_data.csv'
data = get_data(file_path)

# get geofile
url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
geofile = get_geofile(url)

# add features
# 1 ft = 0,3048 m
ft_to_m = 0.3048
data['sqm_lot'] = data['sqft_lot'] * ft_to_m
data['sqm_price'] = data['price'] / data['sqm_lot']

# data selection
f_attributes = st.sidebar.multiselect('Select features', data.columns.sort_values())
f_zipcode = st.sidebar.multiselect('Enter zipcode', data['zipcode'].sort_values().unique())

# applying filters
if (f_zipcode != []) & (f_attributes != []):
    data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
elif (f_zipcode != []) & (f_attributes == []):
    data = data.loc[data['zipcode'].isin(f_zipcode), :]
elif (f_zipcode == []) & (f_attributes != []):
    data = data.loc[:, f_attributes]
else:
    data = data.copy()

# average metrics
df1 = data[['id', 'zipcode']].groupby('zipcode').count().reset_index()
df2 = data[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df3 = data[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
df4 = data[['sqm_price', 'zipcode']].groupby('zipcode').mean().reset_index()

m1 = pd.merge(df1, df2, on='zipcode', how='inner')
m2 = pd.merge(m1, df3, on='zipcode', how='inner')
df = pd.merge(m2, df4, on='zipcode', how='inner')

df.columns = ['ZipCode', 'Total Houses', 'Price', 'SqFT Living', 'SqM Price']

# statistics descriptive
num_attributes = data.select_dtypes(include=['int64', 'float64'])

media = pd.DataFrame(num_attributes.apply(np.mean))
median = pd.DataFrame(num_attributes.apply(np.median))
std = pd.DataFrame(num_attributes.apply(np.std))
max_ = pd.DataFrame(num_attributes.apply(np.max))
min_ = pd.DataFrame(num_attributes.apply(np.min))

df1 = pd.concat([max_, min_, media, median, std], axis=1).reset_index()

df1.columns = ['Attributes', 'Max', 'Min', 'Media', 'Median', 'Std']

###########################
# Data Overview
###########################
st.title('Data Overview')
st.dataframe(data)

c1, c2 = st.beta_columns((1, 1))

c1.header('Average Metrics')
c1.dataframe(df, height=500)

c2.header('Statistics Descriptive')
c2.dataframe(df1, height=500)

###########################
# Portfolio Density
###########################
st.title('Region Overview')

c1, c2 = st.beta_columns((1, 1))

c1.subheader('Portfolio Density')

full = 0
if full == 1:
    df = data.copy()
else:
    df = data.sample(10)

# Base Map - Folium
density_map = folium.Map(
    location=[data['lat'].mean(), data['long'].mean()],
    default_zoom_start=15
)

marker_cluster = MarkerCluster().add_to(density_map)
for name, row in df.iterrows():
    folium.Marker([row['lat'], row['long']],
                  popup=f"Sold for {row['price']} on {row['date']}\n"
                        f"Features: {row['sqft_living']} sqft,\n"
                        f"{row['bedrooms']} bedrooms,\n"
                        f"{row['bathrooms']} bathrooms,\n"
                        f"Year built: {row['yr_built']}\n"
                  ).add_to(marker_cluster)

with c1:
    folium_static(density_map)

###########################
# Portfolio Density
###########################

c2.subheader('Price Density')

df = df[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
df.columns = ['ZIP', 'PRICE']

geofile = geofile[geofile['ZIP'].isin(df['ZIP']).to_list()]

region_price_map = folium.Map(
    location=[data['lat'].mean(), data['long'].mean()],
    default_zoom_start=15
)

region_price_map.choropleth(
    data=df,
    geo_data=geofile,
    columns=['ZIP', 'PRICE'],
    key_on='feature.properties.ZIP',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='AVG PRICE'
)

with c2:
    folium_static(region_price_map)

###########################
# Commercial Attributes
###########################
st.sidebar.title('Commercial Options')
st.title('Commercial Attributes')

# -- Filters
min_yr_built = int(data['yr_built'].min())
max_yr_built = int(data['yr_built'].max())

st.sidebar.subheader('Select Max Year Built')
f_yr_built = st.sidebar.slider(
    'Year Built',
    min_yr_built,
    max_yr_built,
    max_yr_built
)

min_date = datetime.strptime(data['date'].min(), '%Y%m%dT%H%M%S')
max_date = datetime.strptime(data['date'].max(), '%Y%m%dT%H%M%S')

st.sidebar.subheader('Select Max Date')
f_date = st.sidebar.slider(
    'Date',
    min_date,
    max_date,
    max_date
)

min_price = int(data['price'].min())
max_price = int(data['price'].max())
median_price = int(data['price'].median())

st.sidebar.subheader('Select Max Price')
f_price = st.sidebar.slider(
    'Price',
    min_price,
    max_price,
    median_price
)

# -- Graphs
st.subheader('Average Price per Year Built')
df = data.loc[data['yr_built'] <= f_yr_built]
df = df[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

fig = px.line(df, x='yr_built', y='price')
st.plotly_chart(fig, use_container_width=True)


st.subheader('Average Price per Day')
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
df = data.loc[data['date'] <= f_date]
df = df[['date', 'price']].groupby('date').mean().reset_index()

fig = px.line(df, x='date', y='price')
st.plotly_chart(fig, use_container_width=True)


# -- Histogram
st.header('Price Distribution')
df = data.loc[data['price'] <= f_price]
fig = px.histogram(df, x='price', nbins=50)
st.plotly_chart(fig, use_container_width=True)

###########################
# Physical characteristics
###########################

# -- Filters
st.sidebar.title('Attributes Options')
f_bedrooms = st.sidebar.selectbox(
    'Max number of bedrooms',
    sorted(set(data['bedrooms'].unique()))
)

f_bathrooms = st.sidebar.selectbox(
    'Max number of bathrooms',
    sorted(set(data['bathrooms'].unique()))
)

f_floors = st.sidebar.selectbox(
    'Max number of floors',
    sorted(set(data['floors'].unique()))
)

f_waterview = st.sidebar.checkbox(
    'Only houses with water view'
)

# -- Graphs
st.title('House Attributes')

c1, c2 = st.beta_columns(2)

# House per bedrooms
c1.subheader('House per Bedrooms')
df = data[data['bedrooms'] <= f_bedrooms]
fig = px.histogram(df, x='bedrooms', nbins=19)
c1.plotly_chart(fig, use_container_width=True)

# House per bathrooms
c2.subheader('House per Bathrooms')
df = data[data['bathrooms'] <= f_bathrooms]
fig = px.histogram(df, x='bathrooms', nbins=19)
c2.plotly_chart(fig, use_container_width=True)

c1, c2 = st.beta_columns(2)

# House per floors
c1.subheader('House per Floors')
df = data[data['floors'] <= f_floors]
fig = px.histogram(df, x='floors', nbins=19)
c1.plotly_chart(fig, use_container_width=True)

# House per water view
c2.subheader('Water View')

if f_waterview:
    df = data[data['waterfront'] == 1]
else:
    df = data.copy()

fig = px.histogram(df, x='waterfront', nbins=6)
c2.plotly_chart(fig, use_container_width=True)






















