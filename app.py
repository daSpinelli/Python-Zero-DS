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
    _geofile_ = geopandas.read_file(_url_)
    return _geofile_


def set_features(_data_):
    # 1 ft = 0,3048 m
    ft_to_m = 0.3048
    _data_['sqm_lot'] = _data_['sqft_lot'] * ft_to_m
    _data_['sqm_price'] = _data_['price'] / _data_['sqm_lot']
    return _data_


def overview_data(_data_):
    # data selection

    # filters
    f_attributes = st.sidebar.multiselect(
        'Select features',
        _data_.columns.sort_values()
    )
    f_zipcode = st.sidebar.multiselect(
        'Enter zipcode',
        sorted(set(_data_['zipcode'].unique()))
    )

    if (f_zipcode != []) & (f_attributes != []):
        _data_ = _data_.loc[_data_['zipcode'].isin(f_zipcode), f_attributes]

    elif (f_zipcode != []) & (f_attributes == []):
        _data_ = _data_.loc[_data_['zipcode'].isin(f_zipcode), :]

    elif (f_zipcode == []) & (f_attributes != []):
        _data_ = _data_.loc[:, f_attributes]

    else:
        _data_ = _data_.copy()

    # average metrics dataframe: _df_metrics_
    _df1_ = _data_[['id', 'zipcode']].groupby('zipcode').count().reset_index()
    _df2_ = _data_[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    _df3_ = _data_[['sqft_living', 'zipcode']].groupby('zipcode').mean().reset_index()
    _df4_ = _data_[['sqm_price', 'zipcode']].groupby('zipcode').mean().reset_index()

    _m1_ = pd.merge(_df1_, _df2_, on='zipcode', how='inner')
    _m2_ = pd.merge(_m1_, _df3_, on='zipcode', how='inner')

    _df_metrics_ = pd.merge(_m2_, _df4_, on='zipcode', how='inner')
    _df_metrics_.columns = ['ZipCode', 'Total Houses', 'Price', 'SqFT Living', 'SqM Price']

    # statistics descriptive dataframe: _df_statistics_
    num_attributes = _data_.select_dtypes(include=['int64', 'float64'])

    media = pd.DataFrame(num_attributes.apply(np.mean))
    median = pd.DataFrame(num_attributes.apply(np.median))
    std = pd.DataFrame(num_attributes.apply(np.std))
    max_ = pd.DataFrame(num_attributes.apply(np.max))
    min_ = pd.DataFrame(num_attributes.apply(np.min))

    _df_statistics_ = pd.concat([max_, min_, media, median, std], axis=1).reset_index()
    _df_statistics_.columns = ['Attributes', 'Max', 'Min', 'Media', 'Median', 'Std']

    # grid 1: data overview
    st.title('Data Overview')
    st.dataframe(_data_.head())

    _c1_, _c2_ = st.beta_columns((1, 1))

    # grid 2: average metrics
    _c1_.header('Average Metrics')
    _c1_.dataframe(_df_metrics_, height=500)

    # grid 3: statistics descriptive
    _c2_.header('Statistics Descriptive')
    _c2_.dataframe(_df_statistics_, height=500)

    return None


def portfolio_density(_data_, _geofile_):

    st.title('Region Overview')

    _c1_, _c2_ = st.beta_columns((1, 1))

    # map 1: region overview
    _c1_.subheader('Portfolio Density')

    full = 0
    if full == 1:
        _df_ = _data_.copy()
    else:
        _df_ = _data_.sample(10)

    # Base Map - Folium
    density_map = folium.Map(
        location=[_data_['lat'].mean(), _data_['long'].mean()],
        default_zoom_start=15
    )

    marker_cluster = MarkerCluster().add_to(density_map)
    for name, row in _df_.iterrows():
        folium.Marker([row['lat'], row['long']],
                      popup=f"Sold for {row['price']} on {row['date']}\n"
                            f"Features: {row['sqft_living']} sqft,\n"
                            f"{row['bedrooms']} bedrooms,\n"
                            f"{row['bathrooms']} bathrooms,\n"
                            f"Year built: {row['yr_built']}\n"
                      ).add_to(marker_cluster)

    with _c1_:
        folium_static(density_map)

    # map 2: price density
    _c2_.subheader('Price Density')

    _df_ = _df_[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    _df_.columns = ['ZIP', 'PRICE']

    _geofile_ = _geofile_[_geofile_['ZIP'].isin(_df_['ZIP']).to_list()]

    region_price_map = folium.Map(
        location=[_data_['lat'].mean(), data['long'].mean()],
        default_zoom_start=15
    )

    region_price_map.choropleth(
        data=_df_,
        geo_data=_geofile_,
        columns=['ZIP', 'PRICE'],
        key_on='feature.properties.ZIP',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='AVG PRICE'
    )

    with _c2_:
        folium_static(region_price_map)

    return None


def commercial_attributes(_data_):

    st.sidebar.title('Commercial Options')
    st.title('Commercial Attributes')

    # filters
    min_yr_built = int(_data_['yr_built'].min())
    max_yr_built = int(_data_['yr_built'].max())

    st.sidebar.subheader('Select Max Year Built')
    f_yr_built = st.sidebar.slider(
        'Year Built',
        min_yr_built,
        max_yr_built,
        max_yr_built
    )

    min_date = datetime.strptime(datetime.strftime(_data_['date'].min(), format='%Y-%m-%d'), '%Y-%m-%d')
    max_date = datetime.strptime(datetime.strftime(_data_['date'].max(), format='%Y-%m-%d'), '%Y-%m-%d')

    st.sidebar.subheader('Select Max Date')
    f_date = st.sidebar.slider(
        'Date',
        min_date,
        max_date,
        max_date
    )

    min_price = int(_data_['price'].min())
    max_price = int(_data_['price'].max())
    median_price = int(_data_['price'].median())

    st.sidebar.subheader('Select Max Price')
    f_price = st.sidebar.slider(
        'Price',
        min_price,
        max_price,
        median_price
    )

    # graph 1: average price per year built
    st.subheader('Average Price per Year Built')
    _df_ = _data_.loc[_data_['yr_built'] <= f_yr_built]
    _df_ = _df_[['yr_built', 'price']].groupby('yr_built').mean().reset_index()

    _fig_ = px.line(_df_, x='yr_built', y='price')
    st.plotly_chart(_fig_, use_container_width=True)

    # graph 2: average price per day
    st.subheader('Average Price per Day')
    _data_['date'] = pd.to_datetime(_data_['date'], format='%Y-%m-%d')
    _df_ = _data_.loc[_data_['date'] <= f_date]
    _df_ = _df_[['date', 'price']].groupby('date').mean().reset_index()

    _fig_ = px.line(_df_, x='date', y='price')
    st.plotly_chart(_fig_, use_container_width=True)

    # graph 3 [histogram]: price distribution
    st.header('Price Distribution')
    _df_ = _data_.loc[_data_['price'] <= f_price]
    _fig_ = px.histogram(_df_, x='price', nbins=50)
    st.plotly_chart(_fig_, use_container_width=True)

    return None


def physical_attributes(_data_):

    # filters
    st.sidebar.title('Attributes Options')
    f_bedrooms = st.sidebar.selectbox(
        'Max number of bedrooms',
        sorted(set(_data_['bedrooms'].unique()), reverse=True)
    )

    f_bathrooms = st.sidebar.selectbox(
        'Max number of bathrooms',
        sorted(set(_data_['bathrooms'].unique()), reverse=True)
    )

    f_floors = st.sidebar.selectbox(
        'Max number of floors',
        sorted(set(_data_['floors'].unique()), reverse=True)
    )

    f_waterview = st.sidebar.checkbox(
        'Only houses with water view'
    )

    st.title('House Attributes')

    _c1_, _c2_ = st.beta_columns(2)

    # histogram 1: house per bedrooms
    _c1_.subheader('House per Bedrooms')
    _df_ = _data_[_data_['bedrooms'] <= f_bedrooms]
    _fig_ = px.histogram(_df_, x='bedrooms', nbins=19)
    _c1_.plotly_chart(_fig_, use_container_width=True)

    # histogram 2: house per bathrooms
    _c2_.subheader('House per Bathrooms')
    _df_ = _data_[_data_['bathrooms'] <= f_bathrooms]
    _fig_ = px.histogram(_df_, x='bathrooms', nbins=19)
    _c2_.plotly_chart(_fig_, use_container_width=True)

    _c1_, _c2_ = st.beta_columns(2)

    # histogram 3: house per floors
    _c1_.subheader('House per Floors')
    _df_ = _data_[_data_['floors'] <= f_floors]
    _fig_ = px.histogram(_df_, x='floors', nbins=19)
    _c1_.plotly_chart(_fig_, use_container_width=True)

    # histogram 4: house per water view
    _c2_.subheader('Water View')

    if f_waterview:
        _df_ = _data_[_data_['waterfront'] == 1]
    else:
        _df_ = _data_.copy()

    _fig_ = px.histogram(_df_, x='waterfront', nbins=6)
    _c2_.plotly_chart(_fig_, use_container_width=True)

    return None


if __name__ == '__main__':

    # data extraction
    file_path = 'database/kc_house_data.csv'
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'

    data = get_data(file_path)

    geofile = get_geofile(url)

    # data transformation
    data = set_features(data)

    overview_data(data)

    portfolio_density(data, geofile)

    commercial_attributes(data)

    physical_attributes(data)
