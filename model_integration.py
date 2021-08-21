# Imports
from os import system
from pandas.core.algorithms import mode
from pandas.core.frame import DataFrame
import streamlit as st
import pandas as pd
import geocoder
import numpy as np
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from geopy.distance import geodesic
import plotly.graph_objects as go
import sys
import pyarrow

sys.tracebacklimit = 0
token = st.secrets['map_token']


# Model

@st.cache
def load_files():
    return pd.read_parquet('hospital_model3')


@st.cache
def load_files2():
    prices = pd.read_parquet('prices_pruned')
    prices.set_index('npi_number', inplace=True)
    return prices


class HospitalPricingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, threshold=50):

        self.hospital_loc = load_files2()
        self.prices = load_files2()

    def _get_distance(
        self,
        p_lat,
        p_lng,
        threshold=50,
        ):

        df = pd.DataFrame()
        df['distance'] = self.hospital_loc.apply(lambda x: \
                geodesic((p_lat, p_lng), (x['Lat'], x['Lng'])).miles,
                axis=1)

        return self.hospital_loc.loc[df.distance <= threshold,
                ['npi_number']]

    def fit(self):
        return self

    def description(self):
        new_df = self.prices
        return new_df['short_description'].unique().tolist()

    def convert_loc(self, address):
        error_catcher = geocoder.osm(address)
        if error_catcher.ok:
            g = geocoder.mapbox(address, key=token)
            return (g.json['lat'], g.json['lng'])
        else:
            st.error('Enter valid location!')
            sys.exit()

    def get_filtered(self, X):
        (address, description) = X
        (patient_lat, patient_lng) = self.convert_loc(address)
        available_hospitals = self._get_distance(patient_lat,
                patient_lng)
        available_prices = \
            self.prices.join(available_hospitals.set_index('npi_number'
                             ), on='npi_number', how='inner')
        filtered = \
            available_prices.loc[available_prices.short_description.str.contains(description.upper())].reset_index()
        return filtered

    def predict(self, filtered):
        prediction = {'min price': filtered['price'].min(),
                      'mean price': filtered['price'].mean(),
                      'max price': filtered['price'].max()}
        return pd.DataFrame(prediction, index=[0])

    def get_mean_prices(self, filtered):
        prices = self.hospital_loc.loc[self.hospital_loc['npi_number'
                ].isin(filtered['npi_number'].tolist())]
        mean_prices = pd.merge(prices, filtered[['npi_number', 'price'
                               ]], on='npi_number')
        mean_prices = mean_prices.groupby(by=['npi_number', 'Lat', 'Lng'
                , 'name', 'url'], as_index=False)['price'].mean()
        mean_prices.sort_values(by=['price'])
        return mean_prices


# Initialize model

model = HospitalPricingClassifier()


# Mapping

def make_fig(mean_prices, address):
    fig = go.Figure()
    (lat, lng) = model.convert_loc(address)
    fig.add_trace(go.Scattermapbox(
        lat=mean_prices['Lat'],
        lon=mean_prices['Lng'],
        mode='markers',
        marker=go.scattermapbox.Marker(size=17, color='rgb(0, 255, 127)'
                , opacity=0.7),
        text=mean_prices['name'],
        hoverinfo='text',
        ))

    fig.add_trace(go.Scattermapbox(
        lat=(lat, ),
        lon=(lng, ),
        mode='markers',
        marker=go.scattermapbox.Marker(size=17,
                color='rgb(250, 128, 114)', opacity=0.7),
        text=str(address),
        hoverinfo='text',
        ))

    fig.update_layout(hoverlabel=dict(bgcolor='white', font_size=16,
                      font_family='Rockwell'), autosize=True,
                      hovermode='closest', showlegend=False,
                      mapbox=dict(
        accesstoken=token,
        bearing=0,
        center=dict(lat=38, lon=-96),
        pitch=0,
        zoom=3,
        style='light',
        ))

    return fig


# Streamlit

with st.form(key='form_one'):
    st.title('Hospital Pricing Model')
    address = st.text_input('Enter location')
    procedure = st.selectbox('Choose procedure', model.description())
    value = st.slider('Radius search for hospitals in miles',
                      min_value=0, max_value=50)
    submit = st.form_submit_button('Find')

if submit:
    model.threshold = value
    filtered = pd.DataFrame(model.get_filtered((str(address),
                            str(procedure))))
    st.header('Procedure Pricing')
    st.dataframe(pd.DataFrame(model.predict(filtered)))
    st.header('Mapped Data')
    st.plotly_chart(make_fig(model.get_mean_prices(filtered), address),
                    use_container_width=True)
    mean_prices = pd.DataFrame(model.get_mean_prices(filtered))
    st.dataframe(pd.DataFrame(mean_prices.drop(columns=['npi_number',
                 'Lat', 'Lng'])))

st.header('Data Visualization')
