# Imports
from os import system
import streamlit as st
import pandas as pd
import geocoder
import numpy as np
import json
from geopy.distance import geodesic
import plotly.graph_objects as go
import sys
import pyarrow

sys.tracebacklimit = 0
token = st.secrets['map_token']


# Model

@st.cache
def load_files():
    prices = pd.read_parquet('prices_pruned')
    prices.set_index('npi_number', inplace=True)
    return prices


class HospitalPricingClassifier():

    def __init__(self):

        self.hospital_loc = pd.read_parquet('hospital_model3')
        self.prices = load_files()

    
    def _get_distance(
        self,
        p_lat,
        p_lng,
        threshold,
        ):
        
        self.hospital_loc['distance'] = self.hospital_loc.apply(lambda x: \
                geodesic((p_lat, p_lng), (x['Lat'], x['Lng'])).miles,
                axis=1)

        return self.hospital_loc.loc[self.hospital_loc['distance'] <= threshold,
                ['npi_number']]
    
    
    def hospital_list(self):
        return self.hospital_loc

    
    def description(self):
        return self.prices['short_description'].unique().tolist()
    
    
    def convert_address(self, lat, lng):
        latlng = [lat, lng]
        g = geocoder.mapbox(latlng, method='reverse')
        return g.json['address']

    
    def convert_loc(self, address):
        g = geocoder.osm(address)
        if g.ok == False:
            return []
        else:
            g = geocoder.mapbox(address, key=token)
            return [g.json['lat'], g.json['lng']]

    
    def get_filtered(self, description, cli_loc, threshold):
        patient_lat = cli_loc[0]
        patient_lng = cli_loc[1]
        available_hospitals = self._get_distance(patient_lat,
                patient_lng, threshold)
        available_prices = \
            self.prices.join(available_hospitals.set_index('npi_number'
                             ), on='npi_number', how='inner')
        filtered = \
            available_prices.loc[available_prices.short_description.str.contains(description.upper())].reset_index()
        return filtered

    
    def predict(self, filtered):
        prediction = {'mean price': filtered['price'].mean(),
                      'min price': filtered['price'].min(),
                      'max price': filtered['price'].max()}
        return pd.DataFrame(prediction, index=[0])

    
    def get_mean_prices(self, filtered):
        prices = self.hospital_loc.loc[self.hospital_loc['npi_number'
                ].isin(filtered['npi_number'].tolist())]
        mean_prices = pd.merge(prices, filtered[['npi_number', 'price'
                               ]], on='npi_number')
        mean_prices = mean_prices.groupby(by=['npi_number', 'Lat', 'Lng'
                , 'name', 'url', 'distance'], as_index= False)['price'].mean()
        mean_prices.sort_values(by=['price'], inplace = True)
        return mean_prices


# Initialize model

model = HospitalPricingClassifier()


# Mapping

def make_fig(mean_prices, cli_loc):
    fig = go.Figure()
    lat = cli_loc[0]
    lng = cli_loc[1]
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
    threshold = st.slider('Radius search for hospitals in miles',
                      min_value=0, max_value=50)
    submit = st.form_submit_button('Find')
    
if submit:
    cli_loc = model.convert_loc(address)
    if not cli_loc:
        st.error('Please enter valid location.')
    else:
        filtered = pd.DataFrame(model.get_filtered(procedure, cli_loc, threshold))
        if filtered.empty:
            st.error('Sorry, no hospitals within radius threshold contains searched procedure.')
        else:
            st.header('Procedure Pricing')
            st.dataframe(pd.DataFrame(model.predict(filtered)))
            st.header('Mapped Data')
            st.plotly_chart(make_fig(model.get_mean_prices(filtered), cli_loc),
                            use_container_width=True)
            mean_prices = pd.DataFrame(model.get_mean_prices(filtered))
            mean_prices.drop(columns=['npi_number',
                         'Lat', 'Lng'], inplace = True)
            mean_prices.reset_index()
            st.dataframe(pd.DataFrame(mean_prices))


with st.form(key='form_two'):
    st.title('Hospital List')
    searched_hospital = st.selectbox('Search if your hospital is included!', model.hospital_list()['name'].tolist())
    search = st.form_submit_button('Search')

if search:
    hospital_df = model.hospital_list()
    searched_row = hospital_df.loc[hospital_df['name'] == searched_hospital]
    lat = searched_row['Lat'].iloc[0]
    lng = searched_row['Lng'].iloc[0]
    st.header('Hospital Information')
    st.text('Hospital: ' + str(searched_hospital))
    st.text('NPI Number: ' + str(searched_row['npi_number'].iloc[0]))
    st.text('URL: ' + str(searched_row['url'].iloc[0]))
    st.text('Address: ' + str(model.convert_address(lat, lng)))
