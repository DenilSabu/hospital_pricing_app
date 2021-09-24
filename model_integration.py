# Imports
import os.path
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
import unittest
import sys
from selenium import webdriver

sys.tracebacklimit = 0
token = st.secrets['map_token']
username = st.secrets['username']
access_key = st.secrets['access_key']


# Preproccessing 

@st.cache
def load_prices():
    prices = pd.read_parquet('prices_pruned')
    prices.set_index('npi_number', inplace=True)
    return prices


def load_hospitals():
    hospitals =  pd.read_parquet('hospital_model3')
    if os.path.exists('test.txt'):
        test = []
        
        with open('test.txt') as f:
            for line in f:
                test.append(line.replace("\n", ""))
        
        st.write(test)
                
        if test[0] not in hospitals['npi_number'].unique():
            new_df = pd.DataFrame([test[0], 'Denil', test[1], 0.0, 0.0], columns  = ['npi_number', 'name', 'url', 'Lat', 'Lng'], ignore_index=True)
            hospitals.append(new_df)
            
    return hospitals

            
def convert_address(lat, lng):
        latlng = [lat, lng]
        g = geocoder.mapbox(latlng, method='reverse', key = token)
        return g.json['address']
    
    
def convert_loc(address):
        g = geocoder.osm(address)
        if g.ok == False:
            return []
        else:
            g = geocoder.mapbox(address, key=token)
            return [g.json['lat'], g.json['lng']]
    
#Model  

class HospitalPricingClassifier():

    def __init__(self):

        self.hospital_loc = load_hospitals()
        self.prices = load_prices()

    
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

# Find NPI number from Git

def findNPI(npi_number):
    
    desired_caps = {
        "build": 'PyunitTest sample build',
        "name": 'Py-unittest',
        "platform": 'Windows 10', 
        "browserName": 'Firefox', 
        "version": '92.0',
        "resolution": '1024x768', 
        "console": 'true', 
        "network":'true'   
    }

    driver = webdriver.Remote(
        command_executor="https://{}:{}@hub.lambdatest.com/wd/hub".format(username, access_key),
        desired_capabilities= desired_caps)

    driver.get('https://npiregistry.cms.hhs.gov/')

    npi_box = driver.find_element_by_name('number')
    npi_box.send_keys(npi_number)

    npi_button = driver.find_element_by_xpath("/html/body/div[2]/div[2]/div/form/div[7]/div/div/input[2]")
    npi_button.click()
    
    if (len(str(auth)) > 8):
        hospital_name = driver.find_element_by_xpath("/html/body/div[2]/div[2]/div/table/tbody/tr/td[2]").text
        hospital_address = driver.find_element_by_xpath("/html/body/div[2]/div[2]/div/table/tbody/tr/td[4]").text
        hospital_address = hospital_address.replace("\n", " ")
        driver.quit()
        return hospital_name, hospital_address
    
    driver.quit()


# Mapping with Plotly

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
    cli_loc = convert_loc(address)
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
            mean_prices.drop(columns=['npi_number', 'url',
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
    npi_number = str(searched_row['npi_number'].iloc[0])
    st.text('NPI Number: ' + npi_number)
    st.text('URL: ' + str(searched_row['url'].iloc[0]))
    st.text('Address: ' + str(convert_address(lat, lng)))
    findNPI(npi_number)
