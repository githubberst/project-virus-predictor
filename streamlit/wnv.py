import streamlit as st
from streamlit_option_menu import option_menu

from datetime import datetime
import pickle
import json
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objs as go
import altair as alt
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

# Load your model and define the function for predicting WNV probabilities
model = pickle.load(open('models/model.pkl', 'rb'))
data = pd.read_csv('./data/final_df.csv')
streamlit_data = pd.read_csv('data/streamlit_df.csv')

# Load the Chicago boundaries from GeoJSON file
with open('./data/communityareas.geojson') as f:
    chicago_geojson = json.load(f)

# Set the Mapbox access token
px.set_mapbox_access_token('pk.eyJ1IjoiZ2l0aHViYmVyc3QiLCJhIjoiY2xqb3RtcjlwMWp4aDNscWNjdHZuNmU1ayJ9.BizJFoOXaa2H5jsYDkFeSg')

# dictionaries that we will be using
species_map = {'Culex Pipiens': 0.088922,
                   'Culex Pipiens/Restuans': 0.055135,
                   'Culex Restuans': 0.017883}
trap_map = {'T001': 0.0,
                 'T002': 0.0972972972972973,
                 'T003': 0.11475409836065574,
                 'T004': 0.0,
                 'T005': 0.14285714285714285,
                 'T006': 0.16666666666666666,
                 'T007': 0.0,
                 'T008': 0.07194244604316546,
                 'T009': 0.08108108108108109,
                 'T011': 0.08270676691729323,
                 'T012': 0.0380952380952381,
                 'T013': 0.09615384615384616,
                 'T014': 0.13333333333333333,
                 'T015': 0.11428571428571428,
                 'T016': 0.10606060606060606,
                 'T017': 0.0,
                 'T018': 0.0,
                 'T019': 0.0,
                 'T025': 0.0,
                 'T027': 0.07526881720430108,
                 'T028': 0.07692307692307693,
                 'T030': 0.06349206349206349,
                 'T031': 0.03424657534246575,
                 'T033': 0.05154639175257732,
                 'T034': 0.0,
                 'T035': 0.041666666666666664,
                 'T036': 0.041666666666666664,
                 'T037': 0.034482758620689655,
                 'T039': 0.05,
                 'T040': 0.0,
                 'T043': 0.0,
                 'T044': 0.0,
                 'T045': 0.08571428571428572,
                 'T046': 0.0,
                 'T047': 0.02631578947368421,
                 'T048': 0.0273972602739726,
                 'T049': 0.011764705882352941,
                 'T050': 0.0,
                 'T051': 0.0,
                 'T054': 0.03680981595092025,
                 'T054C': 0.1111111111111111,
                 'T060': 0.0,
                 'T061': 0.07766990291262135,
                 'T062': 0.012195121951219513,
                 'T063': 0.019230769230769232,
                 'T065': 0.045871559633027525,
                 'T066': 0.05172413793103448,
                 'T067': 0.03333333333333333,
                 'T069': 0.012048192771084338,
                 'T070': 0.1,
                 'T071': 0.0,
                 'T072': 0.0,
                 'T073': 0.03125,
                 'T074': 0.017391304347826087,
                 'T075': 0.0,
                 'T076': 0.0,
                 'T077': 0.030303030303030304,
                 'T078': 0.0,
                 'T079': 0.012048192771084338,
                 'T080': 0.021739130434782608,
                 'T081': 0.06818181818181818,
                 'T082': 0.08163265306122448,
                 'T083': 0.02702702702702703,
                 'T084': 0.04,
                 'T085': 0.030303030303030304,
                 'T086': 0.09302325581395349,
                 'T088': 0.0,
                 'T089': 0.04395604395604396,
                 'T090': 0.046357615894039736,
                 'T091': 0.03125,
                 'T092': 0.0,
                 'T094': 0.03305785123966942,
                 'T094B': 0.0,
                 'T095': 0.05555555555555555,
                 'T096': 0.12,
                 'T097': 0.058823529411764705,
                 'T099': 0.0,
                 'T100': 0.0,
                 'T102': 0.017857142857142856,
                 'T103': 0.07228915662650602,
                 'T107': 0.09090909090909091,
                 'T114': 0.060810810810810814,
                 'T115': 0.07564575645756458,
                 'T128': 0.06875,
                 'T129': 0.0,
                 'T135': 0.04371584699453552,
                 'T138': 0.050955414012738856,
                 'T141': 0.0,
                 'T142': 0.05555555555555555,
                 'T143': 0.1935483870967742,
                 'T144': 0.02702702702702703,
                 'T145': 0.01098901098901099,
                 'T146': 0.0,
                 'T147': 0.04081632653061224,
                 'T148': 0.0,
                 'T149': 0.0,
                 'T150': 0.0,
                 'T151': 0.038461538461538464,
                 'T152': 0.021505376344086023,
                 'T153': 0.0,
                 'T154': 0.09523809523809523,
                 'T155': 0.06976744186046512,
                 'T156': 0.047619047619047616,
                 'T157': 0.0,
                 'T158': 0.04081632653061224,
                 'T159': 0.036036036036036036,
                 'T160': 0.04395604395604396,
                 'T161': 0.0,
                 'T162': 0.047619047619047616,
                 'T200': 0.007751937984496124,
                 'T206': 0.0,
                 'T209': 0.023076923076923078,
                 'T212': 0.019736842105263157,
                 'T215': 0.06666666666666667,
                 'T218': 0.02702702702702703,
                 'T219': 0.0,
                 'T220': 0.04081632653061224,
                 'T221': 0.08035714285714286,
                 'T222': 0.016666666666666666,
                 'T223': 0.10344827586206896,
                 'T224': 0.018518518518518517,
                 'T225': 0.10679611650485436,
                 'T226': 0.05194805194805195,
                 'T227': 0.05128205128205128,
                 'T228': 0.0967741935483871,
                 'T229': 0.0,
                 'T230': 0.1076923076923077,
                 'T231': 0.09523809523809523,
                 'T232': 0.04,
                 'T233': 0.14,
                 'T235': 0.11290322580645161,
                 'T236': 0.030303030303030304,
                 'T237': 0.0,
                 'T238': 0.0,
                 'T900': 0.088,
                 'T903': 0.07142857142857142}

def update_df(df, features_to_fill, updated_features):
    for i, feature in enumerate(features_to_fill):
        df[feature] = updated_features[i]
    return df

def model_predict(df, model=model):
    df2 = df.copy()
    df2['Trap'] = df2['Trap'].map(trap_map)
    df2.drop(['AddressNumberAndStreet','Latitude', 'Longitude'], axis=1, inplace=True)
    x = model.predict_proba(df2)[:,1]
    df['WnvProbability'] = x
    return df

# Format page
st.set_page_config(page_title='West Nile Virus Prediction',
                    page_icon=':mosquito:',
                    layout='wide',
                    initial_sidebar_state='expanded')

st.title("West Nile Virus Prediction")


# Top navigation bar
selected = option_menu(menu_title = None,
                           options = ['Exploratory Data Analysis', 'Predictor'],
                           icons = ['bar-chart',':mirror:'],
                           default_index = 0, # which tab it should open when page is first loaded
                           orientation = 'horizontal',
                           styles={'nav-link-selected': {'background-color': '#FF7F0E'}}
                           )



# Handle click events on the map
if selected == 'Exploratory Data Analysis':
    pass


if selected == 'Predictor':
    st.subheader('Please fill in the following details')
    species = st.selectbox('Species',['Culex Pipiens/Restuans', 'Culex Restuans', 'Culex Pipiens'], index=0)
    species = species_map[species]

    depart = st.slider('Departure from normal temperature', min_value=-20, max_value=20, step=1)

    sunrise = st.time_input('Sunrise')
    sunset = st.time_input('Sunset')
    timediff = datetime.combine(datetime.today(), sunset) - datetime.combine(datetime.today(), sunrise)
    timediff = 24 - timediff.seconds / 3600

    sunrise = sunrise.hour * 100 + sunrise.minute
    sunset = sunset.hour * 100 + sunset.minute

    codesum = st.multiselect('CodeSum', ['Normal', 'BR', 'HZ', 'RA', 'TS', 'VCTS'])
    codesum = 0.042585423329405826 # Codesum score for normal

    roll_sum_21_PrecipTotal = st.slider('Rolling Sum of Precipitation (inches) (21 days)', min_value=00, max_value=25, step=1)
    roll_sum_28_PrecipTotal = st.slider('Rolling Sum of Precipitation (inches) (28 days)', min_value=00, max_value=25, step=1)
    roll_mean_7_Tmin = st.slider('Minimum Temperature (°F) (7 days rolling mean)', min_value=40, max_value=90, step=1)
    roll_mean_28_Tmin = st.slider('Minimum Temperature (°F) (28 days rolling mean)', min_value=40, max_value=90, step=1)
    roll_mean_28_Tavg = st.slider('Average Temperature (°F) (28 days rolling mean)', min_value=40, max_value=90, step=1)
    
    date = st.date_input('Date')
    month = date.month
    year = date.year

    num_trap = st.number_input('Average number of times checked for each trap a day', min_value=0, step=1)
    roll_sum_14_num_trap = num_trap * 14
    speciesXroll_sum_28_num_trap =  species * (num_trap * 28)


    features_to_fill = ['Species', 'Depart', 'Sunrise', 'Sunset', 'CodeSum',
                        'roll_sum_21_PrecipTotal', 'roll_sum_28_PrecipTotal',
                        'roll_mean_7_Tmin', 'roll_mean_28_Tmin', 'roll_mean_28_Tavg',
                        'Month', 'Year', 'num_trap', 'roll_sum_14_num_trap',
                        'speciesXroll_sum_28_num_trap', 'timediff']
    if num_trap is not None:
        updated_features = [species, depart, sunrise, sunset, codesum,
                            roll_sum_21_PrecipTotal, roll_sum_28_PrecipTotal,
                            roll_mean_7_Tmin, roll_mean_28_Tmin, roll_mean_28_Tavg,
                            month, year, num_trap, roll_sum_14_num_trap,
                            speciesXroll_sum_28_num_trap, timediff]
        update_df(streamlit_data, features_to_fill, updated_features)
        model_predict(streamlit_data)

    st.subheader('Map')
    fig = px.scatter_mapbox(streamlit_data,
                            lat='Latitude',
                            lon='Longitude',
                            hover_name='Trap',
                            hover_data=['AddressNumberAndStreet', 'Latitude', 'Longitude', 'WnvProbability'],
                            size='WnvProbability',
                            color='WnvProbability',
                            color_continuous_scale=[[0, 'rgb(255, 200, 200)'], [1, 'rgb(255, 0, 0)']],
                            range_color=[0, 1],
                            zoom=10)
    # Create a layer for the community area boundaries
    layer_chicago = dict(
    sourcetype = 'geojson',
    source = chicago_geojson,
    type='fill',
    color='hsla(0, 100%, 90%, 0.2)',  
    below='traces',
    )


    fig.update_layout(mapbox_layers=[layer_chicago],
                      mapbox_style='carto-positron',
                      margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                      height=600,
                      width=1000)

    st.plotly_chart(fig)
    
    st.subheader('Areas with the highest probability of WNV')
    st.write(streamlit_data[streamlit_data['WnvProbability'] == streamlit_data['WnvProbability'].max()][['AddressNumberAndStreet', 'WnvProbability']])

    st.subheader('Areas with the lowest probability of WNV')
    st.write(streamlit_data[streamlit_data['WnvProbability'] == streamlit_data['WnvProbability'].min()][['AddressNumberAndStreet', 'WnvProbability']])