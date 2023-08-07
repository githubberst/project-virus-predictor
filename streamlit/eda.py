import streamlit as st
from streamlit_option_menu import option_menu

import pickle
import pandas as pd
import numpy as np

import json
import geopandas as gpd

import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import folium as fs
from streamlit_folium import folium_static



from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from pathlib import Path

# import datasets


# import spray dataset
spray_path = Path(__file__).parent / 'data/spray2.csv'
spray_df = pd.read_csv(spray_path)

# import combined train and weather (feature engineered) dataset
final_path = Path(__file__).parent / 'data/final_eda.csv'
eda_df = pd.read_csv(final_path)

# import chicago map data
chicago_path = Path(__file__).parent / './data/chicago-community-areas.geojson'
chicago = gpd.read_file(chicago_path)

with open(chicago_path) as f:
    chicago_geojson = json.load(f)





# streamlit shell (layouts etc)
# set webpage name and icon
st.set_page_config(
    page_title='The West Nile Virus in Chicago',
    page_icon='🦟',
    layout='wide',
    initial_sidebar_state='expanded'
    )

# top navigation bar
selected = option_menu(
    menu_title = None,
    options = ['Facts And Figures', 'WNV in Chicago','Risk In Your Area'],
    icons = ['eyeglasses','bar-chart','microbe'],
    default_index = 0, # which tab it should open when page is first loaded
    orientation = 'horizontal',
    styles={
        'nav-link-selected': {'background-color': '#FF7F0E'}
        }
    )

if selected == 'Facts And Figures':
    # title
    st.title('Historic Data For Weather Patterns And Occurrence of the WNV')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)



    # comparative bar/line chart 
    option = st.selectbox('Pick a variable to see the relationship with the presence of the WNV',
                            ('Precipitation', 'Average Temperature', 'Wind Direction', 
                             'Length of Night','Fog', 'Haze', 'Mist','Rain','Thunderstorms'))
    
    if option == 'Precipitation':
        variable = 'roll_sum_28_PrecipTotal'
    elif option == 'Average Temperature':
        variable = 'roll_mean_28_Tavg'
    elif option == 'Wind Direction':
        variable = 'ResultDir'
    elif option == 'Length of Night':
        variable = 'night_length'
    elif option == 'Fog':
        variable = 'roll_sum_28_isFG'
    elif option == 'Haze':
        variable = 'roll_sum_28_isHZ'
    elif option == 'Mist':
        variable = 'roll_sum_28_isBR'
    elif option == 'Rain':
        variable = 'roll_sum_28_isRA'
    elif option == 'Thunderstorms':
        variable = 'roll_sum_28_isTS'

        
    variable_WnvP_df = eda_df[['Month',variable,'WnvPresent']].groupby('Month').mean()
    variable_WnvP_df = variable_WnvP_df.reset_index(col_level=1)
    variable_WnvP_df = variable_WnvP_df.astype({'Month': 'string'})
    
    fig, ax1 = plt.subplots(figsize=(7,7))
    sns.lineplot(data=variable_WnvP_df,x='Month',y=variable,ax=ax1,
                 color='orange',sort=False).set(ylabel=variable)
    ax1_patch = mpatches.Patch(color='orange', label=variable)
    ax1.legend(handles=[ax1_patch], loc="upper left")
    
    ax2 = ax1.twinx()
    sns.barplot(data=variable_WnvP_df,x='Month',y='WnvPresent',color='blue',ax=ax2, 
                alpha=0.5).set(ylabel='WnvPresent')
    ax2.grid(False)
    ax2_patch = mpatches.Patch(color='blue', label='WnvPresent')
    ax2.legend(handles=[ax2_patch], loc="upper right")
    
    fontsize=8
    labelsize=5
    ax1.set_title(str(option) + ' vs WnvPresent', fontsize=fontsize)
    ax1.set_ylabel(str(option),fontsize=fontsize)
    ax2.set_ylabel('WnvPresent',fontsize=fontsize)
    ax1.set_xlabel('Month',fontsize=fontsize)
    ax1.tick_params(labelsize=labelsize)
    
    st.pyplot(fig)
    
    
    # 
    species = eda_df[['WnvPresent', 'Species']].groupby('Species').sum() 
    


    # Weather patterns in Chicago
    st.header('Patterns between the weather and presence of the WNV')
  
    # text explaination for  
    st.write(':')
    st.write('The..')

    st.text("") # add extra line in between


if selected == 'WNV in Chicago':
    # title
    st.title('Visualisation of the WNV in Chicago')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)

    st.header('Chicago')
    
    # explain the animation
    st.write('Pick a date on the slider to see the number of WNV positive mosquitoes, or press play to see the changing values over time.')
    
    # time to make the mosquito dataframe for mapping
    
    eda_df['Date'] = pd.to_datetime(eda_df['Date'])
    eda_df['Year-Month'] = eda_df['Date'].dt.strftime('%Y %m')
    eda_df['Year-Month'] = pd.to_datetime(eda_df['Year-Month'], format='%Y %m').dt.to_period('M')
    
    # Calculate total 'NumMosquitos'
    total_mosquito = eda_df.groupby(['Address','Year-Month'], as_index=False)['NumMosquitos'].sum()
    total_mosquito.sort_values(by='Year-Month', inplace=True)
    
    # Calculate median 'latitude' and 'longitude' for each address
    areas = eda_df.groupby('Address', as_index=False)[['Latitude', 'Longitude']].median()
    
    # Calculate total number of 'WnvPresent'
    virus = eda_df.groupby('Address', as_index=False)['WnvPresent'].sum()
    
    # merge datasets together
    mos_data = pd.merge(total_mosquito, areas, on='Address')
    mos_data = pd.merge(mos_data, virus, on='Address')
    
    # since we no longer need 'Address', drop col
    mos_data.drop('Address', axis = 1, inplace = True)
    
    # sort by 'Year-Month'
    mos_data.sort_values(by='Year-Month', inplace=True)
    
    
    # Convert dataframe to geodataframe
    mos_geo = gpd.GeoDataFrame(mos_data, geometry= gpd.points_from_xy(mos_data.Longitude, mos_data.Latitude))
    
    # Output with community areas added to the mosquito dataframe
    mos_chicago = gpd.sjoin(mos_geo, chicago, op='within')
    
    # Summary with some actionable content
    # community_infections is the df we will use for display because it is the cleanest summary
    # it will be called later with an option to choose how many rows you want to see
    community_infections = mos_chicago[['community','NumMosquitos', 'WnvPresent']].groupby('community').sum()
    community_infections.sort_values('WnvPresent', inplace = True, ascending = False)
    chicago_ltd = chicago[['community', 'geometry']]
    community_infections2 = chicago_ltd.merge(community_infections, on='community')
    community_infections2.sort_values('WnvPresent', inplace = True, ascending = False)
    community_infections2.reset_index(inplace = True)
    
    
    # Now for the actual animated map
    
    # Set the Mapbox access token
    px.set_mapbox_access_token('pk.eyJ1IjoiZ2l0aHViYmVyc3QiLCJhIjoiY2xqb3RtcjlwMWp4aDNscWNjdHZuNmU1ayJ9.BizJFoOXaa2H5jsYDkFeSg')
    
    
    # Create a scatter mapbox
    fig_m = px.scatter_mapbox(mos_chicago, 
                            lat=mos_chicago.geometry.y, 
                            lon=mos_chicago.geometry.x,
                            color='NumMosquitos', size='WnvPresent',
                            color_continuous_scale=px.colors.sequential.Jet,
                            hover_data=['NumMosquitos', 'WnvPresent', 'community'], zoom=9, animation_frame='Year-Month')
    
    
    # Create a layer for the community area boundaries
    layer_chicago = dict(
        sourcetype = 'geojson',
        source = chicago_geojson,
        type='fill',
        color='hsla(0, 100%, 90%, 0.2)',  
        below='traces',
        )
    
    # Add the community area boundaries layer to the scatter map
    fig_m.update_layout(mapbox_layers=[layer_chicago])
    
    # Update the layout
    fig_m.update_layout(mapbox_style= 'stamen-toner',
                      title='WNV+ vs. Mosquito count',
                      autosize=False,
                      width=1200,
                      height=1200,
                      )
    
    # Display the figure
    st.plotly_chart(fig_m)
    
    
    

    
    # Request input for how many rows to show
    header_n = st.slider('Select the top number of communities affected by positive WNV mosquitoes', 1, 20, 5)
    
    st.write('Note: Numbers shown below are cumulative. There are a total of 61 communities in our data set.')
    
    # Display the top n communities in table form
    show_communities = community_infections.head(header_n)
    st.dataframe(show_communities)
    
    # Display the top n communities in map form
    show_communities2 = community_infections2.head(header_n)
    
    m2 = fs.Map(location=[41.881832, -87.623177],tiles = 'Stamen Terrain', zoom_start=10, scrollWheelZoom=False)
    m2.choropleth(geo_data = chicago_geojson, 
                    data = show_communities2,
                    columns = ['community', 'WnvPresent'],
                    key_on = 'feature.properties.community',
                    fill_color = 'YlOrRd', 
                    fill_opacity = 0.7, 
                    line_opacity = 0.2,
                    legend_name = 'Number of WNV Positive Mosquitos per Neighbourhood in the years 2007, 2009, 2011, 2013')
        
    for lat, lon, community, wnv in zip(community_infections2.geometry.centroid.y, community_infections2.geometry.centroid.x , community_infections2.community, community_infections2.WnvPresent):
        fs.CircleMarker(location=[f'{lat}',f'{lon}'], radius = 1, color = 'green', fill = True, tooltip=f'{community}, Total Number of WNV Positive Mosquitoes: {wnv}').add_to(m2)    
    
    st_map = folium_static(m2, width=1200)


if selected == 'Risk In Your Area':
    # title
    st.title('Risk of WNV in different neighbourhoods')
    st.subheader('by Eden, Enoch, Sandra, and Wynne')
    style = "<div style='background-color:#FF7F0E; padding:2px'></div>"
    st.markdown(style, unsafe_allow_html = True)
    
    st.header('Risk in your area')
    st.subheader("Click on the map to see the risk for that area.")
    
    
    