import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopandas.tools import sjoin
import pydeck as pdk
import os
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np

# Set page configuration to wide layout
st.set_page_config(page_title="Interactive PM2.5 Data", layout="wide")
# Title
st.markdown("""
    <h1 style='text-align: center;'>Interactive PM&#8322;&#x2E;&#8325; Data with spatio-temporal Selection</h1>
""", unsafe_allow_html=True)
# gdf = gpd.read_file(
#     'bgd_adm_bbs_20201113_shp/bgd_admbnda_adm3_bbs_20201113.shp')
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
# Apply custom CSS to adjust selectbox width
st.markdown(
    """
    <style>
    .stSelectbox select {
        width: fit-content !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)
with col1:
    selected_month = st.selectbox(
        'Select a month', months, key='select_month')
with col2:
    selected_year = st.selectbox(
        'Select a year', list(range(2000, 2020)), key='select_year')

data = pd.read_hdf(
    'pm25_final.h5', key=f'{selected_month}_{selected_year}')

corr_df = pd.read_csv('corr_df.csv')
# # Convert lat/lon data points into a GeoDataFrame
# geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
# geo_data = gpd.GeoDataFrame(data, crs="EPSG:4326", geometry=geometry)
# geo_corr = gpd.GeoDataFrame(corr_df, crs="EPSG:4326", geometry=geometry)

# Dropdown to select the scope (Bangladesh, Division, District)
scope = st.radio(
    "Select Map Scope",
    ('Whole Bangladesh', 'Division', 'District'), horizontal=True
)

if scope == 'Whole Bangladesh':
    # # No need for spatial join, use the entire filtered data for Bangladesh
    # filtered_geo_data = sjoin(
    #     geo_data, gdf, how="inner", predicate="within")
    # filtered_corr = sjoin(
    #     geo_corr, gdf, how="inner", predicate="within")
    filtered_geo_data= data
    filtered_corr= corr_df
    s = 'Bangladesh'
    z = 5.5

elif scope == 'Division':
    # Select a division
    division_options = list(data['ADM1_EN'].unique())
    division_options.sort()
    selected_division = st.selectbox(
        'Select Division', division_options, key='select_division')
    s = f'{selected_division}'
    z = 7.5

    # # Filter the division boundary from the shapefile
    # selected_division_shape = gdf[gdf['ADM1_EN']
    #                               == selected_division]

    # Spatial join to filter climate data points that fall within the selected division
    # filtered_geo_data = sjoin(
    #     geo_data, selected_division_shape, how="inner", predicate="within")
    # filtered_corr = sjoin(
    #     geo_corr, selected_division_shape, how="inner", predicate="within")
    filtered_geo_data = data[data['ADM1_EN'] == selected_division]
    filtered_corr = corr_df[corr_df['ADM1_EN'] == selected_division]

elif scope == 'District':
    # Select a district
    district_options = list(data['ADM2_EN'].unique())
    district_options.sort()
    selected_district = st.selectbox(
        'Select District', district_options, key='select_district')
    s = f'{selected_district}'
    z = 8

    # # Filter the district boundary from the shapefile
    # selected_district_shape = gdf[gdf['ADM2_EN']
    #                               == selected_district]

    
    filtered_geo_data = data[data['ADM2_EN'] == selected_district]
    filtered_corr = corr_df[corr_df['ADM2_EN'] == selected_district]

    

# Input to select the model for comparison
selected_model = st.radio('Select Model to compare prediction with observed values', [
    'GNN+LSTM', 'GNN', 'CNN+LSTM', 'CNN'], key='select_model', horizontal=True)
selected_model = selected_model.lower()
df = filtered_geo_data[['latitude', 'longitude', 'observed',
                        'gnn+lstm', 'gnn', 'cnn+lstm', 'cnn', 'ADM1_EN', 'ADM2_EN', 'ADM3_EN']]
corr = filtered_corr[['latitude', 'longitude',
                     'gnn+lstm', 'gnn', 'cnn+lstm', 'cnn', 'ADM1_EN', 'ADM2_EN', 'ADM3_EN']]
# corr[['gnn+lstm', 'gnn', 'cnn+lstm', 'cnn']
#      ] = corr[['gnn+lstm', 'gnn', 'cnn+lstm', 'cnn']]*10000
main_df = df.copy()


def get_status_color(value):
    if value <= 12:
        # Green for Good
        return "Good", [0, 255, 0, 160], "green"
    elif value <= 35.4:
        # Yellow for Moderate
        return "Moderate", [255, 255, 0, 160], "yellow"
    elif value <= 55.4:
        # Orange for Unhealthy for Sensitive Groups
        return "Unhealthy for Sensitive Groups", [255, 165, 0, 160], "orange"
    elif value <= 150.4:
        # Red for Unhealthy
        return "Unhealthy", [255, 0, 0, 160], "red"
    elif value <= 250.4:
        # Purple for Very Unhealthy
        return "Very Unhealthy", [153, 50, 204, 160], "purple"
    else:
        # Dark Red for Hazardous
        return "Hazardous", [128, 0, 0, 160], "darkred"


# Use assign to avoid SettingWithCopyWarning
df = df.assign(observed_status=df['observed'].apply(lambda x: get_status_color(x)[0]),
               observed_color=df['observed'].apply(
                    lambda x: get_status_color(x)[1]),
               observed_color_css=df['observed'].apply(
                    lambda x: get_status_color(x)[2]),
               predicted_status=df[selected_model].apply(
                    lambda x: get_status_color(x)[0]),
               predicted_color=df[selected_model].apply(
                    lambda x: get_status_color(x)[1]),
               predicted_color_css=df[selected_model].apply(
                    lambda x: get_status_color(x)[2]),
            #    observed=df['observed'].round(2),
               predicted_elevation= df[selected_model] * 50,
               observed_elevation= df['observed'] * 50
               )

# For corr DataFrame, continue using assign to avoid issues
corr = corr.assign(scaled_elevation=corr[selected_model] * 5000)
df[['predicted_elevation', 'observed_elevation']] = df[[selected_model,'observed']] * 50
# print(type(df), df.shape)
# corr.rename(columns={'corr_gnn_lstm': 'gnn+lstm', 'corr_gnn':'gnn', 'corr_cnn_lstm':'cnn+lstm','cor_cnn':'cnn'})
# basemap_options = st.selectbox(
#     "Select Basemap", ['light', 'dark', 'satellite', 'streets'])
# region_selector_layer = pdk.Layer(
#     'PolygonLayer',
#     data=None,  # No initial data; user-drawn polygons will be used
#     get_polygon="coordinates",
#     get_fill_color="[200, 0, 0, 50]",
#     pickable=True,
#     auto_highlight=True
# )
if st.button('Generate Plot'):
    # observed_layer = pdk.Layer(
    #     'ScatterplotLayer',
    #     df,
    #     # Longitude and latitude for position
    #     get_position='[longitude, latitude]',
    #     get_radius=750,  # Set radius (adjust for your map)
    #     # Color based on observed value
    #     get_fill_color='[255, observed * 2, 100]',
    #     pickable=True,
    #     tooltip=True
    # )

    # # Step 2: Create a PyDeck layer for predicted values (GNN-LSTM)
    # predicted_layer = pdk.Layer(
    #     'ScatterplotLayer',
    #     df,
    #     # Longitude and latitude for position
    #     get_position='[longitude, latitude]',
    #     get_radius=750,  # Set radius (adjust for your map)
    #     # Color based on predicted value
    #     get_fill_color=f'[255, {selected_model} * 2, 100]',
    #     pickable=True,
    #     tooltip=True
    # )
    observed_layer = pdk.Layer(
        "GridCellLayer",
        df,
        # Specify the center of each grid cell
        get_position='[longitude, latitude]',
        cell_size=1000,  # Size of the grid cells (in meters)
        get_elevation='observed_elevation',  # Elevation represents the observed value
        # Color based on observed value
        get_fill_color='observed_color',
        pickable=True,
        extruded=True,  # Extrude to give a 3D effect
        elevation_scale=1  # Scale of the elevation
    )

    # Step 2: Create a PyDeck GridCellLayer for predicted values (GNN-LSTM)
    predicted_layer = pdk.Layer(
        "GridCellLayer",
        df,
        # Specify the center of each grid cell
        get_position='[longitude, latitude]',
        cell_size=1000,  # Size of the grid cells (in meters)
        get_elevation='predicted_elevation',  # Elevation represents the predicted value
        # Color based on predicted value
        get_fill_color='predicted_color',
        pickable=True,
        extruded=True,  # Extrude to give a 3D effect
        elevation_scale=1  # Scale of the elevation
    )
    corr_layer = pdk.Layer(
        "GridCellLayer",
        corr,
        # Specify the center of each grid cell
        get_position='[longitude, latitude]',
        cell_size=1000,  # Size of the grid cells (in meters)
        get_elevation='scaled_elevation',  # Elevation represents the predicted value
        # Color based on predicted value
        get_fill_color=f'[255, {selected_model}*255, 100, 128]',
        pickable=True,
        extruded=True,  # Extrude to give a 3D effect
        elevation_scale=1  # Scale of the elevation
    )

    # Step 3: Define a view state for both maps (same view for consistency)
    view_state = pdk.ViewState(
        latitude=df['latitude'].mean(),
        longitude=df['longitude'].mean(),
        zoom=z,
        pitch=40
    )

    # Step 4: Create PyDeck Deck objects for both observed and predicted data
    observed_deck = pdk.Deck(
        layers=[observed_layer],
        initial_view_state=view_state,
        map_style=f'mapbox://styles/mapbox/streets-v12',
        tooltip={
            "html": f"""
                <b>Latitude:</b> {{latitude}} <br/>
                <b>Longitude:</b> {{longitude}} <br/>
                <b>Division:</b> {{ADM1_EN}}<br><b>District:</b> {{ADM2_EN}}<br>
                <b>Upazila:</b> {{ADM3_EN}}<br>
                <b>Observed:</b> <span style='color:{{observed_color_css}};'>{{observed}} ({{observed_status}})</span><br/>
                <b>Predicted:</b> <span style='color:{{predicted_color_css}};'>{{{selected_model}}} ({{predicted_status}})</span>
                """,
            "style": {
                "color": "white"
            }
        }
    )

    predicted_deck = pdk.Deck(
        layers=[predicted_layer],
        initial_view_state=view_state,
        map_style=f'mapbox://styles/mapbox/streets-v12',
        tooltip={
            "html": f"""
                <b>Latitude:</b> {{latitude}} <br/>
                <b>Longitude:</b> {{longitude}} <br/>
                <b>Division:</b> {{ADM1_EN}}<br><b>District:</b> {{ADM2_EN}}<br>
                <b>Upazila:</b> {{ADM3_EN}}<br>
                <b>Observed:</b> <span style='color:{{observed_color_css}};'>{{observed}} ({{observed_status}})</span><br/>
                <b>Predicted:</b> <span style='color:{{predicted_color_css}};'>{{{selected_model}}} ({{predicted_status}})</span>
                """,
            "style": {"color": "white"}
        }
    )
    corr_deck = pdk.Deck(
        layers=[corr_layer],
        initial_view_state=view_state,
        map_style=f'mapbox://styles/mapbox/streets-v12',
        tooltip={
            "html": f"<b>Latitude:</b> {{latitude}} <br><b>Longitude:</b> {{longitude}}<br>"
            f"<b>Division:</b> {{ADM1_EN}}<br><b>District:</b> {{ADM2_EN}}<br>"
            f"<b>Upazila:</b> {{ADM3_EN}}<br>"
            f"<b>Correlation:</b> <span style='color:yellow;'>{{{selected_model}}}</span>",
            "style": {"color": "white"}
        }
    )

    # Step 5: Integrate into Streamlit
    st.markdown(f"""
                <h2 style='text-align: center;'>{s} at {selected_month}-{selected_year}</h2>
                """, unsafe_allow_html=True)

    # Columns for the maps
    col1, col2 = st.columns(2)

    # Conditionally display the maps based on checkbox selections
    with col1:
        st.markdown("""
        <h3 style='text-align: center;'>Observed Data Map</h3>
    """, unsafe_allow_html=True)
        st.pydeck_chart(observed_deck)

    with col2:
        st.markdown("""
        <h3 style='text-align: center;'>Predicted Data Map</h3>
    """, unsafe_allow_html=True)
        st.pydeck_chart(predicted_deck)
    observed_counts = df['observed_status'].value_counts().to_dict()
    predicted_counts = df['predicted_status'].value_counts().to_dict()
    total_count = len(df)
    # st.write("""
    # ### PM$_{2.5}$ Concentration Ranges and Health Impacts:
    # - **0-12 µg/m³ (Good)**: Air quality is considered satisfactory, posing little or no health risk.
    # - **12.1-35.4 µg/m³ (Moderate)**: Air quality is acceptable, but sensitive individuals may experience health issues.
    # - **35.5-55.4 µg/m³ (Unhealthy for Sensitive Groups)**: Sensitive groups may experience health effects.
    # - **55.5-150.4 µg/m³ (Unhealthy)**: Everyone may experience health effects.
    # - **150.5-250.4 µg/m³ (Very Unhealthy)**: Severe health effects for everyone.
    # - **>250.5 µg/m³ (Hazardous)**: Health warnings; emergency conditions.
    # """)
    # Calculate statistics for observed and predicted values
    observed_min = df['observed'].min()
    observed_max = df['observed'].max()
    observed_mean = df['observed'].mean()
    observed_median = df['observed'].median()

    predicted_min = df[selected_model].min()
    predicted_max = df[selected_model].max()
    predicted_mean = df[selected_model].mean()
    predicted_median = df[selected_model].median()

    # RMSE (Root Mean Squared Error)
    # Assuming 'actual' and 'predicted' are the columns of your dataframe
    df['squared_diff'] = (df['observed'] - df[selected_model]) ** 2
    rmse = np.sqrt(df['squared_diff'].mean())

    # MAE (Mean Absolute Error)
    mae = mean_absolute_error(df['observed'], df[selected_model])

    # Add the markdown section with statistics and counts combined in a single table
    st.markdown(f"""
    ### PM$_{{2.5}}$ Concentration Ranges and Health Impacts:
    Total data points within selected spatio-temporal range: {total_count}<br>
    Root of mean squared error (RMSE): {rmse:.2f}<br>
    Mean absolute error (MAE): {mae:.2f}<br>
    **Comparison of Observed and Predicted Values**

    | Statistic/Category                 | Observed | Predicted ({selected_model}) |
    |-----------------------------------|-----------------------|------------------------|
    | **Minimum (µg/m³)**               | {observed_min:.2f}    | {predicted_min:.2f}    |
    | **Maximum (µg/m³)**               | {observed_max:.2f}    | {predicted_max:.2f}    |
    | **Mean (µg/m³)**                  | {observed_mean:.2f}   | {predicted_mean:.2f}   |
    | <span style='color:green;'>**Good**</span> count           | {observed_counts.get('Good', 0)}    | {predicted_counts.get('Good', 0)}    |
    | <span style='color:yellow;'>**Moderate**</span> count    | {observed_counts.get('Moderate', 0)} | {predicted_counts.get('Moderate', 0)} |
    | <span style='color:orange;'>**Unhealthy for Sensitive Groups**</span> count | {observed_counts.get('Unhealthy for Sensitive Groups', 0)} | {predicted_counts.get('Unhealthy for Sensitive Groups', 0)} |
    | <span style='color:red;'>**Unhealthy**</span> count  | {observed_counts.get('Unhealthy', 0)} | {predicted_counts.get('Unhealthy', 0)} |
    | <span style='color:purple;'>**Very Unhealthy**</span> count | {observed_counts.get('Very Unhealthy', 0)} | {predicted_counts.get('Very Unhealthy', 0)} |
    | <span style='color:darkred;'>**Hazardous**</span> count     | {observed_counts.get('Hazardous', 0)} | {predicted_counts.get('Hazardous', 0)} |
    - <span style='color:green;'>**0-12 µg/m³ (Good)**</span>: Satisfactory, posing little or no health risk. 
      <br>
    - <span style='color:yellow;'>**12.1-35.4 µg/m³ (Moderate)**</span>: Acceptable, but sensitive individuals may experience health issues. 
      <br>
    - <span style='color:orange;'>**35.5-55.4 µg/m³ (Unhealthy for Sensitive Groups)**</span>: Sensitive groups may experience health effects. 
      <br>
    - <span style='color:red;'>**55.5-150.4 µg/m³ (Unhealthy)**</span>: Everyone may experience health effects. 
      <br>
    - <span style='color:purple;'>**150.5-250.4 µg/m³ (Very Unhealthy)**</span>: Severe health effects for everyone. 
      <br>
    - <span style='color:darkred;'>**>250.5 µg/m³ (Hazardous)**</span>: Health warnings; emergency conditions. 
      <br>
    <br>For more information, visit [here](https://aqicn.org/faq/2013-09-09/revised-pm25-aqi-breakpoints/).
    """, unsafe_allow_html=True)
    # st.markdown("""
    # <h3 style='text-align: center;'>Spatial Correlation Map</h3>
    # """, unsafe_allow_html=True)
    # st.pydeck_chart(corr_deck)
    # # Download filtered data as CSV
    csv_data = main_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv_data,
        file_name=f"PM2.5_{s}_{selected_month}_{selected_year}.csv",
        mime='text/csv',
    )


# # Calculate RMSE, MAE, and R² score between observed and selected model predictions (only for valid entries)
# valid_mask = ~np.isnan(observed_grid) & ~np.isnan(model_grid)
# rmse = np.sqrt(mean_squared_error(
#     observed_grid[valid_mask], model_grid[valid_mask]))
# mae = mean_absolute_error(observed_grid[valid_mask], model_grid[valid_mask])
# r2 = r2_score(observed_grid[valid_mask], model_grid[valid_mask])

# # Display the metrics
# st.write(f"### Model: {selected_model}")
# st.write(f"**Root Mean Square Error (RMSE):** {rmse:.2f}")
# st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
# st.write(f"**R² Score:** {r2:.2f}")

# # Option to download filtered data
# @st.cache_data
# def convert_df_to_csv(df):
#     return df.to_csv(index=False).encode('utf-8')


# csv = convert_df_to_csv(filtered_geo_data)

# st.download_button(
#     label="Download filtered data as CSV",
#     data=csv,
#     file_name='filtered_data.csv',
#     mime='text/csv',
# )
