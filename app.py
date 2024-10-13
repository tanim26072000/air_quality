import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopandas.tools import sjoin
import pydeck as pdk
import os

# Title
st.title("Interactive PM\u2082\u002E\u2085 Data with spatio-temporal Selection")
gdf = gpd.read_file(
    'bgd_adm_bbs_20201113_shp/bgd_admbnda_adm3_bbs_20201113.shp')
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
        'Select a year', list(range(2000,2020)), key='select_year')
        
# selected_date = st.date_input("Select Date", value=datetime.date(
#     2010, 1, 1), min_value=datetime.date(2000, 1, 1), max_value=datetime.date(2019, 12, 31))
# # Extract the year and month from the selected date
# selected_year = selected_date.year
# selected_month = selected_date.month
# Your climate data h5 file
data = pd.read_hdf(
    'pm25_final.h5', key=f'{selected_month}_{selected_year}')

corr_df = pd.read_csv('corr_df.csv')
# Convert lat/lon data points into a GeoDataFrame
geometry = [Point(xy) for xy in zip(data['longitude'], data['latitude'])]
geo_data = gpd.GeoDataFrame(data, crs="EPSG:4326", geometry=geometry)
geo_corr = gpd.GeoDataFrame(corr_df, crs="EPSG:4326", geometry=geometry)

# Dropdown to select the scope (Bangladesh, Division, District)
scope = st.radio(
    "Select Map Scope",
    ('Whole Bangladesh', 'Division', 'District'), horizontal=True
)

if scope == 'Whole Bangladesh':
    # No need for spatial join, use the entire filtered data for Bangladesh
    filtered_geo_data = sjoin(
        geo_data, gdf, how="inner", predicate="within")
    filtered_corr = sjoin(
        geo_corr, gdf, how="inner", predicate="within")
    s = 'Bangladesh'
    z = 5.5

elif scope == 'Division':
    # Select a division
    division_options = list(gdf['ADM1_EN'].unique())
    division_options.sort()
    selected_division = st.selectbox(
        'Select Division', division_options, key='select_division')
    s = f'{selected_division}'
    z = 7.5

    # Filter the division boundary from the shapefile
    selected_division_shape = gdf[gdf['ADM1_EN']
                                  == selected_division]

    # Spatial join to filter climate data points that fall within the selected division
    filtered_geo_data = sjoin(
        geo_data, selected_division_shape, how="inner", predicate="within")
    filtered_corr = sjoin(
        geo_corr, selected_division_shape, how="inner", predicate="within")

elif scope == 'District':
    # Select a district
    district_options = list(gdf['ADM2_EN'].unique())
    district_options.sort()
    selected_district = st.selectbox(
        'Select District', district_options, key='select_district')
    s = f'{selected_district}'
    z = 8

    # Filter the district boundary from the shapefile
    selected_district_shape = gdf[gdf['ADM2_EN']
                                  == selected_district]

    # Spatial join to filter climate data points that fall within the selected district
    filtered_geo_data = sjoin(
        geo_data, selected_district_shape, how="inner", predicate="within")
    filtered_corr = sjoin(
        geo_corr, selected_district_shape, how="inner", predicate="within")

# Input to select the model for comparison
selected_model = st.radio('Select Model for Comparison', [
    'GNN+LSTM', 'GNN', 'CNN+LSTM', 'CNN'], key='select_model', horizontal=True)
selected_model= selected_model.lower()
df = filtered_geo_data[['latitude', 'longitude', 'observed',
                        'gnn+lstm', 'gnn', 'cnn+lstm', 'cnn', 'ADM1_EN', 'ADM2_EN', 'ADM3_EN']]
corr = filtered_corr[['latitude', 'longitude',
                     'gnn+lstm', 'gnn', 'cnn+lstm', 'cnn', 'ADM1_EN', 'ADM2_EN', 'ADM3_EN']]
# corr[['gnn+lstm', 'gnn', 'cnn+lstm', 'cnn']
#      ] = corr[['gnn+lstm', 'gnn', 'cnn+lstm', 'cnn']]*10000
corr = corr.assign(scaled_elevation=corr[selected_model] * 5000)
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
    get_position='[longitude, latitude]',  # Specify the center of each grid cell
    cell_size=1000,  # Size of the grid cells (in meters)
    get_elevation='observed',  # Elevation represents the observed value
    get_fill_color=f'[255, observed*2, 100,128]',  # Color based on observed value
    pickable=True,
    extruded=True,  # Extrude to give a 3D effect
    elevation_scale=100  # Scale of the elevation
)

    # Step 2: Create a PyDeck GridCellLayer for predicted values (GNN-LSTM)
    predicted_layer = pdk.Layer(
        "GridCellLayer",
        df,
        get_position='[longitude, latitude]',  # Specify the center of each grid cell
        cell_size=1000,  # Size of the grid cells (in meters)
        get_elevation=selected_model,  # Elevation represents the predicted value
        get_fill_color=f'[255, {selected_model}*2, 100, 128]',  # Color based on predicted value
        pickable=True,
        extruded=True,  # Extrude to give a 3D effect
        elevation_scale=100  # Scale of the elevation
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
        elevation_scale=1 # Scale of the elevation
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
        layers=[observed_layer, predicted_layer],
        initial_view_state=view_state,
        map_style=f'mapbox://styles/mapbox/streets-v12',
        tooltip={
            "html": f"<b>Latitude:</b> {{latitude}} <br><b>Longitude:</b> {{longitude}}<br>"
            f"<b>Division:</b> {{ADM1_EN}}<br><b>District:</b> {{ADM2_EN}}<br>"
            f"<b>Upazila:</b> {{ADM3_EN}}<br>"
            f"<b>Observed:</b> <span style='color:yellow;'>{{observed}}</span><br>"
            f"<b>Predicted:</b> {{{selected_model}}}",
            "style": {"color": "white"}
        }
    )

    predicted_deck = pdk.Deck(
        layers=[predicted_layer],
        initial_view_state=view_state,
        map_style=f'mapbox://styles/mapbox/streets-v12',
        tooltip={
            "html": f"<b>Latitude:</b> {{latitude}} <br><b>Longitude:</b> {{longitude}}<br>"
            f"<b>Division:</b> {{ADM1_EN}}<br><b>District:</b> {{ADM2_EN}}<br>"
            f"<b>Upazila:</b> {{ADM3_EN}}<br>"
            f"<b>Observed:</b> {{observed}}<br>"
            f"<b>Predicted:</b> <span style='color:yellow;'>{{{selected_model}}}</span>",
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
    st.title(f"{s} at {selected_month}-{selected_year}")

    # Columns for the maps
    col1, col2 = st.columns(2)

    # Conditionally display the maps based on checkbox selections
    with col1:
            st.subheader("Observed Data Map")
            st.pydeck_chart(observed_deck)

    with col2:
            st.subheader("Predicted Data Map")
            st.pydeck_chart(predicted_deck)
    st.write("""
    ### PM$_{2.5}$ Concentration Ranges and Health Impacts:
    - **0-12 µg/m³ (Good)**: Air quality is considered satisfactory, posing little or no health risk.
    - **12.1-35.4 µg/m³ (Moderate)**: Air quality is acceptable, but sensitive individuals may experience health issues.
    - **35.5-55.4 µg/m³ (Unhealthy for Sensitive Groups)**: Sensitive groups may experience health effects.
    - **55.5-150.4 µg/m³ (Unhealthy)**: Everyone may experience health effects.
    - **150.5-250.4 µg/m³ (Very Unhealthy)**: Severe health effects for everyone.
    - **>250.5 µg/m³ (Hazardous)**: Health warnings; emergency conditions.
    """)
    st.subheader("Spatial Correlation Map")
    st.pydeck_chart(corr_deck)
        # Download filtered data as CSV
    csv_data = df.to_csv(index=False).encode('utf-8')

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
