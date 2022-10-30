from urllib import request
import streamlit as st
import pandas as pd
from scipy import spatial
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import pyproj
from streamlit_lottie import st_lottie
import requests
import plotly.express as px


st.set_page_config(
    page_title="Khalil Al Hooti",
    page_icon=":smiley:",
    layout="wide",
)

# Path: app.py


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
            content:'Created by Khalil Al Hooti';
            visibility: visible;
            display: block;
            position: relative;
            # background-color: red;
            padding: 5px;
            top: 2px;
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

plotly_config = {
    'modeBarButtonsToRemove': ['sendDataToCloud',
                               'lasso2d',
                               'select2d',
                               'hoverClosestCartesian',
                               'hoverCompareCartesian',
                               'toggleSpikelines'],
    'displaylogo': False,
}

P = pyproj.Proj(proj='utm', zone=40, ellps='WGS84', preserve_units=True)
G = pyproj.Proj(init='epsg:4326')


def utm_to_latlon(utm_x, utm_y):
    return pyproj.Transformer.from_proj(P, G).transform(utm_x, utm_y)


def latlon_to_utm(lat, lon):
    return pyproj.Transformer.from_proj(G, P).transform(lat, lon)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url = "https://assets2.lottiefiles.com/packages/lf20_cmaqoazd.json"
lottie_json = load_lottieurl(lottie_url)

lottie_done_url = "https://assets2.lottiefiles.com/private_files/lf30_z1sghrbu.json"
lottie_done_json = load_lottieurl(lottie_done_url)

st.title("Find nearest neighbors points in 2D space between two dataframes")

grid_elevation_points = st.file_uploader(
    "DEM points x, y, z coordinates with no column names", type=["xlsx", "txt", "csv"],
    help="Upload grid elevation points of three columns containing x-coordinate, y-coordinate and elavation")

profiles_points = st.file_uploader(
    "Profile points x, y, coordinates with no column names", type=["xlsx", "txt", "csv"],
    help="Upload profile points of two columns containing x-coordinate and y-coordinate")


@st.cache(allow_output_mutation=True, show_spinner=False)
def find_nearest_neighbor(df1, df2):
    """Find nearest neighbor points between two dataframes

    Args:
        df1 (DataFrame): First dataframe
        df2 (DataFrame): Second dataframe

    Returns:
        DataFrame: Nearest neighbor points
    """
    df = pd.DataFrame(columns=['x_prof',
                               'y_prof',
                               'lon_prof',
                               'lat_prof',
                               'distance_difference',
                               "x_dem",
                               "y_dem",
                               "z",
                               "lon_dem",
                               "lat_dem"]
                      )
    tree = spatial.KDTree(df1[['x', 'y']])
    for i, row in df2.iterrows():
        x, y, lon, lat = row['x'], row['y'], row['lon'], row['lat']
        dist, idx = tree.query([(x, y)])
        x_dem, y_dem, z, lon_dem, lat_dem = dme.iloc[idx].values[0]
        df.loc[i] = [x, y, lon, lat, dist[0],
                     x_dem, y_dem, z, lon_dem, lat_dem]

    return df


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


if grid_elevation_points and profiles_points:

    try:
        try:
            dme = pd.read_excel(grid_elevation_points, header=None,
                                names=['x', 'y', 'z'])

            dmp = pd.read_excel(profiles_points, header=None,
                                names=['x', 'y'])

        except:
            try:
                dme = pd.read_csv(grid_elevation_points,
                                  header=None, delimiter="\t",
                                  skipinitialspace=True,
                                  names=['x', 'y', 'z'])

                dmp = pd.read_csv(profiles_points,
                                  header=None, delimiter="\t",
                                  skipinitialspace=True,
                                  names=['x', 'y'])
            except:
                try:
                    dme = pd.read_csv(grid_elevation_points,
                                      header=None, delimiter=",",
                                      skipinitialspace=True,
                                      names=['x', 'y', 'z'])

                    dmp = pd.read_csv(profiles_points,
                                      header=None, delimiter=",",
                                      skipinitialspace=True,
                                      names=['x', 'y'])

                except:
                    dme = pd.read_csv(grid_elevation_points,
                                      header=None, delimiter=" ",
                                      skipinitialspace=True,
                                      names=['x', 'y', 'z'])

                    dmp = pd.read_csv(profiles_points,
                                      header=None, delimiter=" ",
                                      skipinitialspace=True,
                                      names=['x', 'y'])

        dme['lon'], dme['lat'] = utm_to_latlon(dme['x'], dme['y'])
        dmp['lon'], dmp['lat'] = utm_to_latlon(dmp['x'], dmp['y'])

        st.write("DEM points")
        st.write(dme)
        # convert dme and dmp to lat lon

        # st.map(dme[['lat', 'lon']])

        st.write("Profile points")
        st.write(dmp)
        # st.map(dmp[['lat', 'lon']])

        empty = st.empty()
        with empty.container():

            st.write("Finding nearest neighbor points...please wait\n"
                     "This may take a while depending on the size of the data")
            st_lottie(lottie_json, height=300, width=300)

        df = find_nearest_neighbor(dme, dmp)

        empty.empty()

        with empty.container():

            st_lottie(lottie_done_json, height=300, width=300)
            st.write("Done!. You can download the result below")

        st.write("Nearest neighbor points")
        st.write(df)

        csv = convert_df(df)

        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )

        fig = px.scatter_3d(df, x='x_prof', y='y_prof', z='z',
                            color='distance_difference',
                            hover_data=['lon_prof', 'lat_prof',
                                        'lon_dem', 'lat_dem'],
                            title="Nearest neighbor points in 3D space",
                            )

        fig.update_layout(scene=dict(
            xaxis_title='x-coordinate',
            yaxis_title='y-coordinate',
            zaxis_title='elevation'
        ),
            margin=dict(l=0, r=0, b=0, t=0),
            width=800, height=800,
        )

        st.plotly_chart(fig, use_container_width=True, config=plotly_config)

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection=ccrs.PlateCarree())
        ax.set_extent([df['lon_prof'].min(), df['lon_prof'].max(),
                       df['lat_prof'].min(), df['lat_prof'].max()],
                      crs=ccrs.PlateCarree())
        ax.coastlines(resolution='10m', color='black', linewidth=1)

        scatter = ax.scatter(df['lon_prof'], df['lat_prof'], c=df['distance_difference'], s=1,
                             transform=ccrs.PlateCarree(),
                             label='Profile points',
                             cmap='viridis')

        show_dme = st.checkbox("Show DEM points", value=False)
        increase_res = st.slider(
            "Google map resolution. Increase the resolution of the google map.\nVery slow for large data",
            min_value=10,
            max_value=20,
            step=1,
            value=15,
            help="Increase the resolution of the google map.\nVery slow for large data")

        if show_dme:

            scatter2 = ax.scatter(dme['lon'], dme['lat'], c='k', s=0.0005,
                                  transform=ccrs.PlateCarree(), label='DEM points')

        fig2.colorbar(scatter, ax=ax, label='Distance difference (m)')

        request = cimgt.GoogleTiles(style="satellite")

        ax.add_image(request, increase_res, interpolation='spline36')
        gl = ax.gridlines(draw_labels=True, alpha=0.2)
        gl.xlabels_top = gl.ylabels_right = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER

        # ax.legend(loc='upper left', fontsize=10)

        st.pyplot(fig2)

    except Exception as e:
        st.write(e)
