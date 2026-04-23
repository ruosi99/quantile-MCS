import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point
import pandas as pd
import matplotlib.pyplot as plt
# Step 1: Load district shapefile
districts = gpd.read_file("datasets/SZ_districts/SZ_districts.shp")

# Step 2: Load station data
station_data = pd.read_csv("station_summary_with_utilization.csv")  # Replace with your file path
station_gdf = GeoDataFrame(
    station_data,
    geometry=gpd.points_from_xy(station_data['longitude'], station_data['latitude']),
    crs=districts.crs
)

print(station_gdf.head())  # Look at the geometry column

# First, define the correct CRS as EPSG:4326
station_gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)

# Then, reproject to EPSG:3857
station_gdf = station_gdf.to_crs("EPSG:3857")


print(districts.crs)  # CRS of the district shapefile
print(station_gdf.crs)  # CRS of the station data

print("District bounds:", districts.total_bounds)  # [minx, miny, maxx, maxy]
print("Station bounds:", station_gdf.total_bounds)  # [minx, miny, maxx, maxy]

# Step 3: Spatial join to assign stations to districts
stations_with_districts = gpd.sjoin(station_gdf, districts, how='left', op='within')

# Step 4: Check results
print(stations_with_districts.head())
print(stations_with_districts.columns)

stations_with_districts.drop(columns=['geometry'], inplace=True)  # Drop spatial data
stations_with_districts.to_csv("stations_with_districts.csv", index=False)  # Save to CSV


unmatched_stations = stations_with_districts[stations_with_districts['OBJECTID'].isnull()]
print(unmatched_stations[['longitude', 'latitude']])
fig, ax = plt.subplots(figsize=(10, 8))
districts.plot(ax=ax, color='lightgrey', edgecolor='black')
unmatched_stations.plot(ax=ax, color='red', markersize=10)

plt.title("Unmatched Stations")
plt.show()
