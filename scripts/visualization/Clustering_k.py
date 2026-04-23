import pandas as pd
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt
from geopy.distance import geodesic

# Load Shenzhen map
shenzhen_map = gpd.read_file("datasets/SZ_districts/SZ_districts.shp")

# Load station data with utilization frequency
station_data = pd.read_csv('station_summary_with_utilization.csv')

# Extract coordinates and utilization frequency
coordinates = station_data[['longitude', 'latitude']].values

# Step 1: Run KMeans clustering to group stations into busy areas
num_clusters = 20  # Adjust based on the Elbow Method
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
station_data['cluster'] = kmeans.fit_predict(coordinates)


# Step 5: Calculate density for each cluster
densities = []
for cluster_id, group in station_data.groupby('cluster'):
    points = MultiPoint(group[['longitude', 'latitude']].values)
    if points.is_empty or points.convex_hull.area == 0:
        density = 0  # Avoid division by zero for clusters with no valid area
    else:
        density = group['utilization_freq'].sum() / points.convex_hull.area
    densities.append((cluster_id, density))

# Create a DataFrame to store cluster density information
density_df = pd.DataFrame(densities, columns=['cluster', 'density'])

# Identify the busiest cluster based on density
busiest_cluster = density_df.loc[density_df['density'].idxmax(), 'cluster']
print("Busiest cluster:", busiest_cluster)

# Step 6: Convert station data to GeoDataFrame for visualization
station_gdf = gpd.GeoDataFrame(
    station_data,
    geometry=gpd.points_from_xy(station_data['longitude'], station_data['latitude']),
    crs="EPSG:4326"  # Assuming station data is in WGS84
)

# Align Shenzhen map CRS to station data CRS if necessary
if shenzhen_map.crs != station_gdf.crs:
    shenzhen_map = shenzhen_map.to_crs(station_gdf.crs)

# Separate the busiest cluster from other clusters
station_gdf['is_busiest'] = station_gdf['cluster'] == busiest_cluster




# Step 2: Calculate density for each cluster
densities = []
for cluster_id, group in station_data.groupby('cluster'):
    points = MultiPoint(group[['longitude', 'latitude']].values)
    if points.is_empty or points.convex_hull.area == 0:
        density = 0  # Avoid division by zero
    else:
        density = group['utilization_freq'].sum() / points.convex_hull.area
    densities.append((cluster_id, density))

# Create a DataFrame for cluster density information
density_df = pd.DataFrame(densities, columns=['cluster', 'density'])

# Identify the busiest cluster based on density
busiest_cluster = density_df.loc[density_df['density'].idxmax(), 'cluster']
station_data['is_busiest'] = station_data['cluster'] == busiest_cluster

# Convert station data to GeoDataFrame for spatial operations
station_gdf = gpd.GeoDataFrame(
    station_data,
    geometry=gpd.points_from_xy(station_data['longitude'], station_data['latitude']),
    crs="EPSG:4326"  # Assuming station data is in WGS84
)

# Align Shenzhen map CRS to station data CRS if necessary
if shenzhen_map.crs != station_gdf.crs:
    shenzhen_map = shenzhen_map.to_crs(station_gdf.crs)

# Filter stations in the busiest cluster
busiest_stations = station_gdf[station_gdf['is_busiest'] == True]
busiest_stations.drop(columns=['geometry']).to_csv("busiest_cluster_stations.csv", index=False)

# Step 3: Perform spatial join to link stations to districts
stations_with_districts = gpd.sjoin(busiest_stations, shenzhen_map, how='left', op='within')
stations_with_districts.drop(columns=['geometry']).to_csv("busiest_cluster_with_districts.csv", index=False)

# Step 4: Generate district summary
district_summary = stations_with_districts.groupby('OBJECTID').agg(
    total_utilization=('utilization_freq', 'sum'),
    avg_utilization=('utilization_freq', 'mean'),
    station_count=('station_id', 'size')
)
district_summary.to_csv("busiest_cluster_district_summary.csv", index=False)

# Step 5: Compute distances between stations (optional for routing simulation)
station_pairs = []
for i, row_i in busiest_stations.iterrows():
    for j, row_j in busiest_stations.iterrows():
        if i != j:
            dist = geodesic((row_i['latitude'], row_i['longitude']), (row_j['latitude'], row_j['longitude'])).meters
            station_pairs.append({'station_id_1': row_i['station_id'], 'station_id_2': row_j['station_id'], 'distance': dist})
station_distances = pd.DataFrame(station_pairs)
station_distances.to_csv("busiest_cluster_station_distances.csv", index=False)

# Step 6: Visualization of Districts with Station Markers
# fig, ax = plt.subplots(figsize=(10, 12))
#
# # Plot the selected districts
# busiest_districts = shenzhen_map[shenzhen_map['OBJECTID'].isin(stations_with_districts['OBJECTID'])]
#
# busiest_districts.plot(ax=ax, color='steelblue', edgecolor='black', alpha=0.7, label='Selected Districts')
#
# # Plot the station locations as markers
# busiest_stations.plot(ax=ax, color='red', markersize=20, alpha=0.8, label='Stations')
#
# # Add titles, legend, and labels
# ax.set_title("Selected Districts with Station Locations", fontsize=16)
# ax.set_xlabel("Longitude")
# ax.set_ylabel("Latitude")
# ax.legend()
#
# plt.show()


# Step 6: Visualization of Districts with Station Markers and District IDs
fig, ax = plt.subplots(figsize=(10, 12))

# Plot the selected districts
busiest_districts = shenzhen_map[shenzhen_map['OBJECTID'].isin(stations_with_districts['OBJECTID'])]

busiest_districts.plot(ax=ax, color='steelblue', edgecolor='black', alpha=0.7, label='Selected Districts')

# Plot the station locations as markers
busiest_stations.plot(ax=ax, color='red', markersize=50, alpha=0.8, label='Stations')

# Add district IDs as labels
for x, y, label in zip(busiest_districts.geometry.centroid.x,
                       busiest_districts.geometry.centroid.y,
                       busiest_districts['OBJECTID']):
    ax.text(x, y, str(label), fontsize=10, ha='center', color='white', bbox=dict(facecolor='black', alpha=0.5))

# Add titles, legend, and labels
ax.set_title("Selected Districts with Station Locations and IDs", fontsize=16)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()

plt.show()
