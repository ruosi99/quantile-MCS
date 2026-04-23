import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import MultiPoint

# Step 1: Load Shenzhen map
shenzhen_map = gpd.read_file("datasets/SZ_districts/SZ_districts.shp")

# Step 2: Load station data with utilization frequency
station_data = pd.read_csv('station_summary_with_utilization.csv')

# Step 3: Extract coordinates and utilization frequency
coordinates = station_data[['longitude', 'latitude']].values
utilization = station_data['utilization_freq'].values

# Step 4: Run KMeans clustering to group stations into busy areas
num_clusters = 20  # Adjust based on the number of busy areas in your heatmap
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

# Step 7: Plot the Shenzhen map with clustering results
fig, ax = plt.subplots(figsize=(12, 10))

# Plot the Shenzhen map
shenzhen_map.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.5)

# Plot stations in other clusters
station_gdf[station_gdf['is_busiest'] == False].plot(
    ax=ax,
    column='cluster',
    cmap='viridis',
    markersize=50,
    alpha=0.7,
    legend=True
)

# Highlight the busiest cluster
station_gdf[station_gdf['is_busiest'] == True].plot(
    ax=ax,
    color='red',
    markersize=100,
    alpha=0.9,
    label='Busiest Cluster'
)

# Add titles, legend, and labels
plt.title("Station Clusters with Density-Based Busiest Cluster Highlighted", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()


station_gdf[station_gdf['is_busiest'] == True]
busiest_stations = station_gdf[station_gdf['is_busiest'] == True]
busiest_stations.drop(columns=['geometry']).to_csv("busiest_cluster_stations.csv", index=False)


# Perform spatial join to assign districts
stations_with_districts = gpd.sjoin(busiest_stations, shenzhen_map, how='left', op='within')

# Inspect the results
print(stations_with_districts.head())

# Save the results to a CSV file
stations_with_districts.drop(columns=['geometry']).to_csv("busiest_cluster_with_districts.csv", index=False)

# Group by district and calculate metrics
district_summary = stations_with_districts.groupby('OBJECTID').agg(
    total_utilization=('utilization_freq', 'sum'),
    avg_utilization=('utilization_freq', 'mean'),
    station_count=('station_id', 'size')
)

# Save the district summary
district_summary.to_csv("busiest_cluster_district_summary.csv")

# Filter districts that contain stations from the busiest cluster
busiest_districts = shenzhen_map[shenzhen_map['OBJECTID'].isin(stations_with_districts['OBJECTID'])]

# Plot the districts and stations
fig, ax = plt.subplots(figsize=(12, 10))
shenzhen_map.plot(ax=ax, color='lightgrey', edgecolor='black', alpha=0.5)
busiest_districts.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.7)
#busiest_stations.plot(ax=ax, color='red', markersize=100, label='Stations in Busiest Cluster', alpha=0.5)
plt.title("Districts Containing the Busiest Cluster")
plt.legend()
plt.show()

# Plot only the selected districts
fig, ax = plt.subplots(figsize=(8, 10))

# Plot the selected districts
busiest_districts.plot(ax=ax, color='steelblue', edgecolor='black')

# Customize the plot
ax.set_title("Selected Districts", fontsize=16)
ax.axis('off')  # Remove axis for a cleaner map
plt.show()