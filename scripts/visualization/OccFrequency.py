import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

# Load the data
station_info = pd.read_csv("datasets-v2/inf.csv")  # Adjust path as needed
occupancy_data = pd.read_csv("datasets-v2/occupancy.csv")

# Define the threshold for "high utilization" as 80% of pile capacity
threshold_percentage = 70

# Reshape occupancy_data for easier merging (long format)
occupancy_long = occupancy_data.melt(id_vars=['time'], var_name='station_id', value_name='occupancy')

# Convert station_id to integer to match the info file
occupancy_long['station_id'] = occupancy_long['station_id'].astype(int)

# Merge occupancy data with station info to include pile_count
merged_data = pd.merge(occupancy_long, station_info, on='station_id')

# Calculate the high utilization threshold for each station
merged_data['high_utilization'] = (merged_data['occupancy'] >= (merged_data['pile_count'] * threshold_percentage / 100)).astype(int)

# Group by station_id to calculate high utilization frequency
utilization_freq = merged_data.groupby('station_id')['high_utilization'].mean() * 100

# Merge utilization frequency back with station_info
station_summary = station_info.copy()
station_summary = station_summary.merge(utilization_freq.rename('utilization_freq'), on='station_id')

# Save to CSV if needed
station_summary.to_csv("station_summary_with_utilization.csv", index=False)




# Plot station heatmap

# Step 1: Load the Shenzhen shapefile
shenzhen_map = gpd.read_file("datasets/SZ_districts/SZ_districts.shp")



# Step 3: Convert station data into a GeoDataFrame
station_gdf = gpd.GeoDataFrame(
    station_info,
    geometry=gpd.points_from_xy(station_info['longitude'], station_info['latitude']),
    crs="EPSG:4326"  # Assuming WGS84
)

shenzhen_map = shenzhen_map.to_crs(station_gdf.crs)

# Step 4: Prepare data for heatmap (latitude, longitude, and weights)
x = station_info['longitude']
y = station_info['latitude']
weights = station_summary['utilization_freq']

# Calculate KDE for the heatmap
kde = gaussian_kde([x, y], weights=weights)
xi, yi = np.meshgrid(
    np.linspace(x.min(), x.max(), 500),
    np.linspace(y.min(), y.max(), 500)
)
zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)

# Step 5: Plot the map and heatmap
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the Shenzhen shapefile
shenzhen_map.plot(ax=ax, color='white', edgecolor='black', alpha=0.2)

# Plot the heatmap
ax.imshow(zi, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='Blues', alpha=0.9)

# Overlay station points
# station_gdf.plot(ax=ax, color='black', markersize=2, alpha=0.7)

ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())

# Add titles and labels
plt.title("Heatmap of Charging Station Utilization", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar(plt.cm.ScalarMappable(cmap='Blues'), label="Utilization Density")
plt.show()

