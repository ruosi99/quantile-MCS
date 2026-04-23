import geopandas as gpd
from PIL import Image

import matplotlib.pyplot as plt
import pandas as pd

# Load geographic and charging pile data
districts = gpd.read_file("datasets/SZ_districts/SZ_districts.shp")  # replace with actual file path

info = pd.read_csv("datasets/information.csv")


districts = districts.merge(info, left_on='OBJECTID', right_on='num')

# Step 4: Filter CBD areas (CBD == 1)
cbd_areas = districts[districts['CBD'] == 1]

# Step 5: Plot the CBD areas
fig, ax = plt.subplots(figsize=(10, 8))
cbd_areas.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.8)
plt.title("CBD Areas in Shenzhen")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()