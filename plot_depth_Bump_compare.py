import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# File paths
wl_file = os.path.dirname(os.path.abspath(__file__)) + "/WL/WL9.7.csv"
validation_file_1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "validation/WL9.7measured.csv"))
validation_file_2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "validation/WaterLevel9p7_SWFCA.csv"))  # SWFCA validation data

# Read CSV files
wl_data = pd.read_csv(wl_file)
validation_data_1 = pd.read_csv(validation_file_1)
validation_data_2 = pd.read_csv(validation_file_2)

# Extract relevant columns (assuming zero-based index)
wl_distance = wl_data.iloc[:, 0] / 10  # Scale WL distance by 10
wl_elevation = wl_data.iloc[:, 1]  # Elevation
wl_depth = wl_data.iloc[:, 3]  # Depth

val_distance_1 = validation_data_1.iloc[:, 0]  # Validation 1 distance
val_depth_1 = validation_data_1.iloc[:, 1]  # Validation 1 depth

val_distance_2 = validation_data_2.iloc[:, 0]  # Validation 2 distance
val_depth_2 = validation_data_2.iloc[:, 1]  # Validation 2 depth

# Remove zero depth values
wl_mask = wl_depth > 0
val_mask_1 = val_depth_1 > 0
val_mask_2 = val_depth_2 > 0

wl_distance_filtered = wl_distance[wl_mask]
wl_depth_filtered = wl_depth[wl_mask]

val_distance_filtered_1 = val_distance_1[val_mask_1]
val_depth_filtered_1 = val_depth_1[val_mask_1]

val_distance_filtered_2 = val_distance_2[val_mask_2]
val_depth_filtered_2 = val_depth_2[val_mask_2]

# Plot
plt.figure(figsize=(12, 6))

# Plot solid continuous lines for measured depth and elevation
plt.plot(wl_distance_filtered, wl_depth_filtered, 'b-', label="CADFFM", linewidth=2)
plt.plot(wl_distance, wl_elevation, 'g-', label="Elevation", linewidth=2)

# Plot validation depths
plt.plot(val_distance_filtered_1, val_depth_filtered_1, 'ro', markerfacecolor='none', label="Measured", markersize=6)  # Red empty circles
plt.plot(val_distance_filtered_2, val_depth_filtered_2, 'b--', alpha=0.5, label="SWFCA", linewidth=2)  # Blue dashed line with 50% transparency

# Increase font size
plt.xlabel("Distance (m)", fontsize=20)
plt.ylabel("Depth / Elevation (m)", fontsize=20)
# plt.title("Water Level at timestep 9.7s", fontsize=20)

# Move legend to the top-right corner
plt.legend(fontsize=16, loc="upper right")

# Increase tick font size
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Save plot as PNG with the legend in the right top corner
# plt.savefig("Water_Level_9.7s.png", dpi=300, bbox_inches="tight")

# Show plot
plt.show()
