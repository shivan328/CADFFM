import os
import pandas as pd
import matplotlib.pyplot as plt

# Define folder paths
wl_folder = os.path.dirname(os.path.abspath(__file__)) + "/WL/"  # Simulated data folder
validation_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "validation"))  # Measured data folder

# Get list of available files in both folders
wl_files = sorted([f for f in os.listdir(wl_folder) if f.startswith("WL") and f.endswith(".csv")])
validation_files = sorted([f for f in os.listdir(validation_folder) if f.startswith("C_") and f.endswith(".csv")])

# Extract time steps from file names
wl_timesteps = {float(f[2:-4]): f for f in wl_files}  
validation_timesteps = {float(f[2:-5]): f for f in validation_files}  

# Find matching time steps
matching_timesteps = sorted(set(wl_timesteps.keys()) & set(validation_timesteps.keys()))

# If no matching files found, exit
if not matching_timesteps:
    print("No matching time steps found between simulated and measured data.")
    exit()

# Initialize limits
depth_min, depth_max = float('inf'), float('-inf')
velocity_min, velocity_max = float('inf'), float('-inf')

# First loop to find min and max values across all files
for timestep in matching_timesteps:
    wl_data = pd.read_csv(os.path.join(wl_folder, wl_timesteps[timestep]))
    validation_data = pd.read_csv(os.path.join(validation_folder, validation_timesteps[timestep]))

    wl_data.iloc[:, 0] = wl_data.iloc[:, 0] / 10  

    # Extract data
    y_wl_depth = wl_data.iloc[:, 2]  
    y_wl_velocity = wl_data.iloc[:, 5]  
    y_val_depth = validation_data.iloc[:, 1]  
    y_val_velocity = validation_data.iloc[:, 2]  

    valid_wl_depth = y_wl_depth[y_wl_depth > 0]
    valid_wl_velocity = y_wl_velocity[y_wl_velocity > 0]

    # Update min and max values
    depth_min = min(depth_min, y_val_depth.min(), valid_wl_depth.min() if not valid_wl_depth.empty else depth_min)
    depth_max = max(depth_max, y_val_depth.max(), valid_wl_depth.max() if not valid_wl_depth.empty else depth_max)
    velocity_min = min(velocity_min, y_val_velocity.min(), valid_wl_velocity.min() if not valid_wl_velocity.empty else velocity_min)
    velocity_max = max(velocity_max, y_val_velocity.max(), valid_wl_velocity.max() if not valid_wl_velocity.empty else velocity_max)

# Add margin to avoid cutting off data
depth_margin = (depth_max - depth_min) * 0.1
velocity_margin = (velocity_max - velocity_min) * 0.1
depth_min -= depth_margin
depth_max += depth_margin
velocity_min -= velocity_margin
velocity_max += velocity_margin

# Plot settings
num_plots = len(matching_timesteps)
fig, axes = plt.subplots(num_plots, 1, figsize=(10, 4 * num_plots), sharex=True)

# Create empty lists to store legend handles
handles_depth, labels_depth = [], []
handles_velocity, labels_velocity = [], []

# Second loop to generate plots
for i, timestep in enumerate(matching_timesteps):
    wl_data = pd.read_csv(os.path.join(wl_folder, wl_timesteps[timestep]))
    validation_data = pd.read_csv(os.path.join(validation_folder, validation_timesteps[timestep]))

    wl_data.iloc[:, 0] = wl_data.iloc[:, 0] / 10  

    # Extract data
    x_wl = wl_data.iloc[:, 0]  
    y_wl_depth = wl_data.iloc[:, 2]  
    y_wl_velocity = wl_data.iloc[:, 5]  
    x_val = validation_data.iloc[:, 0]  
    y_val_depth = validation_data.iloc[:, 1]  
    y_val_velocity = validation_data.iloc[:, 2]  

    valid_wl_depth = y_wl_depth > 0
    valid_wl_velocity = y_wl_velocity > 0

    # Plot
    ax = axes[i] if num_plots > 1 else axes
    depth_line_sim, = ax.plot(x_wl[valid_wl_depth], y_wl_depth[valid_wl_depth], linestyle="-", color="blue", label="Simulated Depth", linewidth=2)
    depth_line_meas, = ax.plot(x_val, y_val_depth, 'o', markerfacecolor="none", markeredgecolor="blue", alpha=0.5, label="Measured Depth", markersize=8)

    ax2 = ax.twinx()
    velocity_line_sim, = ax2.plot(x_wl[valid_wl_velocity], y_wl_velocity[valid_wl_velocity], linestyle="-", color="red", label="Simulated Velocity", linewidth=2)
    velocity_line_meas, = ax2.plot(x_val, y_val_velocity, 'o', markerfacecolor="none", markeredgecolor="red", alpha=0.5, label="Measured Velocity", markersize=8)

    if i == 0:  
        handles_depth.extend([depth_line_sim, depth_line_meas])
        labels_depth.extend(["Simulated Depth", "Measured Depth"])
        handles_velocity.extend([velocity_line_sim, velocity_line_meas])
        labels_velocity.extend(["Simulated Velocity", "Measured Velocity"])

    ax.set_ylim(depth_min, depth_max)
    ax2.set_ylim(velocity_min, velocity_max)

    ax.set_title(f"Time Step: {timestep}s", fontsize=20)
    ax.set_ylabel("Depth (m)", fontsize=20)

    # **Make velocity y-axis title red**
    ax2.set_ylabel("Velocity (m/s)", fontsize=20, color="red")

    # Set tick font sizes and increase tick box size
    ax.tick_params(axis='both', labelsize=20, width=2, length=6, pad=10)  
    ax2.tick_params(axis='both', labelsize=20, width=2, length=6, pad=10, colors="red")  

# Set x-axis label only for the last subplot
axes[-1].set_xlabel("Distance (m)", fontsize=20)

# Create unified legend at the bottom
fig.legend(handles_depth + handles_velocity, labels_depth + labels_velocity, 
           loc="lower center", ncol=2, fontsize=18, frameon=False)


plt.tight_layout(rect=[0, 0.05, 1, 1])  

# Save the figure
plot_filename = "comparison_plot.png"
plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
plt.show()

print(f"Plot saved as {plot_filename}")
