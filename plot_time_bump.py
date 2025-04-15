import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import math


# Define the folder where your CSV files are located
folder_path = os.path.dirname(os.path.abspath(__file__)) + '/WL/'
skip_no = 5 
locations = [0, 195, 255, 280, 350]
col = 'd'

all_files = os.listdir(folder_path)
csv_files = [f for f in all_files if f.startswith('WL') and f.endswith('.csv')]
csv_files = sorted(csv_files)
csv_files = csv_files[::skip_no]

G_locations = np.array([195, 255, 280, 350])
G_measure_path = './dam_break_bump/validation/G_measure'
G_measure_files = os.listdir(G_measure_path)
G_measure_files = sorted(G_measure_files)

G_model_path = './dam_break_bump/validation/G_model'
G_model_files = os.listdir(G_model_path)
G_model_files = sorted(G_model_files)

G_measure_list = []
G_model_list = []
time_series = np.zeros([len(csv_files),len(locations)+1])

i = 0
for file in csv_files:
    file_path = os.path.join(folder_path, file)  
    
    df = pd.read_csv(file_path, skiprows=[1])
    file_label = os.path.splitext(file)[0]
    time_series[i, 0] = float(file_label[2:])
    
    for j in range(len(locations)):
        time_series[i, j+1] = df.iloc[locations[j]][col]
    i += 1

sorted_time_series = time_series[np.argsort(time_series[:, 0])]

for file in G_measure_files:
    file_path = os.path.join(G_measure_path, file)  
    
    df = pd.read_csv(file_path, skiprows=[1])
    G_measure_list.append(df)

for file in G_model_files:
    file_path = os.path.join(G_model_path, file)  
    
    df = pd.read_csv(file_path, skiprows=[1])
    G_model_list.append(df)

x = sorted_time_series[:, 0]
num_cols = sorted_time_series.shape[1] - 1
grid_size = math.ceil(math.sqrt(num_cols))
fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 8))
axs = axs.flatten()

for i in range(1, sorted_time_series.shape[1]):
    y = sorted_time_series[:, i]
    axs[i - 1].plot(x, y, color='b', label='CADFFM')
    axs[i - 1].set_title(f'cell {locations[i - 1]*0.1} m')  
    axs[i - 1].set(xlabel='time (s)', ylabel=col)
    indices = np.where(G_locations == locations[i - 1])
    indices = np.array(indices)
    if indices.shape[1]==1:
        n = indices.item()-1
        axs[i - 1].scatter(G_measure_list[n].iloc[:, 0], G_measure_list[n].iloc[:, 1], edgecolor='red', facecolor='none', label='measured')
        axs[i - 1].plot(G_model_list[n].iloc[:, 0], G_model_list[n].iloc[:, 1], color='green', label='SWFCA')
    axs[i - 1].set_xlim([0, 40])
    axs[i - 1].set_ylim([0,0.8])
    axs[i - 1].grid(True)

for j in range(num_cols, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.legend()
plt.show()