import pandas as pd
import matplotlib.pyplot as plt
import os
import math


# Define the folder where your CSV files are located
folder_path = os.path.dirname(os.path.abspath(__file__)) + '/WL/'
skip_no = 10
start_time = 3
end_time = 10

all_files = os.listdir(folder_path)
csv_files = [f for f in all_files if f.startswith('WL') and f.endswith('.csv')]
time = []

for i, file in enumerate(csv_files):
    file_label = os.path.splitext(file)[0]
    time.append(float(file_label[2:]))
    
time = [f"{num:05.2f}" for num in time]
csv_files = [time, csv_files]
csv_files = sorted(zip(csv_files[0], csv_files[1]), key=lambda x: float(x[0]))
csv_files = [(t, f) for t, f in csv_files if start_time <= float(t) <= end_time]
time, csv_files = zip(*csv_files)
csv_files = csv_files[::skip_no]

num_cols = len(csv_files)
grid_size = math.ceil(math.sqrt(num_cols))
fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 8))
axs = axs.flatten()

for i, file in enumerate(csv_files):
    file_path = os.path.join(folder_path, file)  # Create full path to the file
    
    df = pd.read_csv(file_path, skiprows=[1])
    file_label = os.path.splitext(file)[0]
    file_label = 'time ' + file_label[2:]

    axs[i].plot(df['z'], linestyle=':', color='k')
    axs[i].plot(df['wl'], label=f'WL ({file_label})', linestyle='-', color='b')
    axs[i].plot(df['H'], label=f'H ({file_label})', linestyle='--', color='g')
    axs[i].set_xlabel('Index')
    axs[i].set_ylabel('WL / H', color='b')    
    
    ax2 = axs[i].twinx()
    ax2.plot(df['v'], label=f'v ({file_label})', linestyle='-.', color='r')
    ax2.set_ylabel('v', color='r')
    
    axs[i].set_title(file_label)
    axs[i].set_ylim([0,0.8])
    axs[i].grid(True)

for j in range(num_cols, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
plt.legend()
plt.show()

