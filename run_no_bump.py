import sys
import shutil
import os
sys.path.append('./src')
from core_vect import CADFFM

if __name__ == "__main__":
    
    # Path to the input raster file
    tif_file = "./test_cases/dam_break_no_bump/DamBreak_noBump_0.1size.tif"

    # Set the simulation parameters
    tot_time = 10                                       # Total simulation time
    time_step = 1e-6                                    # Time step
    n = 0.01                                            # Manning's n      
    CFL = 0.02                                          # CFL number
    res_depth = 0.3                                     # Depth of reservoir
    ds_depth = 0                                        # Depth of downstream
    reserv_length = 3.3                                 # Length of reservoir
    min_WD = 1e-3                                       # Minimum water depth
    min_Head = 1e-4                                     # Minimum water head difference
    g = 9.81                                            # Acceleration due to gravity
    
    # Create the output folder
    output_folder = './test_cases/dam_break_no_bump/results_cfl_' + "{:1.0e}".format(CFL) + \
    '_mh_' + "{:1.1e}".format(min_Head) + '_size/'
    source_file = './src/core_vect.py'
    plot_file_1 = './plot_depth_subplot.py'
    plot_depth_vel = './plot_depth_vel_compare.py'
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy(plot_file_1, output_folder)
    shutil.copy(plot_depth_vel, output_folder)
    shutil.copy(source_file, output_folder)
 
    # Model run
    model=CADFFM(tif_file, n, CFL, min_WD, min_Head, g)
    model.set_simulation_time(tot_time, time_step, 0)
    model.set_output_path(output_folder)
    model.set_reservoir(res_depth, ds_depth, reserv_length)
    model.run_simulation(catchment = False)
    
    