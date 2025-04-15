
import shutil
import os
import pandas as pd
import sys
import os
sys.path.append('./src')
from cadffm_swmm_coupled import csc

if __name__ == "__main__":
    # Path to the input raster file
    tif_file = "./catchment/Scotchmans_creek/Scotchmans_Creek.tif"      
    SWMM_inp = "./Scotchmans_creek/SWMM.inp"
    open_bnd_file = "./Scotchmans_creek/open_bnd_sctochmans_creek.csv"
    # Set the simulation parameters
    interaction_time = 600                                    # Total simulation time
    time_step = 1e-1                                    # Time step
    n = 0.025                                          # Manning's n      
    CFL = 0.6                                         # CFL number
    min_WD = 5e-3                                       # Minimum water depth
    min_Head = 5e-4                                     # Minimum water head difference
    g = 9.81                                            # Acceleration due to gravity                                 
    
    # Create the output folder
    output_folder = './Scotchmans_creek/results_cfl_' + "{:1.0e}".format(CFL) + '_mh_' + \
    "{:1.1e}".format(min_Head) + '_mwd_' + "{:1.1e}".format(min_WD) + \
        '_interv' + str(interaction_time) + "{:1.1e}".format(n) + 'BD_n2/'
    source_file = './src/core_vect.py'
    run_file = './run_scotchmans_csc.py'
    csc_file = './src/cadffm_swmm_coupled.py'
    os.makedirs(output_folder, exist_ok=True)
    shutil.copy(source_file, output_folder)
    shutil.copy(run_file, output_folder)
    shutil.copy(csc_file, output_folder)
    
    # Model run
    csc_obj = csc()
    csc_obj.LoadCadffm(tif_file, n, CFL, min_WD, min_Head, g)
    
    #set the outflow in the boundary edge
    bnds = csc_obj.cadffm.bounds[0], csc_obj.cadffm.bounds[1]
    df_open_bnd = pd.read_csv(open_bnd_file)
    open_bnd_coord = df_open_bnd.iloc[:,0:2].to_numpy()
    open_bnd_coord[:,0] = open_bnd_coord[:,0]- bnds[0]
    open_bnd_coord[:,1] = bnds[1] - open_bnd_coord[:,1]
    open_bnd_coord[:,[0,1]] = open_bnd_coord[:,[1,0]]
    open_bnd_depth = df_open_bnd.iloc[:, 2].to_numpy()
    csc_obj.cadffm.OpenBCMapArray_outlet(open_bnd_coord, open_bnd_depth)
    
    # set parameters for the simulation
    csc_obj.cadffm.set_simulation_time(interaction_time, time_step, 0)
    csc_obj.cadffm.set_output_path(output_folder)
    csc_obj.LoadSwmm(SWMM_inp)
    csc_obj.NodeElvCoordChecks()
    csc_obj.InteractionInterval(interaction_time)
    
    csc_obj.ManholeProp(0.5, 1)
    # csc_obj.RunOne_SWMMtoCadffm(catchment = True)
    # csc_obj.RunMulti_SWMMtoCadffm(catchment = True)
    csc_obj.Run_Cadffm_BD_SWMM(catchment = True)
    

    
