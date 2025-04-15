from cswmm import swmm
import numpy as np
import matplotlib.pyplot as plt
from core_vect import CADFFM
import sys
from colorama import Fore, Back, Style
from copy import deepcopy
import time
import util
import csv

class csc:
    def __init__(self):
        print("\n .....Initiating 2D-1D modelling using coupled SWMM & CADFFM.....")
        print("\n", time.ctime(), "\n")
        self.t = time.time()

        self.g = 9.81
        self.rel_diff = np.zeros(2)
        self.elv_dif = 0.01
        self.plot_Node_DEM = False
        self.failed_node_check = np.array([], dtype=int)
        self.weir_approach = False
        self.volume_conversion = 0.001
        self.recrusive_run = False
        self.active_nodes = np.array([])
        self.report_max = False
        self.report_interval = False
        self.project_name = ''

    def ActiveNodes(self, filename):
        with open(filename, 'r') as file:
            names = file.read().splitlines()
        self.active_nodes = np.array(names)
        
    def LoadCadffm(self, DEM_file, n, CFL, min_WD, min_Head, g):
        self.cadffm = CADFFM(DEM_file, n, CFL, min_WD, min_Head, g)
        
    def LoadSwmm(self, SWMM_inp):
        self.swmm = swmm(SWMM_inp)
        self.swmm.LoadNodes()
        print("\nSWMM system unit is: ", self.swmm.sim.system_units)
        print("SWMM flow unit is:   ", self.swmm.sim.flow_units, "\n")

    def NodeElvCoordChecks(self):
        # domain size check is conducted here
        c_b = self.cadffm.bounds
        s_b = self.swmm.bounds
        if (c_b[0] <= s_b[0] and c_b[1] >= s_b[1] and c_b[2] >= s_b[2] and c_b[3] <= s_b[3]):
            print('All SWMM junctions are bounded by the provided DEM')
        else:
            sys.exit(
                "The SWMM junctions coordinates are out of the boundry of the provided DEM")

        # convert node locations to DEM numpy system
        # coulumns are: 1 and 2 coordinates, 3 elevation, 4 outfall and 5 active
        self.swmm_node_info = self.swmm.nodes_info.to_numpy()
        self.swmm_node_info = np.column_stack(
            (self.swmm_node_info[:, 0],
             self.swmm_node_info[:, 1],
             self.swmm_node_info[:, 2] + self.swmm_node_info[:, 3],
             self.swmm_node_info[:, 4]))
        self.rel_diff = [self.cadffm.bounds[0], self.cadffm.bounds[1]]
        self.swmm_node_info[:, 0] = np.int_(
            (self.swmm_node_info[:, 0]-self.rel_diff[0]) / self.cadffm.cell_length)
        self.swmm_node_info[:, 1] = np.int_(
            (self.rel_diff[1]-self.swmm_node_info[:, 1]) / self.cadffm.cell_length)

        if self.active_nodes.size > 0:
            active_nodes = np.isin(self.swmm.node_list, self.active_nodes)
        else:
            active_nodes = np.ones((len(self.swmm.node_list)), dtype=bool)

        self.swmm_node_info = np.column_stack(
            (self.swmm_node_info, active_nodes))

        # when a DEM loaded coordinate system is reverse (Rasterio)
        self.swmm_node_info[:, [0, 1]] = self.swmm_node_info[:, [1, 0]]
        # saving nodes on a raster
        tmp = np.zeros_like(self.cadffm.z, dtype=np.double)
        for i in self.swmm_node_info:
            tmp[i[0], i[1]] = 1

        name = self.cadffm.outputs_path + self.cadffm.outputs_name + '_swmm_nodes_raster.tif'
        util.ArrayToRaster(tmp, name, self.cadffm.dem_file,mask=None)

        # plot SWMM nodes on DEM
        if self.plot_Node_DEM:
            temp = deepcopy(self.cadffm.z)
            temp[self.cadffm.ClosedBC == True] = np.max(temp)
            plt.imshow(temp, cmap='gray')
            plt.scatter(self.swmm_node_info[:, 1], self.swmm_node_info[:, 0])
            plt.show()

        # Elevation check is conducted here
        err = False
        i = 0
        for r in self.swmm_node_info:
            if ((abs(r[2] - self.cadffm.z[np.int_(r[0]), np.int_(r[1])])
                    > self.elv_dif * self.cadffm.z[np.int_(r[0]), np.int_(r[1])]) and r[3] == False):
                if not (self.recrusive_run):
                    print(
                        self.swmm.node_list[i],
                        " diff = ", self.cadffm.z
                        [np.int_(r[0]),
                         np.int_(r[1])] - r[2])
                self.failed_node_check = np.append(
                    self.failed_node_check, np.int_(i))
                err = True
            i += 1

        if (err and not (self.recrusive_run)):
            print("The above SWMM junctions surface elevation have >",
                  self.elv_dif*100, "% difference with the provided DEM.\n")
            temp = deepcopy(self.cadffm.z)
            temp[self.cadffm.ClosedBC == True] = np.max(temp)
            plt.imshow(temp, cmap='gray')
            plt.scatter(self.swmm_node_info[self.failed_node_check, 1],
                        self.swmm_node_info[self.failed_node_check, 0])
            plt.show()

        while (err and not (self.recrusive_run)):
            # answer = input("Do you want to continue? (yes/no)")
            answer = "yes"
            if answer == "yes":
                pass
                break
            elif answer == "no":
                sys.exit()
            else:
                print("Invalid answer, please try again.")

        if not (err):
            print(
                "-----Nodes elevation and coordinates are compatible in both models (outfalls excluded)")

    def InteractionInterval(self, sec):
        self.swmm.InteractionInterval(sec)
        self.IntTimeStep = sec
        
    def RunOne_SWMMtoCadffm(self, catchment):
        if not (self.recrusive_run):
            print("\nFor one-time one-way coupling of SWMM to Caffe apprach, SWMM model should generate a report for all nodes.")
            continue_choice = input("Do you want to continue? (yes/no): ")
            if continue_choice.lower() != "yes":
                raise Exception("Program terminated to revise SWMM input file")

        self.swmm.sim.execute()
        print("\n .....finished one-directional coupled SWMM & CAdffm - one timestep.....")

        floodvolume = self.swmm.Output_getNodesFlooding()
        print("\n .....floodvolume retrieved.....")
        # it is multiplied by DEM length as the caffe excess volume will get coordinates
        # not cell. swmm_node_info is already converted to cell location in NodeElvCoordChecks function
        floodvolume = np.column_stack((self.swmm_node_info[:, 0:2]*self.cadffm.cell_length,
                                       np.transpose(floodvolume*self.volume_conversion)))

        self.cadffm.ExcessVolumeMapArray(floodvolume)
        self.cadffm.run_simulation(catchment)
        self.swmm.CloseSimulation()
        print("\n .....finished one-directional coupled SWMM & CADFFM - one timestep.....")
        print("\n", time.ctime(), "\n")
        print("\n Duration: ", time.time()-self.t, "\n")
        
    def RunMulti_SWMMtoCadffm(self, catchment):
        origin_name = self.cadffm.outputs_name
        old_mwd = np.zeros_like(self.cadffm.z, dtype=np.double)
        self.exchange_amount = []

        for step in self.swmm.sim:
            print(Fore.GREEN + "SWMM @ "
                  + str(self.swmm.sim.current_time) + Style.RESET_ALL + "\n")
            floodvolume = self.swmm.getNodesFlooding()*self.IntTimeStep*self.volume_conversion
            total_surcharged = np.sum(floodvolume)
            self.exchange_amount.append(
                [self.swmm.sim.current_time, total_surcharged])

            if (total_surcharged > 0):
                print(Fore.RED + "SWMM surcharged "
                      + str(total_surcharged) + Style.RESET_ALL + "\n")

                floodvolume = np.column_stack(
                    (self.swmm_node_info[:, 0:2] * self.cadffm.cell_length, np.transpose(floodvolume)))
                self.cadffm.ExcessVolumeMapArray(floodvolume, False)
                self.cadffm.outputs_name = origin_name + \
                    "_" + str(self.swmm.sim.current_time)
                self.cadffm.run_simulation(catchment)

        # save all exchanges between models in a csv file including the exchange time
        name = self.cadffm.outputs_path + self.cadffm.outputs_name + '_exchange_report.csv'
        with open(name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for entry in self.exchange_amount:
                formatted_entry = [entry[0].strftime(
                    '%Y-%m-%d %H:%M:%S')] + entry[1:]
                writer.writerow(formatted_entry)

        self.swmm.CloseSimulation()
        print(
            "\n .....finished one-directional coupled SWMM & CADFFM - multi timestep.....")
        print("\n", time.ctime(), "\n")
        print("\n Duration: ", time.time()-self.t, "\n")
        
         
    def Run_Cadffm_BD_SWMM(self, catchment):
        origin_name = self.cadffm.outputs_name
        max_mwd = np.array([[], []])
        max_wd = np.array([[], []])
        max_wl = np.array([[], []])
        First_Step = True
        self.exchange_amount = []
        total_surcharged = 0
        total_drained = 0
        manhole_properties = np.column_stack((self.swmm_node_info[:, 0:2]*self.cadffm.cell_length, 
                                             self.WeirDisCoef, self.WeirCrest))
        self.cadffm.manhole_properties(manhole_properties)
        
        print(Fore.RED + "Starts running SWMM" + Style.RESET_ALL + "\n\n")

        for step in self.swmm.sim:
            print(Fore.GREEN + "SWMM @ "
                  + str(self.swmm.sim.current_time) + Style.RESET_ALL + "\n")

            if First_Step:
                First_Step = False
            else:
                self.cadffm.Reset_WL_EVM()

            # convert flood flowrate at each node to volume
            flooded_nodes_flowrates = np.transpose(
                self.swmm.getNodesFlooding()) * self.IntTimeStep * self.volume_conversion
            total_surcharged = np.sum(flooded_nodes_flowrates)

            if total_surcharged > 0:
                print(Fore.RED + "SWMM surcharged "
                      + str(total_surcharged) + Style.RESET_ALL + "\n")
                # convert node cells to node coordinates (local) for feeding ExcessVolumeArray
                # since it reads the last water depth to initialise, the flood
                # volumes needs to be converted to water depths
                floodvolume = np.column_stack(
                    (self.swmm_node_info[:, 0: 2] * self.cadffm.cell_length,
                     flooded_nodes_flowrates))
                self.cadffm.ExcessVolumeMapArray(floodvolume, False)

            # extract active non-flooded nodes for setting as open BCs
            # convert node cells to node coordinates (local) for feeding caffe.OpenBCArray

            nonflooded_nodes = (flooded_nodes_flowrates == 0) * self.swmm_node_info[:, 4]
            nonflooded_nodes_coords = self.swmm_node_info[nonflooded_nodes >
                                                          0, 0:2] * self.cadffm.cell_length

            self.cadffm.OpenBCMapArray(nonflooded_nodes_coords)

            if (total_surcharged > 0 or np.sum(self.cadffm.excess_volume_map) > 0):
                
                self.cadffm.outputs_name = origin_name + \
                    "_" + str(self.swmm.sim.current_time)
                    
                self.cadffm.run_simulation(catchment)
                # self.cadffm.ReportScreen()

                if (np.sum(self.cadffm.d_inflows) > 0):
                    j = 0
                    k = 0
                    inflow = np.zeros(self.swmm_node_info.shape[0])

                    for i in nonflooded_nodes:
                        if i == 1:
                            x = int(self.cadffm.OBC_cells[k, 0])
                            y = int(self.cadffm.OBC_cells[k, 1])

                            inflow[j] = np.copy(
                                self.cadffm.d_inflows[x, y])

                            k += 1
                        j += 1

                    self.swmm.setNodesInflow(
                        inflow * self.cadffm.cell_area / self.IntTimeStep /
                        self.volume_conversion)
                    total_drained = np.sum(
                        inflow) * self.cadffm.cell_area
                    print(Fore.BLUE + "\nCADFFM drained "
                          + str(total_drained) + Style.RESET_ALL + "\n")

                self.cadffm.outputs_name = origin_name + \
                    "_" + str(self.swmm.sim.current_time)
                if self.report_interval == True:
                    self.cadffm.report_file(self.cadffm.outputs_name)
                    
                if self.report_max == True:
                    if np.any(max_mwd) == False:
                        max_mwd = self.cadffm.max_WD.copy()
                        max_wd = self.cadffm.d.copy()
                        max_wl = self.cadffm.WL.copy()
                    else:
                        max_mwd = np.maximum(
                            max_mwd, self.cadffm.max_WD)
                        max_wd = np.maximum(max_wd, self.cadffm.d)
                        max_wl = np.maximum(max_wl, self.cadffm.WL)

            self.exchange_amount.append(
                [self.swmm.sim.current_time, total_surcharged, total_drained])

        self.swmm.CloseSimulation()

        if self.report_max == True:
            self.cadffm.Reset_WL_EVM()
            self.cadffm.max_WD = max_mwd.copy()
            self.cadffm.d = max_wd.copy()
            self.cadffm.WL = max_wl.copy()
            self.cadffm.report_file(self.project_name + '_max_all')

        # save all exchanges between models in a csv file including the exchange time
        name = self.cadffm.outputs_path + self.cadffm.outputs_name + '_exchange_report.csv'
        with open(name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for entry in self.exchange_amount:
                formatted_entry = [entry[0].strftime(
                    '%Y-%m-%d %H:%M:%S')] + entry[1:]
                writer.writerow(formatted_entry)

        print("\n .....finished bi-directional coupled SWMM & CADFFM.....")
        print("\n", time.ctime(), "\n")
        print("\n Duration: ", time.time()-self.t, "\n")
        
    def ManholeProp(self, coef, length):
        coef = np.asarray(coef)
        length = np.asarray(length)
        if coef.size == 1 or coef.size == self.swmm_node_info.shape[0]:
            if coef.size == 1:
                coef = np.repeat(coef, self.swmm_node_info.shape[0])
        else:
            sys.exit("Weir discharge coefficient array size does not match ",
                     "junction numbers")

        if length.size == 1 or length.size == self.swmm_node_info.shape[0]:
            if length.size == 1:
                length = np.repeat(length, self.swmm_node_info.shape[0])
        else:
            sys.exit("Weir crest length array size does not match junction numbers")

        self.WeirDisCoef = coef
        self.WeirCrest = length
