import numpy as np
import util as ut
import os
import time
import math
import sys
import pandas as pd

#Quadratic Equation Solver

def quad_Eq(a, b, c, sign):
    discriminant = b**2 - 4*a*c
    if discriminant > 0:
        # two real solutions
        sqrt_discriminant = math.sqrt(discriminant)
        denom = 2*a
        x1 = (-b + sqrt_discriminant) / denom
        x2 = (-b - sqrt_discriminant) / denom
    
        solution2 = sign*x2 if sign*x2 > 0 else 0
        solution1 = sign*x1 if sign*x1 > 0 else 0
        solution = min(solution1, solution2) if solution1 and solution2 else max(solution1, solution2)
        solution *= sign
    elif discriminant == 0:
        # one real solution
        x = -b / (2*a)
        solution = x if sign*x > 0 else 0
    else:
        solution = 0

    return solution

def delta(a):
    if a>0:
        return 1
    else:
        return 0

def non_zero_min(a, b):
    if a * b != 0:
        return min(a, b)
    else:
        return (a + b)
    
# global constants to reduce computational time
c4_3 = 4/3
c5_3 = 5/3
c2_3 = 2/3

# A Cellular Automata Dynamic Fast Flood Model (main class)
class CADFFM:
    def __init__(self, dem_file, n, CFL, min_wd, min_head, g):
        print("\n .....loading DEM file using CADFFM.....")
        self.begining = time.time()
        print("\n", time.ctime(), "\n")

        self.dem_file = dem_file
        self.z, self.ClosedBC, self.bounds, self.cell_length = ut.RasterToArray(dem_file)
        self.dem_shape = self.z.shape
        self.OpenBC = np.zeros_like(self.z, dtype=np.bool_)
        self.OpenBC_outlet = np.zeros_like(self.z, dtype=np.bool_)
        self._initialize_arrays()                           # to initialize arrays
        self.n0[:] = n                                      # Manning's n
        self.t = 0
        self.delta_t = 0
        # how fast approach the calculated timestep from adaptive reduced timestep
        self.delta_t_bias = 0
        self.g = g                                          
        self.CFL = CFL                                      
        self.BCtol= 1e6                                     # set as a boundary cell elevation
        self.min_WD = min_wd                                
        self.initial_min_Head = min_head                    # head factor to consider the flow direction         
        self.remaining_volume = 0                           # total remaining volume of water
        self.initial_volume = 0                             # initial volume of water
        self.cell_area = self.cell_length**2
        self.sqrt_2_g = 0
        self.weir_eq_const = 0
        self.csv_merged = []
        self.d_inflows = np.zeros_like(self.z, dtype=np.float64)
        
        #constant for model simplification
        self.gravity_inverse = 0.5 / self.g
        self.sqrt_2_g = np.sqrt(2*self.g)
        self.weir_eq_const = c2_3 * self.cell_length * self.sqrt_2_g
        self.half_cell_size = 0.5 * self.cell_length
        self.weir_dis_coef = np.zeros_like(self.z, dtype=np.float64)
        self.weir_crest = np.zeros_like(self.z, dtype=np.float64)
        
    # initialize arrays   
    def _initialize_arrays(self):
        dtype0 = np.float64
        self.u = np.zeros_like(self.z, dtype=dtype0)
        self.v = np.zeros_like(self.z, dtype=dtype0)
        self.d = np.zeros_like(self.z, dtype=dtype0)
        self.WL = np.zeros_like(self.z, dtype=dtype0)
        self.max_WD = np.zeros_like(self.z, dtype=dtype0)
        self.max_vel = np.zeros_like(self.z, dtype=dtype0)
        self.vel_res = np.zeros_like(self.z, dtype=dtype0)
        self.n0 = np.zeros_like(self.z, dtype=dtype0)
        self.OpenBC = np.zeros_like(self.z, dtype=np.bool_)
        self.theta = np.array([1, 1, -1, -1], dtype=dtype0)
        self.excess_volume_map = np.zeros_like(self.z, dtype=np.float64)
        self.OBC_cells = np.array([])
        self.OBC_cells_outlet = np.array([])
        self.CBC_cells = np.array([])
        self.OBC_depth = np.array([])
        self.H_i_dir = np.zeros(4, dtype=np.int8)
        self.Q_i_dir = np.zeros(4, dtype=np.int8)
        self.vel_ini = np.zeros(4, dtype=np.float64)
        
    def set_simulation_time(self, t, delta_t, delta_t_bias=0):
        self.t = t
        self.delta_t = delta_t
        self.delta_t_bias = delta_t_bias
    
    def set_output_path(self, output_path):
        self.outputs_path = output_path
        self.DEM_path = "./"
        name = self.dem_file.split('/')
        name = name[-1].split('.')
        self.outputs_name = name[0] + "_out"
        self.WL_output = os.path.join(output_path, 'WL/')
        # create a file if the file path is not created for above paths 
        for path in [self.WL_output]:
            if not os.path.exists(path):
                os.makedirs(path)
                
   # set the open boundary cells for outlet at the edge of the domain
    def OpenBCMapArray_outlet(self, OBCM_np_outlet, OBC_depth):
        self.OBC_cells_outlet = np.zeros_like(OBCM_np_outlet, dtype=np.int64)
        OBCM_np_outlet[:, 0] = OBCM_np_outlet[:, 0] / self.cell_length
        OBCM_np_outlet[:, 1] = OBCM_np_outlet[:, 1] / self.cell_length
        i = 0
        for r in OBCM_np_outlet:
            self.OpenBC_outlet[int(r[0]), int(r[1])] = True
            self.ClosedBC[int(r[0]), int(r[1])] = False
            self.OBC_cells_outlet[i, 0] = int(r[0])
            self.OBC_cells_outlet[i, 1] = int(r[1])
            i += 1
        if OBC_depth is not None:
            self.OBC_depth = OBC_depth
            
    def OpenBCMapArray(self, OBCM_np):
        self.OBC_cells = np.zeros_like(OBCM_np, dtype=np.int64)
        OBCM_np[:, 0] = OBCM_np[:, 0] / self.cell_length
        OBCM_np[:, 1] = OBCM_np[:, 1] / self.cell_length

        i = 0
        for r in OBCM_np:
            self.OpenBC[int(np.ceil(r[0])), int(np.ceil(r[1]))] = True
            self.OBC_cells[i, 0] = int(np.ceil(r[0]))
            self.OBC_cells[i, 1] = int(np.ceil(r[1]))
            i += 1
            
    def ClosedBCMapArray(self, CBCM_np):
        self.CBC_cells = np.zeros_like(CBCM_np, dtype=np.int)
        CBCM_np[:, 0] = CBCM_np[:, 0] / self.cell_length
        CBCM_np[:, 1] = CBCM_np[:, 1] / self.cell_length

        i = 0
        for r in CBCM_np:
            self.ClosedBC[int(np.ceil(r[0])), int(np.ceil(r[1]))] = True
            self.CBC_cells[i, 0] = int(np.ceil(r[0]))
            self.CBC_cells[i, 1] = int(np.ceil(r[1]))
            i += 1
            
    def set_BCs(self):
        # it modified the defined boundary cells' elevation
        self.z[self.ClosedBC] += self.BCtol 
        self.d[self.OpenBC] = 0       

    def reset_BCs(self):
        # it restore boundary cells' elevation to normal
        self.z[self.ClosedBC] -= self.BCtol
        # self.z[self.OpenBC] += self.BCtol
        self.d[self.OpenBC_outlet] = np.minimum(self.d[self.OpenBC_outlet], self.OBC_depth) 
    
    # set the weir/orifice properties
    def manhole_properties(self, manhole_np):
        manhole_np[:, 0] = manhole_np[:, 0] / self.cell_length
        manhole_np[:, 1] = manhole_np[:, 1] / self.cell_length
        for r in manhole_np:
            self.weir_dis_coef[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] = (r[2])
            self.weir_crest[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] = (r[3])
    
    # calculate inflows to the nodes for drainage system   
    def inflows_to_nodes(self, delta_t):

        Q_weir_nodes = np.zeros_like(self.z, dtype=np.float64)
        d_to_nodes = np.zeros_like(self.z, dtype=np.float64)

        mask = self.OpenBC.copy()  # Boolean mask for open boundary cells
        # Calculate inflow to nodes with weir/orifice equation
        Q_weir_nodes[mask] = self.weir_dis_coef[mask] * self.weir_crest[mask] * self.d[mask] * np.sqrt(2 * self.d[mask] * self.g)
        d_to_nodes[mask] = Q_weir_nodes[mask] / self.cell_area * delta_t
        # avoiding excess outflow from the open boundary cells 
        d_to_nodes[mask] = np.minimum(np.maximum(d_to_nodes[mask]-self.min_WD,0)        
                                      , np.maximum(self.d[mask]-self.min_WD,0))
        # d_to_nodes[mask] = np.minimum(d_to_nodes[mask], self.d[mask])
        self.d_inflows[mask] += d_to_nodes[mask]                        # storing inflows to nodes for drainage system
        self.d[mask] -= d_to_nodes[mask]
    # set the depth of the reservoir as input   
    def set_reservoir(self, res_depth, ds_depth, length):
        dam_length = round(length/self.cell_length)+1
        self.d[1:-1, 1:dam_length] = res_depth
        self.d[1:-1, dam_length:] = ds_depth

    # calculate max timestep based on Courant–Friedrichs–Lewy condition
    def CFL_delta_t(self):
        velocity_magnitude = np.sqrt(self.u**2 + self.v**2) + np.sqrt(self.g * self.d)
        velocity_magnitude = np.maximum(velocity_magnitude, 1e-10)
        return self.CFL * np.min(self.cell_length / velocity_magnitude)

    # Calculate the Bernoulli head
    def compute_Bernoulli_head(self):
        return self.z + self.d + (self.u**2 + self.v**2)/(2*self.g)
    
    def normal_flow_direction(self, u0, v0, H, d):
        self.H_i_dir.fill(0)
        self.Q_i_dir.fill(0)

        H_diff = H[0]-H[1:5]
        self.H_i_dir[H_diff > self.min_Head] = 1
        
        Q_i = np.array([u0, v0, u0, v0]) * self.theta
        np.around(Q_i, 2, out=Q_i)
        self.Q_i_dir[Q_i >= 0] = 1        
        self.Q_i_dir *= self.H_i_dir
                
        return self.Q_i_dir

    # Calculate if (Q*theta)>0 for Special Flow Condition
    def special_flow_direction(self, u0, v0, H, d):
        self.H_i_dir.fill(0)
        self.Q_i_dir.fill(0)
        H_diff = H[1:5]-H[0]
        self.H_i_dir[H_diff > self.min_Head] = 1

        Q_i = np.array([u0, v0, u0, v0]) * self.theta
        self.Q_i_dir[Q_i > 0] = 1
        
        self.Q_i_dir *= self.H_i_dir

        return self.Q_i_dir
    
    def Q_Mannings(self, n0, H, d, i):
        # calculate mass flux with Mannings equation
        Hloc = H[0] - H[i+1]
        Q = (self.cell_length / n0) * d[0]**(c5_3) * (Hloc / self.cell_length)**0.5                 # d[0] = d_bar_i

        return Q
    
    def Q_Weir(self, h0_i, ratio):
        # calculate mass flux with Mannings equation
        psi_i = (1 - ratio**1.5)**0.385  
        return self.weir_eq_const * psi_i * h0_i**1.5
                
    def compute_normal_flow_mass_flux_i(self, n0, H, d, z, u, v, i):
        # to calculate submergence ratio of the weir
        Q_weir = 0
        ratio_h = ratio_d = 1
        z_bar_i = max(z[0], z[i+1])
        h0_i = H[0] - z_bar_i
        h_i =  H[i+1] - z_bar_i
        d0_i = d[0] + z[0] - z_bar_i
        d_i = d[i+1] + z[i+1] - z_bar_i

        if h_i > 0:
            ratio_h = h_i/h0_i
        elif h_i <= 0:
            ratio_h = 0
        if d0_i > 0:
            if d_i > 0:
                ratio_d = d_i/d0_i
            elif d_i <= 0:
                ratio_d = 0
        min_ratio = min(ratio_h, ratio_d)                   #min ratio for max flux
        if min_ratio < 1:
            Q_weir = self.Q_Weir(h0_i, min_ratio)
            
        Q_mannings = self.Q_Mannings(n0, H, d, i)
        
        Q = non_zero_min(Q_mannings, Q_weir) * self.theta[i] 
        
        return  Q
    
    # Calculate mass flux for special flow condition
    def compute_special_flow_mass_flux_i(self, d, z, u, v, H, n0, deltat, i):
        dQ_1 = self.cell_area / (2 * deltat) * (H[i+1] - H[0] + self.min_Head)
        dQ_2 = self.cell_area / deltat * (d[i+1] - self.min_WD)
        dQ = min(dQ_1, dQ_2)
        
        # to calculate ratio for the weir
        Q_weir = 0
        ratio_d = 1
        z_bar_i = max(z[0], z[i+1])
        d0_i = d[0] + z[0] - z_bar_i
        d_i = d[i+1] + z[i+1] - z_bar_i
        if d0_i > 0:
            if d_i > 0:
                ratio_d = d_i/d0_i
            elif d_i <= 0:
                ratio_d = 0
            if ratio_d < 1:
                Q_weir = self.Q_Weir(d0_i, ratio_d)
        Q = Q_weir
        
        if i == 0 or i == 2:
            Q_vel = self.cell_length * d[0] * u[0] * self.theta[i]
        else:
            Q_vel = self.cell_length * d[0] * v[0] * self.theta[i]

        if Q_vel < 0:
            Q_vel = 0
        if Q_vel != 0:
            Q = Q_vel
            
        if Q != 0:
            tmp = abs(Q)-dQ
            if tmp > 0:
                Q = tmp
            else:
                Q = 0
  
        return Q * self.theta[i]
    
    def compute_velocity(self, FD, n, d, z, u, v, H0, d0):
        self.vel_ini.fill(0)

        for i in [1, 3]:
            if FD[i-1] == 1:
                b = self.theta[i-1] * self.half_cell_size * n[i]**2 * abs(u[i]) / d[i]**c4_3
                c = v[i]**2 * self.gravity_inverse + d[i] + z[i] + self.half_cell_size * \
                    n[0]**2 * u[0]**2 / d0**c4_3  - H0
                self.vel_ini[i-1] = quad_Eq(self.gravity_inverse, b, c, self.theta[i-1])

        for i in [2, 4]:
            if FD[i-1] == 1:
                b = self.theta[i-1] * self.half_cell_size * n[i]**2 * abs(v[i]) / (d[i]**c4_3)
                c = u[i]**2 * self.gravity_inverse + d[i] + z[i] + self.half_cell_size * \
                    n[0]**2 * v[0]**2 / d0**c4_3  - H0
                self.vel_ini[i-1] = quad_Eq(self.gravity_inverse, b, c, self.theta[i-1])
                
        return self.vel_ini
    
    def ExcessVolumeMapArray(self, EVM_np, add=False):
        EVM_np[:, 0] = EVM_np[:, 0] / self.cell_length
        EVM_np[:, 1] = EVM_np[:, 1] / self.cell_length
        for r in EVM_np:
            if add:
                self.excess_volume_map[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] += (r[2])
            else:
                self.excess_volume_map[int(
                    np.ceil(r[0])), int(np.ceil(r[1]))] = (r[2])
       
    def Reset_WL_EVM(self):
        # it resets water levels and excess volume map and it also removes
        # defined boundary cells (closed and open) from the lists
        self.WL = self.z.copy()
        self.excess_volume_map = np.zeros_like(self.z, dtype=np.double)
        self.d_inflows.fill(0)              # reset inflows         
    
        for r in self.CBC_cells:
            self.ClosedBC[r[0], r[1]] = False

        for r in self.OBC_cells:
            self.OpenBC[r[0], r[1]] = False
    
    def run_simulation(self, catchment=False):
        # to ensure if user changed it 
        current_time = 0
        iteration = 1
        self.begining = time.time()
        if self.t * self.delta_t == 0:
            sys.exit("Simulation time or timestep have not been defined")

        delta_t = min(self.delta_t, self.CFL_delta_t())
        Time = True
        t_value = 0.1                                               # assigned to use later for depth extraction of every multiples of 0.1sec

        self.initial_volume = np.sum(self.d[:,:-1]) * self.cell_area
        initial_min_head = self.initial_min_Head
        mask = self.ClosedBC
        factor = 1e6
        self.vel_res = self.u**2 + self.v**2
        while current_time < self.t:
            
            self.set_BCs()
            if np.sum(self.excess_volume_map) > 1e-9:
                current_volume = self.excess_volume_map / (self.t - current_time) * delta_t
                self.d += current_volume / self.cell_area
                self.excess_volume_map -= current_volume
                
            H = self.compute_Bernoulli_head()         # calculate Bernoulli head
            Max_KE = np.max(self.vel_res * self.d)
            self.min_Head = max(Max_KE * initial_min_head, 1e-6)
                
            FD = np.zeros_like(self.z, dtype=np.int8)
            delta_d= np.zeros_like(self.z, dtype=np.float64)
            d_new = np.zeros_like(self.z, dtype=np.float64)                  # new water depth
            u_new = np.zeros_like(self.u, dtype=np.float64)
            v_new = np.zeros_like(self.v, dtype=np.float64)

            # this loop caculates updated d, mass fluxes and directions
            for i in range(1, self.dem_shape[0] - 1):
                for j in range(1, self.dem_shape[1] - 1):
                    # check if the cell is dry or wet (d0>=delta)
                    if self.d[i,j] > self.min_WD:
                        # For simplicity, the central cell is indexed as 0, and
                        # its neighbor cells at the east, north, west, and
                        # south sides are indexed as 1, 2, 3, and 4.
                        z = np.array([self.z[i, j], self.z[i+1, j],
                                      self.z[i, j+1], self.z[i-1, j],
                                      self.z[i, j-1]], dtype=np.float64)
                        d = np.array([self.d[i, j], self.d[i+1, j],
                                      self.d[i, j+1], self.d[i-1, j],
                                      self.d[i, j-1]], dtype=np.float64)
                        u = np.array([self.u[i, j], self.u[i+1, j],
                                      self.u[i, j+1], self.u[i-1, j],
                                      self.u[i, j-1]], dtype=np.float64)
                        v = np.array([self.v[i, j], self.v[i+1, j],
                                      self.v[i, j+1], self.v[i-1, j],
                                      self.v[i, j-1]], dtype=np.float64)
                        H_loc = np.array([H[i, j], H[i+1, j],
                                         H[i, j+1], H[i-1, j],
                                         H[i, j-1]], dtype=np.float64)
                        # calculate normal flow                        
                        normal_flow = self.normal_flow_direction(u[0], v[0], H_loc, d)
                        Qn = np.zeros(4, dtype=np.float64)
                        for n in range(4):
                            if normal_flow[n] > 0:
                                Qn[n] = self.compute_normal_flow_mass_flux_i(self.n0[i, j], H_loc, d, z, u, v, n)
                                
                        # compute special flow
                        # special_flow = copy.deepcopy(d[1:])
                        special_flow = np.copy(d[1:])
                        special_flow[special_flow > self.min_WD] = 1
                        special_flow *= self.special_flow_direction(u[0], v[0], H_loc, d)

                        Qs = np.zeros(4, dtype=np.float64)
                        for n in range(4):
                            if special_flow[n] > 0:
                                Qs[n] = self.compute_special_flow_mass_flux_i(d, z, u, v, H_loc, self.n0[i, j], delta_t, n)
                  
                        # final flow flux values considering both conditions
                        Q = np.trunc((Qn + Qs) * factor) / factor
                        
                        #  save flow directions as a binary value to reterive later
                        FD_loc = np.copy(Q)
                        FD_loc[FD_loc != 0] = 1
                        FD[i, j] = int(
                            ''.join(map(str, FD_loc.astype(np.int8))), 2)
                        
                        # update the water depth according to the mass flux
                        delta_d[i,j] += 1 / self.cell_area * \
                                    np.sum(-self.theta * Q)             
                        if Q[0] != 0:
                            delta_d[i+1, j] += 1 / self.cell_area * \
                                (self.theta[0] * Q[0])
                        if Q[1] != 0:
                            delta_d[i, j+1] += 1 / self.cell_area * \
                                (self.theta[1] * Q[1])
                        if Q[2] != 0:
                            delta_d[i-1, j] += 1 / self.cell_area * \
                                (self.theta[2] * Q[2])
                        if Q[3] != 0:
                            delta_d[i, j-1] += 1 / self.cell_area * \
                                (self.theta[3] * Q[3]) 
                        
            d_new = self.d + delta_d * delta_t

            # adaptive time stepping => find the min required timestep
            # this section avoids over drying cells (negative d)
            if (d_new < 0).any():
                neg_indices = np.where(d_new < 0)
                tmp_values = np.abs(self.d[neg_indices] / delta_d[neg_indices])
                # for better stability, devided by 10 
                delta_t_new = np.min(tmp_values) / 10
                d_new = self.d + delta_d * delta_t_new
            else:
                delta_t_new = delta_t
            current_time += delta_t_new
            
            # this loop calculate velocities in both directions
            for i in range(1, self.dem_shape[0]-1):
                for j in range(1, self.dem_shape[1]-1):
                    # check if the cell is dry or wet (d0>=delta)
                    if self.d[i,j] > self.min_WD and FD[i,j] != 0:
                        # For simplicity, the central cell is indexed as 0, and
                        # its neighbor cells at the east, north, west, and
                        # south sides are indexed as 1, 2, 3, and 4.
                        z = np.array([self.z[i, j], self.z[i+1, j],
                                      self.z[i, j+1], self.z[i-1, j],
                                      self.z[i, j-1]])
                        d_n = np.array([d_new[i, j], d_new[i+1, j],
                                      d_new[i, j+1], d_new[i-1, j],
                                      d_new[i, j-1]])
                        u = np.array([self.u[i, j], self.u[i+1, j],
                                      self.u[i, j+1], self.u[i-1, j],
                                      self.u[i, j-1]])
                        v = np.array([self.v[i, j], self.v[i+1, j],
                                      self.v[i, j+1], self.v[i-1, j],
                                      self.v[i, j-1]])
                        n = np.array([self.n0[i, j], self.n0[i+1, j],
                                      self.n0[i, j+1], self.n0[i-1, j],
                                      self.n0[i, j-1]])

                        # retrieve the flow direction from binary value
                        FD_loc = np.binary_repr(FD[i, j], 4)
                        FD_loc = [int(digit) for digit in FD_loc] 
                        
                        vel = self.compute_velocity(
                            FD_loc, n, d_n, z, u, v, H[i, j], self.d[i, j])
                        
                        u_new[i-1, j] += vel[2]
                        u_new[i+1, j] += vel[0]
                        v_new[i, j-1] += vel[3]
                        v_new[i, j+1] += vel[1]
            
            curr_v_sqr = v_new**2
            curr_u_sqr = u_new**2
            vel_max_head = np.maximum(self.u**2 + curr_v_sqr, curr_u_sqr + self.v**2)             # eliminated 2g from the equation
            vel_new_head = (curr_u_sqr + curr_v_sqr)
            
            for i in range(1, self.dem_shape[0]-1):
                for j in range(1, self.dem_shape[1]-1):
                    if d_new[i,j] > self.min_WD:
                            
                        if (u_new[i, j] != 0 and v_new[i, j] != 0) and (self.u[i, j] == 0 and self.v[i, j] == 0):
                            max_vel = max(abs(u_new[i,j]), abs(v_new[i,j]))
                            self.u[i,j] = u_new[i,j]/ np.sqrt(vel_new_head[i,j]) * max_vel
                            self.v[i,j] = v_new[i,j]/ np.sqrt(vel_new_head[i,j]) * max_vel
                        
                        elif (u_new[i, j] != 0 and v_new[i, j] != 0) and (vel_new_head[i,j] > vel_max_head[i,j]):
                            if abs(u_new[i, j]) < abs(v_new[i, j]):
                                self.u[i,j] = (self.u[i,j] + u_new[i,j]) / 2  
                                self.v[i,j] = np.sqrt(vel_max_head[i,j] - self.u[i,j]**2) * np.sign(v_new[i,j])
                            elif abs(u_new[i, j]) > abs(v_new[i, j]):
                                self.v[i,j] = (self.v[i,j] + v_new[i,j]) / 2
                                self.u[i,j] = np.sqrt(vel_max_head[i,j] - self.v[i,j]**2) * np.sign(u_new[i,j])
                            else:
                                self.u[i,j] = u_new[i,j] / np.sqrt(2)
                                self.v[i,j] = v_new[i,j] / np.sqrt(2)
                        else:
                            if u_new[i, j] != 0:
                                self.u[i, j] = u_new[i, j]
                            if v_new[i, j] != 0:
                                self.v[i, j] = v_new[i, j]

                        if mask[i-1,j] or mask[i+1,j]:
                            self.u[i, j] = 0
                        if mask[i,j-1] or mask[i,j+1]:
                            self.v[i, j] = 0
                                
                        # if mask[i,j+1] and mask[i-1,j]:
                        #     if not mask[i+1,j+1]:
                        #         self.u[i, j] = v_new[i, j]
                        #     if not mask[i-1,j-1]:
                        #         self.v[i, j] = u_new[i, j]
                        # if mask[i,j+1] and mask[i+1,j]:
                        #     if not mask[i-1,j+1]:
                        #         self.u[i, j] = -v_new[i, j]
                        #     if not mask[i+1,j-1]:
                        #         self.v[i, j] = -u_new[i, j]
                        # if mask[i,j-1] and mask[i-1,j]:
                        #     if not mask[i+1,j-1]:
                        #         self.u[i, j] = -v_new[i, j]
                        #     if not mask[i-1,j+1]:
                        #         self.v[i, j] = -u_new[i, j]
                        # if mask[i,j-1] and mask[i+1,j]:
                        #     if not mask[i-1,j-1]:
                        #         self.u[i, j] = v_new[i, j]
                        #     if not mask[i+1,j+1]:
                        #         self.v[i, j] = u_new[i, j]
                    else:
                        self.u[i, j] = 0
                        self.v[i, j] = 0
                        
            # Set velocities to zero at the boundary cells
            self.u[self.OpenBC] = 0
            self.v[self.OpenBC] = 0
            self.v = np.trunc(self.v * factor) / factor
            self.u = np.trunc(self.u * factor) / factor 
            
            self.d = d_new
            self.inflows_to_nodes(delta_t_new)
            self.reset_BCs()
            self.max_WD = np.maximum(self.max_WD, self.d)
            self.vel_res = self.u**2 + self.v**2
            self.max_vel = np.maximum(self.max_vel, np.sqrt(self.vel_res))
            
            # to print the outputs for time steps/iterations
            if iteration % 100 == 0:
                self.report_screen(iteration, delta_t_new, current_time)

            # time step adjustment
            delta_t = self.CFL_delta_t()
            if delta_t_new < delta_t:
                delta_t = (delta_t_new + (self.delta_t_bias + 1)
                           * delta_t) / (self.delta_t_bias + 2)

            # self.time_step.append([current_time, delta_t_new])
            if current_time >= t_value:
                Time = True
                t_value += 0.1
            
            self.reset_BCs()
            self.WL = self.d + self.z
            # self.export_time_series(current_time, self.z, self.d, self.d + self.z , H, self.v)
            # if iteration == 10:
            #     os.abort()
            # to export the water level and velocity at specified times
            if Time and not catchment: 
                self.export_time_series(current_time, self.z, self.d, self.d + self.z , H, self.v)
                Time = False              
            iteration += 1
        # self.export_merged_all(First, Concat)
        # csv_name = self.outputs_path + './merged.csv'
        # csv_all = pd.concat(self.csv_merged, ignore_index=True)
        # csv_all.to_csv(csv_name, index=False)
        
        self.delta_t = delta_t
        self.close_simulation(self.outputs_name, iteration, current_time, delta_t_new)
        print("\nSimulation finished in", (time.time() - self.begining),
        "seconds")
    
    def export_merged_all(self, First, Concat):
        gap = pd.DataFrame([None] * 1)
        h_gap =self.z.shape[1] + 1
        if First:
            First = False
            d_name_df = pd.DataFrame(['d']*self.z.shape[1]).transpose()
            d_name_df[h_gap] = np.nan
            
            u_name_df = pd.DataFrame(['u']*self.z.shape[1]).transpose()
            u_name_df[h_gap] = np.nan

            v_name_df = pd.DataFrame(['v']*self.z.shape[1]).transpose()
            v_name_df[h_gap] = np.nan
            
        d_df = pd.DataFrame(self.d) #this is dnew_dataframe
        d_df[h_gap] = np.nan
                
        u_df = pd.DataFrame(self.u) #this is  u dataframe
        u_df[h_gap] = np.nan
                
        v_df = pd.DataFrame(self.v) #this is  v dataframe
        v_df[h_gap] = np.nan
        
        concat_df = pd.concat([d_df,u_df, v_df], axis=1)
        concat_df.reset_index(drop=True, inplace=True)
        concat_name = pd.concat([d_name_df,u_name_df, v_name_df], axis=1)        
        concat_name.reset_index(drop=True, inplace=True)
        if Concat==True:
            Concat = False
            merge_df = pd.concat([concat_name,concat_df], axis = 0, ignore_index=True)
            self.csv_merged.append(merge_df)
            self.csv_merged.append(gap)
        else:
            self.csv_merged.append(concat_df)
            self.csv_merged.append(gap)
    
    def export_time_series(self, current_time, z, d, wl, H, v):
        z_array = z[z.shape[0]//2, :]
        z_array = z_array.reshape(-1, 1)
        z_df = pd.DataFrame(z_array)
        
        d_array = d[d.shape[0]//2, :]
        d_array = d_array.reshape(-1, 1)
        d_df = pd.DataFrame(d_array)
        
        wl_array = wl[wl.shape[0]//2, :]   
        wl_array = wl_array.reshape(-1, 1)
        wl_df = pd.DataFrame(wl_array)
        
        H_array = H[H.shape[0]//2, :]
        H_array = H_array.reshape(-1, 1)
        H_df = pd.DataFrame(H_array)
        
        v_array = v[v.shape[0]//2, :]
        v_array = v_array.reshape(-1, 1)
        v_df = pd.DataFrame(v_array)  

        WL_name = self.WL_output +"WL" + str(np.around(current_time, decimals = 1)) + '.csv'
        # WL_name = self.WL_output +"WL" + str(current_time) + '.csv'
        concat_df = pd.concat([z_df, d_df, wl_df, H_df, v_df], axis=1)
        concat_df.to_csv(WL_name)
        concat_df.to_csv(WL_name, header = ['z', 'd', 'wl', 'H', 'v'])
            
    def close_simulation(self, name, iteration, current_time, delta_t_new):
        print("\n .....closing and reporting the CADFFM simulation.....")
        self.report_screen(iteration, delta_t_new, current_time)
        self.report_file()
        self.export_merged_all(True, True)
        print("\n", time.ctime(), "\n")
    
    def report_screen(self, iteration, delta_t_new, current_time):

        self.remaining_volume = np.sum(self.d[:,:-1]) * self.cell_area
        print("iteration: ", iteration)
        print("simulation time: ", "{:.3f}".format(current_time))
        print('delta_t: ', "{:.3e}".format(delta_t_new))
        print('min_Head: ', "{:.3e}".format(self.min_Head))
        print('max velocity (u, v): ', "{:.3f}".format(np.max(np.absolute(self.u))),
              ", ", "{:.3f}".format(np.max(np.absolute(self.v))))
        print('water depth (min, max): ', "{:.3f}".format(np.min(self.d)),
              ", ", "{:.3f}".format(np.max(self.d)))
        print('remaining excess volume: ', "{:.3f}".format(np.sum(self.excess_volume_map))) 
        print('volume of water (initial, current): ',  "{:.3f}".format(self.initial_volume), ", ", "{:.3f}".format(self.remaining_volume))
        print("\n")
        
    def report_file(self, name = None):
        if name is None:
            name = self.outputs_name

        if not os.path.exists(self.outputs_path):
            os.mkdir(self.outputs_path)

        # indices = np.logical_and(self.ClosedBC == False, self.OpenBC == False)
        indices = self.ClosedBC.copy() 
        # indices = ~indices
        
        wl = np.copy(self.WL)
        wl[indices] = self.z[indices]
        fn = self.outputs_path + name + '_wl.tif'
        ut.ArrayToRaster(wl, fn, self.dem_file, mask = None)

        wd = np.copy(self.d)
        wd[indices] = 0
        mask = np.where(wd > 0, 1, 0)
        fn = self.outputs_path + name + '_wd.tif'
        ut.ArrayToRaster(wd, fn, self.dem_file, mask)

        mwd = np.copy(self.max_WD)
        mwd[indices] = 0
        mask = np.where(mwd > 0, 1, 0)
        fn = self.outputs_path + name + '_mwd.tif'
        ut.ArrayToRaster(mwd, fn, self.dem_file, mask)
        
        max_vel = np.copy(self.max_vel)
        max_vel[indices] = 0
        mask = np.where(max_vel > 0, 1, 0)
        fn = self.outputs_path + name + '_max_vel.tif'
        ut.ArrayToRaster(max_vel, fn, self.dem_file, mask)
                
        u = np.copy(self.u)
        v = np.copy(self.v)
        u[indices] = 0
        v[indices] = 0
        mask = np.where(u**2 + v**2 > 0, 1, 0)
        fn = self.outputs_path + name + '_u_v.tif'
        ut.VelocitytoRasterIO(u, v, fn, self.dem_file, mask)
        
        
        
        