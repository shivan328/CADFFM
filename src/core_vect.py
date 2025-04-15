import numpy as np
import util as ut
import os
import sys
import time
import pandas as pd

#Quadratic Equation Solver
def quad_Eq(a, b, c, sign):
    """
    Solve the quadratic equation ax^2 + bx + c = 0 using sign-based filtering.
    Used in velocity calculations where Bernoulli head and friction losses are modeled.
    """
    discriminant = b**2 - 4*a*c
    # Ensure discriminant is non-negative to avoid sqrt of negative number
    discriminant = np.clip(discriminant, 0, None)
    # Compute square root safely (avoid sqrt of negative values)
    sqrt_discriminant = np.where(discriminant >= 0, np.sqrt(discriminant), 0)

    # Compute both solutions
    denom = 2 * a
    x1 = (-b + sqrt_discriminant) / denom
    x2 = (-b - sqrt_discriminant) / denom

    # Apply sign convention
    solution1 = np.where(sign * x1 > 0, sign * x1, 0)
    solution2 = np.where(sign * x2 > 0, sign * x2, 0)

    # Select minimum nonzero solution if both exist, otherwise take maximum
    solution = np.where((solution1 > 0) & (solution2 > 0), 
                        np.minimum(solution1, solution2), 
                        np.maximum(solution1, solution2))

    return solution*sign

def non_zero_min(a, b):
    min_val = np.minimum(a, b)

    # Compute sum where at least one of them is zero
    sum_val = a + b

    # Create a mask where either a or b is zero
    zero_mask = (a == 0) | (b == 0)

    # Use np.where to apply the condition element-wise
    return np.where(zero_mask, sum_val, min_val)
    
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
        self.u = np.zeros_like(self.z, dtype=np.double)
        self.v = np.zeros_like(self.z, dtype=np.double)
        self.d = np.zeros_like(self.z, dtype=np.double)
        self.WL = np.zeros_like(self.z, dtype=np.double)
        self.max_WD = np.zeros_like(self.z, dtype=np.double)
        self.max_vel = np.zeros_like(self.z, dtype=np.double)
        self.vel_res = np.zeros_like(self.z, dtype=np.double)
        self.n0 = np.zeros_like(self.z, dtype=np.double)
        self.theta = np.array([1, 1, -1, -1])
        self.excess_volume_map = np.zeros_like(self.z, dtype=np.float64)
        self.OBC_cells = np.array([])
        self.OBC_cells_outlet = np.array([])
        self.CBC_cells = np.array([])
        self.OBC_depth = np.array([])
        self.H_i_dir = np.zeros(4, dtype=np.int8)
        self.Q_i_dir = np.zeros(4, dtype=np.int8)
        self.vel_ini = np.zeros((self.dem_shape[0] - 2, self.dem_shape[1] - 2, 4), dtype=np.float64)
        
    # set simulation time and timestep
    def set_simulation_time(self, t, delta_t, delta_t_bias=0):
        self.t = t
        self.delta_t = delta_t
        self.delta_t_bias = delta_t_bias
    
    # set output path
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
            
    # set the open boundary cells for outflows inside the domain
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
    
    # set the closed boundary cells
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
        # Elevate closed boundary cells to artificially prevent flow
        self.z[self.ClosedBC] += self.BCtol
    
    def reset_BCs(self):
        # it restores boundary cells' elevation to normal
        self.z[self.ClosedBC] -= self.BCtol
        # it sets the water depth of open boundary cells from benchmark model
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
        velocity_magnitude = np.maximum(velocity_magnitude, 1e-10)                      # Avoid division by zero
        
        return self.CFL * np.min(self.cell_length/ velocity_magnitude)

    # Calculate the Bernoulli head
    def compute_Bernoulli_head(self):
        
        return self.z + self.d + (self.u**2 + self.v**2)/(2*self.g)
    
    # normal flow direction 
    def normal_flow_direction(self, u0, v0, H):
    # Ensure H_i_dir has the same shape as H_diff
        self.H_i_dir = np.zeros_like(H[..., 1:5], dtype=np.int8)  # Shape (3, 100, 4)
        self.Q_i_dir = np.zeros_like(H[..., 1:5], dtype=np.int8)  # Shape (3, 100, 4)
        H_diff = H[..., 0:1] - H[..., 1:5]                        #head difference
        self.H_i_dir[H_diff > self.min_Head] = 1                  # Apply tolerance
        
        # Compute flow direction
        Q_i = np.array([u0, v0, u0, v0]) * self.theta[:, np.newaxis, np.newaxis]  # Shape (4,3,100)
        Q_i = Q_i.transpose(1, 2, 0)  # Convert shape to (3,100,4) to match self.Q_i_dir
        np.around(Q_i, 3, out=Q_i)      # round to 3 decimal places to omit small velocities
        self.Q_i_dir[Q_i >= 0] = 1        
        self.Q_i_dir *= self.H_i_dir

        return self.Q_i_dir

    # Calculate if (Q*theta)>0 for Special Flow Condition
    def special_flow_direction(self, u0, v0, H):
        # Ensure H_i_dir has the same shape as H_diff
        self.H_i_dir = np.zeros_like(H[..., 1:5], dtype=np.int8)  # Shape (3, 100, 4)
        self.Q_i_dir = np.zeros_like(H[..., 1:5], dtype=np.int8)  # Shape (3, 100, 4)
        H_diff = H[..., 1:5] - H[..., 0:1]                        # head difference in reverse order compared to normal flow
        self.H_i_dir[H_diff > self.min_Head] = 1                  # Apply tolerance
        
        # Compute flow direction using velocity components
        Q_i = np.array([u0, v0, u0, v0]) * self.theta[:, np.newaxis, np.newaxis]  # Shape (4,3,100)
        Q_i = Q_i.transpose(1, 2, 0)  # Convert shape to (3,100,4) to match self.Q_i_dir
        self.Q_i_dir[Q_i > 0] = 1  # Set flow direction for positive values
        self.Q_i_dir *= self.H_i_dir  # Apply head difference condition

        return self.Q_i_dir

    # Calculate mass flux for normal flow condition
    def compute_normal_flow_mass_flux_i(self, n0, H, d, z, i):
        # Vectorized way to calculate `z_bar_i`
        z_bar_i = np.maximum(z[:,:,0:1], z[:,:,1:5])
        h0_i = H[:,:, 0:1] - z_bar_i
        h_i = H[:,:,1:5] - z_bar_i
        d0_i = (d[:,:,0:1] + z[:,:,0:1]) - z_bar_i
        d_i = (d[:,:,1:5] + z[:,:,1:5]) - z_bar_i
        ratio_h = np.where(h0_i > 0, np.divide(h_i, h0_i, out=np.zeros_like(h_i), where=h0_i > 0), 1)
        ratio_d = np.where((h0_i > 0) & (d0_i > 0), np.divide(d_i, d0_i, out=np.zeros_like(d_i), where=d0_i > 0), 1)
        min_ratio = np.minimum(ratio_h, ratio_d)        # Min ratio means max flux
        # Compute Weir flux 
        psi_i = (1 - np.clip(min_ratio, 0, 1)**1.5)**0.385  
        Q_weir = np.where(min_ratio < 1, self.weir_eq_const * psi_i * np.clip(h0_i, 0, None)**1.5, 0)
        
        # Compute Mannings flux
        Hloc = H[:,:,0:1] - H[:,:,1:5]
        d0 = d[:,:,0:1]                             # Ensure shapes match for broadcasting
        n0 = n0[:,:,np.newaxis]
        Q_mannings = (self.cell_length / n0) * d0**(5/3) * (np.clip(Hloc, 0, None) / self.cell_length)**0.5
        
        # Compute final flow
        Q = non_zero_min(Q_mannings, Q_weir) * self.theta[i]
        
        return Q, ratio_d, h0_i

    # Calculate mass flux for special flow condition
    def compute_special_flow_mass_flux_i(self, d, u, v, H, deltat, i, ratio_d, h0_i):

        dQ_1 = self.cell_area / (2 * deltat) * (H[:,:,1:5] - H[:,:,0:1] + self.min_Head)
        dQ_2 = self.cell_area / deltat * (d[:,:,1:5] - self.min_WD)
        dQ = np.minimum(dQ_1, dQ_2)
        psi_i = (1 - np.clip(ratio_d, 0, 1)**1.5)**0.385
        Q_weir = np.where(ratio_d < 1, self.weir_eq_const * psi_i * np.clip(h0_i, 0, None)**1.5, 0)
        Q_vel = np.zeros_like(d[:, :, 0:1], dtype=np.float64)

        # Apply velocity conditions
        u_component = u[:, :, 0:1] * self.theta[i] * self.cell_length * d[:, :, 0:1]
        v_component = v[:, :, 0:1] * self.theta[i] * self.cell_length * d[:, :, 0:1]
        
        # Use velocity magnitude to compute directional flow
        Q_vel = np.where((i == 0) | (i == 2), u_component, v_component)
        Q_vel = np.clip(Q_vel, 0, None)  # Ensure non-negative velocity flow
        
        # Use velocity-driven flow if it's not zero
        Q = np.where(Q_vel != 0, Q_vel, Q_weir)

        # Apply mass conservation check
        Q = np.where(np.abs(Q) > dQ, np.abs(Q) - dQ, 0)
    
        return Q * self.theta[i]
    
    def compute_velocity(self, FD, n, d, z, u, v, H0, d0):
        self.vel_ini.fill(0)
        eps = 1e-8  # Small number to prevent division by zero
        d_safe = np.where(d > eps, d, eps)
        d0 = d0[:,:,np.newaxis]
        d0_safe = np.where(d0 > eps, d0, eps)
        H0 = H0[:,:,np.newaxis]
        FD_mask = (FD == 1)
        
        # for x-direction
        i_x = np.array([0, 2])
        self.theta_x = self.theta[i_x][np.newaxis, np.newaxis, :]
        b_x = self.theta_x * self.half_cell_size * n[..., i_x+1]**2 * np.abs(u[..., i_x+1]) / d_safe[..., i_x+1]**c4_3
        c_x = (v[..., i_x+1]**2 * self.gravity_inverse + d_safe[..., i_x+1] + z[..., i_x+1] +
            self.half_cell_size * n[..., 0:1]**2 * u[..., 0:1]**2 / d0_safe**c4_3 - H0)
        self.vel_ini[..., i_x] = np.where(FD_mask[..., i_x], quad_Eq(self.gravity_inverse, b_x, c_x, self.theta_x), 0)
        
        # for y-direction
        i_y = np.array([1, 3])  # North (index 2), South (index 4)
        self.theta_y = self.theta[i_y][np.newaxis, np.newaxis, :]
        b_y = self.theta_y * self.half_cell_size * n[..., i_y+1]**2 * np.abs(v[..., i_y+1]) / (d_safe[..., i_y+1]**c4_3)
        c_y = (u[..., i_y+1]**2 * self.gravity_inverse + d_safe[..., i_y+1] + z[..., i_y+1] +
                self.half_cell_size * n[..., 0:1]**2 * v[..., 0:1]**2 / d0_safe**c4_3 - H0)
        self.vel_ini[..., i_y] = np.where(FD_mask[..., i_y], quad_Eq(self.gravity_inverse, b_y, c_y, self.theta_y), 0)

        return self.vel_ini
    
    # set the excess volume map for inflows in 2D domain
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
            
    # main function to run the simulation
    def run_simulation(self, catchment):
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
        mask = self.ClosedBC.copy()                                 #masked to use boundary condition
        factor = 1e6
        self.vel_res = self.u**2 + self.v**2
        FD = np.zeros_like(self.z, dtype=np.int8)
        delta_d= np.zeros_like(self.z, dtype=np.float64)
        d_new = np.zeros_like(self.z, dtype=np.float64)                  # new water depth
        u_new = np.zeros_like(self.u, dtype=np.float64)
        v_new = np.zeros_like(self.v, dtype=np.float64)
        
        # Precompute indices for computational cells
        # Exclude boundary cells
        i, j = np.arange(1, self.dem_shape[0] - 1), np.arange(1, self.dem_shape[1] - 1)
        # Create 2D meshgrid for indexing
        I, J = np.meshgrid(i, j, indexing='ij')
        
        while current_time < self.t:
            self.set_BCs()          # set boundary conditions
            
            FD.fill(0)
            delta_d.fill(0)
            u_new.fill(0)
            v_new.fill(0)
            
            # input water in 2D catchment
            if np.sum(self.excess_volume_map) > 0:
                current_volume = self.excess_volume_map / (self.t - current_time) * delta_t
                self.d += current_volume / self.cell_area
                self.excess_volume_map -= current_volume
                
            H = self.compute_Bernoulli_head()         # calculate Bernoulli head
            Max_KE = np.max(self.vel_res * self.d)
            self.min_Head = max(Max_KE * initial_min_head, 1e-6)
            
            # Extract central and neighbor values
            z_central = self.z[I, J]
            z_east, z_north, z_west, z_south = self.z[I, J+1], self.z[I-1, J], self.z[I, J-1], self.z[I+1, J]
            d_central = self.d[I, J]
            d_east, d_north, d_west, d_south = self.d[I, J+1], self.d[I-1, J], self.d[I, J-1], self.d[I+1, J]
            u_central = self.u[I, J]
            u_east, u_north, u_west, u_south = self.u[I, J+1], self.u[I-1, J], self.u[I, J-1], self.u[I+1, J]
            v_central = self.v[I, J]
            v_east, v_north, v_west, v_south = self.v[I, J+1], self.v[I-1, J], self.v[I, J-1], self.v[I+1, J]
            H_central = H[I, J]
            H_east, H_north, H_west, H_south = H[I, J+1], H[I-1, J], H[I, J-1], H[I+1, J]

            # Create arrays
            z = np.stack([z_central, z_south, z_east, z_north, z_west], axis=-1)
            d = np.stack([d_central, d_south, d_east, d_north, d_west], axis=-1)
            u = np.stack([u_central, u_south, u_east, u_north, u_west], axis=-1)
            v = np.stack([v_central, v_south, v_east, v_north, v_west], axis=-1)
            H_loc = np.stack([H_central, H_south, H_east, H_north, H_west], axis=-1)
            
            # Check where cells are wet
            wet_mask = self.d[1:-1,1:-1] > self.min_WD
            wet_mask_3d= wet_mask[..., np.newaxis]
            
            # Compute normal flow directions
            normal_flow = self.normal_flow_direction(u[:, :, 0], v[:, :, 0], H_loc)

            # Compute normal flow mass flux
            Qn = np.zeros_like(d[:, :, 1:], dtype=np.float64)
            cond_nm = np.bitwise_and(normal_flow > 0, wet_mask_3d)
            Qn, ratio_d, h0_i = np.where(cond_nm, self.compute_normal_flow_mass_flux_i(self.n0[I, J], H_loc, d, z, np.arange(4)), 0)

            # Compute special flow
            special_flow = np.where(d[:, :, 1:] > self.min_WD, 1, 0)
            special_flow *= self.special_flow_direction(u[:, :, 0], v[:, :, 0], H_loc)

            # Compute special flow mass flux
            Qs = np.zeros_like(d[:, :, 1:], dtype=np.float64)
            cond_sp = np.bitwise_and(special_flow > 0, wet_mask_3d)
            Qs = np.where(cond_sp, self.compute_special_flow_mass_flux_i(d, u, v, H_loc, delta_t, np.arange(4), ratio_d, h0_i), 0)

            # Compute final flow flux values
            Q = np.trunc((Qn + Qs) * factor) / factor                   # factor to avoid floating point errors
            
            # Convert to binary flow direction
            FD_loc = (Q != 0).astype(np.int8)
            
            # Update water depth
            delta_d[I, J] += np.sum(-self.theta * Q, axis=-1) / self.cell_area
            delta_d[I+1, J] += (self.theta[0] * Q[..., 0]) / self.cell_area
            delta_d[I, J+1] += (self.theta[1] * Q[..., 1]) / self.cell_area
            delta_d[I-1, J] += (self.theta[2] * Q[..., 2]) / self.cell_area
            delta_d[I, J-1] += (self.theta[3] * Q[..., 3]) / self.cell_area
                        
            d_new = self.d + delta_d * delta_t

            # adaptive time stepping => find the min required timestep
            # this section avoids over drying cells (negative d)
            if (d_new < 0).any():
                print("Negative water depth detected")
                neg_indices = np.where(d_new < 0)
                tmp_values = np.abs(self.d[neg_indices] / delta_d[neg_indices])
                # for better stability, devided by 10 
                delta_t_new = np.min(tmp_values) / 10
                d_new = self.d + delta_d * delta_t_new
            else:
                delta_t_new = delta_t
            current_time += delta_t_new
            
            # this loop calculate velocities in both directions
            # extract central cells and neighbors
            d_n_central = d_new[I, J]
            d_n_east, d_n_north, d_n_west, d_n_south = d_new[I, J+1], d_new[I-1, J], d_new[I, J-1], d_new[I+1, J]
            n_central = self.n0[I, J]
            n_east, n_north, n_west, n_south = self.n0[I, J+1], self.n0[I-1, J], self.n0[I, J-1], self.n0[I+1, J]
            d_n = np.stack([d_n_central, d_n_south, d_n_east, d_n_north, d_n_west], axis=-1)
            n = np.stack([n_central, n_south, n_east, n_north, n_west], axis=-1)
           
            # Compute flow velocities
            velocities = np.where(wet_mask_3d, self.compute_velocity(FD_loc, n, d_n, z, u, v, H[I, J], self.d[I, J]), 0)

            # Apply velocity updates based on flow directions
            u_new[I-1, J] += velocities[..., 2]  # North
            u_new[I+1, J] += velocities[..., 0]  # South
            v_new[I, J-1] += velocities[..., 3]  # West
            v_new[I, J+1] += velocities[..., 1]  # East
            
            # velocity update and modification
            eps = 1e-8  # to prevent division by zero

            # Slices for the interior cells
            u_n = u_new[1:-1, 1:-1]
            v_n = v_new[1:-1, 1:-1]

            # Square speeds from the new velocities
            curr_u_sqr = u_n**2
            curr_v_sqr = v_n**2
            vel_new_head = curr_u_sqr + curr_v_sqr
            safe_vel_new_head = np.clip(vel_new_head, eps, None)

            # Use the current (old) velocities in the interior for comparisons
            u_old = self.u[1:-1, 1:-1]
            v_old = self.v[1:-1, 1:-1]

            # Compute vel_max_head exactly as in the loop:
            # (It is computed using the old velocity for one component and the new for the other.)
            vel_max_head = np.maximum(u_old**2 + curr_v_sqr, curr_u_sqr + v_old**2)

            # Prepare temporary arrays to hold the updates
            u_temp = u_old.copy()
            v_temp = v_old.copy()         
            
            # (a) First branch: if u_new and v_new are nonzero and old velocities are zero.
            valid_update = (u_n != 0) & (v_n != 0) & (u_old == 0) & (v_old == 0)
            max_vel = np.maximum(np.abs(u_n), np.abs(v_n))
            u_temp = np.where(valid_update, u_n / np.sqrt(safe_vel_new_head) * max_vel, u_temp)
            v_temp = np.where(valid_update, v_n / np.sqrt(safe_vel_new_head) * max_vel, v_temp)

            # (b) Second branch: if u_new and v_new are nonzero, and new head exceeds vel_max_head,
            # but exclude cells already handled by valid_update.
            exceeds_max_head = (u_n != 0) & (v_n != 0) & (~((u_old == 0) & (v_old == 0))) & (vel_new_head > vel_max_head)

            # Define conditions based on the relative sizes of |u_new| and |v_new|
            less_u    = np.abs(u_n) < np.abs(v_n)
            less_v    = np.abs(v_n) < np.abs(u_n)
            equal_uv  = np.abs(u_n) == np.abs(v_n)

            # Apply the branch when |u_new| < |v_new|
            u_temp = np.where(exceeds_max_head & less_u, (u_old + u_n) / 2, u_temp)
            v_temp = np.where(exceeds_max_head & less_u,
                            np.sqrt(vel_max_head - u_temp**2) * np.sign(v_n),
                            v_temp)

            # When |u_new| > |v_new|
            v_temp = np.where(exceeds_max_head & less_v, (v_old + v_n) / 2, v_temp)
            u_temp = np.where(exceeds_max_head & less_v,
                            np.sqrt(vel_max_head - v_temp**2) * np.sign(u_n),
                            u_temp)

            # When |u_new| == |v_new|
            u_temp = np.where(exceeds_max_head & equal_uv, u_n / np.sqrt(2), u_temp)
            v_temp = np.where(exceeds_max_head & equal_uv, v_n / np.sqrt(2), v_temp)

            # (c) Else branch: if not already updated by (a) or (b), then update if nonzero.
            # (The loop says “if u_new != 0 then u = u_new”, etc.)
            # We only update those cells that have not been updated in (a) or (b).
            other = ~valid_update & ~exceeds_max_head
            u_temp = np.where(other & (u_n != 0), u_n, u_temp)
            v_temp = np.where(other & (v_n != 0), v_n, v_temp)

            # --- Boundary conditions (neighbors) ---
        
            # Set u to zero if the north or south neighbor (in mask) is True.
            u_temp = np.where(mask[:-2, 1:-1] | mask[2:, 1:-1], 0, u_temp)
            # Set v to zero if the west or east neighbor (in mask) is True.
            v_temp = np.where(mask[1:-1, :-2] | mask[1:-1, 2:], 0, v_temp)

            # # --- Diagonal conditions ---
            # # For the interior, define slices for the neighbors:
            # mask_N = mask[:-2,   1:-1]   # north
            # mask_S = mask[2:,    1:-1]   # south
            # mask_E = mask[1:-1,  2:]     # east
            # mask_W = mask[1:-1,  :-2]    # west
            # mask_NE = mask[:-2,  2:]     # northeast
            # mask_NW = mask[:-2,  :-2]    # northwest
            # mask_SE = mask[2:,   2:]     # southeast
            # mask_SW = mask[2:,   :-2]    # southwest

            # # Diagonals – exactly following the loop:
            # # 1. If mask[i,j+1] and mask[i-1,j]:
            # #    (a) if not mask[i+1,j+1]: set u = v_new.
            # #    (b) if not mask[i-1,j-1]: set v = u_new.
            # u_temp = np.where((mask_N & mask_E & ~mask_SE & ~mask_S), v_n, u_temp)
            # v_temp = np.where((mask_N & mask_E & ~mask_NW & ~mask_W), u_n, v_temp)

            # # 2. If mask[i,j+1] and mask[i+1,j]:
            # #    (a) if not mask[i-1,j+1]: set u = -v_new.
            # #    (b) if not mask[i+1,j-1]: set v = -u_new.
            # u_temp = np.where((mask_S & mask_E & ~mask_NE & ~mask_N), -v_n, u_temp)
            # v_temp = np.where((mask_S & mask_E & ~mask_SW & ~mask_W), -u_n, v_temp)

            # # 3. If mask[i,j-1] and mask[i-1,j]:
            # #    (a) if not mask[i+1,j-1]: set u = -v_new.
            # #    (b) if not mask[i-1,j+1]: set v = -u_new.
            # u_temp = np.where((mask_N & mask_W & ~mask_SW & ~mask_S), -v_n, u_temp)
            # v_temp = np.where((mask_N & mask_W & ~mask_NE & ~mask_E), -u_n, v_temp)

            # # 4. If mask[i,j-1] and mask[i+1,j]:
            # #    (a) if not mask[i-1,j-1]: set u = v_new.
            # #    (b) if not mask[i+1,j+1]: set v = u_new.
            # u_temp = np.where((mask_S & mask_W & ~mask_NW & ~mask_N), v_n, u_temp)
            # v_temp = np.where((mask_S & mask_W & ~mask_SE & ~mask_E), u_n, v_temp)

            # --- Finally, set velocities to zero where d_new <= min_WD ---
            u_temp[~wet_mask] = 0
            v_temp[~wet_mask] = 0

            # Write the computed interior back to the full arrays.
            self.u[1:-1, 1:-1] = u_temp
            self.v[1:-1, 1:-1] = v_temp
            
            # Set velocities to zero at the boundary cells
            self.u[self.OpenBC] = 0
            self.v[self.OpenBC] = 0
            
            self.v = np.trunc(self.v * factor) / factor             # factor to avoid floating point errors
            self.u = np.trunc(self.u * factor) / factor 
   
            self.d = d_new                                          # update water depth
            self.inflows_to_nodes(delta_t_new)
            self.reset_BCs()
            
            self.max_WD = np.maximum(self.max_WD, self.d)
            self.vel_res = self.u**2 + self.v**2
            self.max_vel = np.maximum(self.max_vel, np.sqrt(self.vel_res))
            self.WL = self.d + self.z
            
            # to print the outputs for time steps/iterations
            if iteration % 100 == 0:
                self.report_screen(iteration, delta_t_new, current_time)
                # self.report_file()

            # time step adjustment/ adaptive time step
            delta_t = self.CFL_delta_t()
            if delta_t_new < delta_t:
                delta_t = (delta_t_new + (self.delta_t_bias + 1)
                           * delta_t) / (self.delta_t_bias + 2)

            # self.time_step.append([current_time, delta_t_new])
            if not catchment:
                if current_time >= t_value:
                    Time = True
                    t_value += 0.1

            # to export the water level and velocity at specified times
            if not catchment:
                if Time: 
                    self.export_time_series(current_time, self.z, self.d, self.d + self.z , H, self.v)
                    Time = False  
                            
            iteration += 1

        self.delta_t = delta_t
        self.close_simulation(self.outputs_name, iteration, current_time, delta_t_new)
        print("\nSimulation finished in", (time.time() - self.begining),
        "seconds")
    
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
        self.report_file(name)
        # self.export_merged_all(True, True)
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
        
        
        
        