"""
JAX-based implementation of the BIO formulation 
for 2D Stokes squirmer in a confinement
"""

import jax

import jax.numpy as jnp
from jax import jit
from functools import partial
import sys, os
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the folder containing the module you want to import
utils_path = os.path.join(current_dir, '..')
# Add the folder to the system path
sys.path.append(utils_path)
from Sto_kernel_utils_pytree import *

@jit
def rbm_wrapper(s, obs_cell, ptcl_cell, mu):
    if not obs_cell: # only active ptcl and channel
        E, A, intF, intT= rbm(s, ptcl_cell, mu)
    else: 
        E, A, intF, intT = rbm_obs(s, obs_cell, ptcl_cell, mu)
    return E, A, intF, intT

@jit
def rbm(s, ptcl_cell, mu):
    """
    A DL formulation on wall (like before) and SL+DL on particle -- Rigid body motion
    Close-eval included in all non-self interactions.
    """
    Nsx = s['x'].size
    # Wall self-to-self
    [Asrcsum,_,_] = StoDLP(s, s, mu, jnp.array([]), self=True, near=False, panel=True)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    # Particle to wall from all ptcls in ptcl_cell, summed
    ptcl_keys = ptcl_cell.keys()
    num_ptcl = len(ptcl_keys)
    A12_list = [StoDLP(s, ptcl_cell[k], mu, jnp.array([]), self=False, near=True, panel=False)[0] + StoSLP(s, ptcl_cell[k], mu,  jnp.array([]), self=False, near=True, panel=False)[0] for k in ptcl_keys]
    A12 = jnp.hstack(list(A12_list))
    # Wall to particle
    A21_tuples = [StoDLP(ptcl_cell[k], s, mu, jnp.array([]), self = False, near = True, panel = True) for k in ptcl_keys]
    A21_u_list, _, A21_T_list = zip(*A21_tuples)
    A21 = jnp.vstack(A21_u_list)
    A21_T = jnp.vstack(A21_T_list)

    # FROM ptcl_j (with near) to ptcl_k
    def compute_pair_interaction(key_j, key_k):
        p_j = ptcl_cell[key_j]
        p_k = ptcl_cell[key_k]
        def self_case(_):
            # Diagonal block: Particle interacting with itself (No jump)
            res_sl, _, res_slT = StoSLP(p_j, p_j, mu, jnp.array([]), self=True, near=False, panel=False)
            res_dl, _, _ = StoDLP(p_j, p_j, mu, jnp.array([]), self=True, near=False, panel=False) # self-to-self DL traction is always 0.
            return res_sl + res_dl, res_slT

        def cross_case(_):
            # Off-diagonal block: source particle j, target particle k
            # res_sl, _, res_sl_T = StoSLP(p_k, p_j, mu, jnp.array([]), False)
            # res_dl, _, res_dl_T = StoDLP(p_k, p_j, mu, jnp.array([]), False)
            res_sl, _, res_sl_T = stoSLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e') # Use close-eval to handle close-together particles.
            res_dl, _, res_dl_T = stoDLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e')
            return res_sl + res_dl, res_sl_T + res_dl_T
        
        return jax.lax.cond(key_j == key_k, self_case, cross_case, operand=None)

    A22_dict = {(key_j, key_k): compute_pair_interaction(key_j, key_k) for key_j in ptcl_cell.keys() for key_k in ptcl_cell.keys()}
    grid_u = []
    grid_T = []
    for key_k in ptcl_keys:
        row_u = []
        row_T = []
        for key_j in ptcl_keys:
            if (key_j,key_k) in A22_dict:
                Akj_block = A22_dict[(key_j, key_k)][0]
                Tkj_block = A22_dict[(key_j, key_k)][1]
            else:
                raise Exception(f"ERROR: interaction between particles {key_j} and {key_k} is not stored.")
            row_u.append(Akj_block)
            row_T.append(Tkj_block)
        grid_u.append(row_u)
        grid_T.append(row_T)

    A22 = jnp.block(grid_u)
    A22 = jnp.eye(A22.shape[0]) / 2. + A22 
    A = jnp.block([[A11, A12],
                   [A21, A22]])

    # Rigid body motion added to b.c.
    Nptx_total = sum(len(ptcl_cell[k]['x']) for k in ptcl_keys)
    total_rows = 2 * Nsx + 2 * Nptx_total
    total_cols = total_rows + 3 * num_ptcl
    bc_sqr = jnp.zeros((2 * Nptx_total, 3 * num_ptcl)) # just the ptcl BC part

    current_row_offset = 0
    F_blocks = []
    T_blocks = [] # store values now for force and torque integrals
    for i, k in enumerate(ptcl_keys):
        pt = ptcl_cell[k]  # Complex points: x + iy
        n_nodes = len(pt['x'])
        # X - Xc
        xc = jnp.mean(pt['x'])
        rel_pos = pt['x'] - xc
        dx = jnp.real(rel_pos)
        dy = jnp.imag(rel_pos)
        # Indices
        ptcl_x_rows = jnp.arange(current_row_offset, current_row_offset + n_nodes)
        ptcl_y_rows = jnp.arange(current_row_offset + n_nodes, current_row_offset + 2 * n_nodes)
        col_ux = 3 * i
        col_uy = 3 * i + 1
        col_omega = 3 * i + 2

        bc_sqr = bc_sqr.at[ptcl_x_rows, col_ux].set(-1.0)
        bc_sqr = bc_sqr.at[ptcl_y_rows, col_uy].set(-1.0)
        bc_sqr = bc_sqr.at[ptcl_x_rows, col_omega].set(-dy)
        bc_sqr = bc_sqr.at[ptcl_y_rows, col_omega].set(dx)

        w_row = pt['ws'][None, :] # shape (1, n_nodes)
        z_row = jnp.zeros_like(w_row)
        f_block = jnp.block([
            [w_row, z_row],
            [z_row, w_row]
        ])
        F_blocks.append(f_block)
        t_block = jnp.concatenate([(-dy * pt['ws'])[None, :], (dx * pt['ws'])[None, :]], axis=1)
        T_blocks.append(t_block)

        current_row_offset += 2 * n_nodes

    # Final assembly
    bc_gamma_mat = jnp.zeros((total_rows, total_cols))
    bc_gamma_mat = bc_gamma_mat.at[:, :total_rows].set(A)
    bc_gamma_mat = bc_gamma_mat.at[2*Nsx:, total_rows:].set(bc_sqr)

    # Net force zero and net torque zero
    T3 = jnp.block(grid_T)
    fMat_ptcl = - ( -jnp.eye(T3.shape[0])/2. + T3)
    fMat_wall = - A21_T

    intF_op = jax.scipy.linalg.block_diag(*F_blocks)
    intT_op = jax.scipy.linalg.block_diag(*T_blocks)

    mid_zeros = jnp.zeros((2 * Nptx_total, 3 * num_ptcl))
    big_mat = jnp.concatenate([fMat_wall, fMat_ptcl, mid_zeros], axis=1)
    intF = intF_op @ big_mat
    intT = intT_op @ big_mat

    # --- Full ELS matrix ---
    E = jnp.block([[bc_gamma_mat],
                    [intF],
                   [intT]])


    return E, bc_gamma_mat, intF, intT

@jit
def rbm_obs(s, obs_cell, ptcl_cell, mu):
    """
    A DL formulation on wall (like before)
     SL+DL formulation on obstacles (passive and often fixed particle) 
     and SL+DL on active particle with Rigid body motion
    Close-eval included in all non-self interactions.
    """
    Nsx = s['x'].size
    # Wall self-to-self
    [Asrcsum,_,_] = StoDLP(s, s, mu, jnp.array([]), self=True, near=False, panel=True)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    # Obstacle to wall from all obs in obs_cell, summed
    obs_keys = obs_cell.keys()
    num_obs = len(obs_keys)
    A12_obs_list = [StoDLP(s, obs_cell[k], mu, jnp.array([]), self=False, near=True, panel=False)[0] + StoSLP(s, obs_cell[k], mu, jnp.array([]), self=False, near=True, panel=False)[0] for k in obs_keys]
    # Particle to wall from all ptcls in ptcl_cell, summed
    ptcl_keys = ptcl_cell.keys()
    num_ptcl = len(ptcl_keys)
    A12_ptcl_list = [StoDLP(s, ptcl_cell[k], mu, jnp.array([]), self=False, near=True, panel=False)[0] + StoSLP(s, ptcl_cell[k], mu, jnp.array([]), self=False, near=True, panel=False)[0] for k in ptcl_keys]
    A12_obs_list.extend(A12_ptcl_list) 
    A12 = jnp.hstack(A12_obs_list) 
    # Wall to obstacle
    A21_obs_tuples = [StoDLP(obs_cell[k], s, mu, jnp.array([]), self=False, near=True, panel=True) for k in obs_keys]
    A21_obs_u_list, _, _ = zip(*A21_obs_tuples) # don't need traction on obstacles
    A21_obs_u_list = list(A21_obs_u_list) # zip outputs tuples so switch to list
    # Wall to particle
    A21_ptcl_tuples = [StoDLP(ptcl_cell[k], s, mu, jnp.array([]), self=False, near=True, panel=True) for k in ptcl_keys]
    A21_ptcl_u_list, _, A21_ptcl_T_list = zip(*A21_ptcl_tuples)
    A21_ptcl_u_list = list(A21_ptcl_u_list)
    A21_obs_u_list.extend(A21_ptcl_u_list) 
    A21 = jnp.vstack(A21_obs_u_list)
    A21_T = jnp.vstack(A21_ptcl_T_list)

    # FROM ptcl_j (with near) to ptcl_k 
    def compute_p2p(key_j, key_k):
        p_j = ptcl_cell[key_j]
        p_k = ptcl_cell[key_k]
        def self_case(_):
            # Diagonal block: Particle interacting with itself (No jump)
            res_sl, _, res_slT = StoSLP(p_j, p_j, mu, jnp.array([]), self=True, near=False, panel=False)
            res_dl, _, _ = StoDLP(p_j, p_j, mu, jnp.array([]), self=True, near=False, panel=False) # self-to-self DL traction is always 0.
            return res_sl + res_dl, res_slT

        def cross_case(_):
            # Off-diagonal block: source particle j, target particle k
            # res_sl, _, res_sl_T = StoSLP(p_k, p_j, mu, jnp.array([]), False)
            # res_dl, _, res_dl_T = StoDLP(p_k, p_j, mu, jnp.array([]), False)
            res_sl, _, res_sl_T = stoSLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e') # Use close-eval to handle close-together particles.
            res_dl, _, res_dl_T = stoDLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e')
            return res_sl + res_dl, res_sl_T + res_dl_T
            
        # Set-up of ptcl_cell and obs_cell makes sure keys won't overlap ('ptcl_1' vs 'obs_1', etc)
        return jax.lax.cond(key_j == key_k, self_case, cross_case, operand=None)
    
    # FROM ptcl_j (with near) to obs_k (no self)
    def compute_p2o(key_j, key_k):
        p_j = ptcl_cell[key_j]
        p_k = obs_cell[key_k]
        # Off-diagonal block: source particle j, target particle k
        # res_sl, _, res_sl_T = StoSLP(p_k, p_j, mu, jnp.array([]), False)
        # res_dl, _, res_dl_T = StoDLP(p_k, p_j, mu, jnp.array([]), False)
        res_sl, _, res_sl_T = stoSLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e') # Use close-eval to handle close-together particles.
        res_dl, _, res_dl_T = stoDLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e')
        return res_sl + res_dl, res_sl_T + res_dl_T
    
    # FROM obs_j (with near) to ptcl_k  (no self)
    def compute_o2p(key_j, key_k):
        p_j = obs_cell[key_j]
        p_k = ptcl_cell[key_k]
        # Off-diagonal block: source particle j, target particle k
        # res_sl, _, res_sl_T = StoSLP(p_k, p_j, mu, jnp.array([]), False)
        # res_dl, _, res_dl_T = StoDLP(p_k, p_j, mu, jnp.array([]), False)
        res_sl, _, res_sl_T = stoSLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e') # Use close-eval to handle close-together particles.
        res_dl, _, res_dl_T = stoDLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e')
        return res_sl + res_dl, res_sl_T + res_dl_T
    
    # FROM ptcl_j (with near) to ptcl_k 
    def compute_o2o(key_j, key_k):
        p_j = obs_cell[key_j]
        p_k = obs_cell[key_k]
        def self_case(_):
            # Diagonal block: Particle interacting with itself (No jump)
            res_sl, _, res_slT = StoSLP(p_j, p_j, mu, jnp.array([]), self=True, near=False, panel=False)
            res_dl, _, _ = StoDLP(p_j, p_j, mu, jnp.array([]), self=True, near=False, panel=False) # self-to-self DL traction is always 0.
            return res_sl + res_dl, res_slT

        def cross_case(_):
            # Off-diagonal block: source particle j, target particle k
            # res_sl, _, res_sl_T = StoSLP(p_k, p_j, mu, jnp.array([]), False)
            # res_dl, _, res_dl_T = StoDLP(p_k, p_j, mu, jnp.array([]), False)
            res_sl, _, res_sl_T = stoSLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e') # Use close-eval to handle close-together particles.
            res_dl, _, res_dl_T = stoDLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e')
            return res_sl + res_dl, res_sl_T + res_dl_T
        
        # Set-up of ptcl_cell and obs_cell makes sure keys won't overlap ('ptcl_1' vs 'obs_1', etc)
        return jax.lax.cond(key_j == key_k, self_case, cross_case, operand=None)

    A22_p2p_dict = {(key_j, key_k): compute_p2p(key_j, key_k) for key_j in ptcl_cell.keys() for key_k in ptcl_cell.keys()}
    A22_p2o_dict = {(key_j, key_k): compute_p2o(key_j, key_k) for key_j in ptcl_cell.keys() for key_k in obs_cell.keys()}
    A22_o2p_dict = {(key_j, key_k): compute_o2p(key_j, key_k) for key_j in obs_cell.keys() for key_k in ptcl_cell.keys()}
    A22_o2o_dict = {(key_j, key_k): compute_o2o(key_j, key_k) for key_j in obs_cell.keys() for key_k in obs_cell.keys()}

    def put_to_grid(cell_j_keys, cell_k_keys, A22_jk_dict):
        grid_u = []
        grid_T = []
        for key_k in cell_k_keys:
            row_u = []
            row_T = []
            for key_j in cell_j_keys:
                if (key_j,key_k) in A22_jk_dict:
                    Akj_block = A22_jk_dict[(key_j, key_k)][0]
                    Tkj_block = A22_jk_dict[(key_j, key_k)][1]
                else:
                    raise Exception(f"ERROR: interaction between particles {key_j} and {key_k} is not stored.")
                row_u.append(Akj_block)
                row_T.append(Tkj_block)
            grid_u.append(row_u)
            grid_T.append(row_T)
        A22_jk = jnp.block(grid_u)
        A22_jk_T = jnp.block(grid_T)
        return A22_jk, A22_jk_T

    [A22_p2p, T_p2p] = put_to_grid(ptcl_cell.keys(), ptcl_cell.keys(), A22_p2p_dict)
    [A22_p2o, _] = put_to_grid(ptcl_cell.keys(), obs_cell.keys(), A22_p2o_dict)
    [A22_o2p, T_o2p] = put_to_grid(obs_cell.keys(), ptcl_cell.keys(), A22_o2p_dict)
    [A22_o2o, _] = put_to_grid(obs_cell.keys(), obs_cell.keys(), A22_o2o_dict)
    A22 = jnp.block([[A22_o2o, A22_p2o],
                     [A22_o2p, A22_p2p]])
    A22 = jnp.eye(A22.shape[0]) / 2. + A22 
    A = jnp.block([[A11, A12],
                   [A21, A22]])

    # Rigid body motion added to b.c.
    Nptx_total = sum(len(ptcl_cell[k]['x']) for k in ptcl_keys)
    Nobsx_total = sum(len(obs_cell[k]['x']) for k in obs_keys)
    total_rows = 2 * Nsx + 2 * Nobsx_total + 2 * Nptx_total
    total_cols = total_rows + 3 * num_ptcl
    bc_sqr = jnp.zeros((2 * Nptx_total, 3 * num_ptcl)) # just the ptcl BC part

    current_row_offset = 0
    F_blocks = []
    T_blocks = [] # store values now for force and torque integrals
    for i, k in enumerate(ptcl_keys):
        pt = ptcl_cell[k]  # Complex points: x + iy
        n_nodes = len(pt['x'])
        # X - Xc
        xc = jnp.mean(pt['x'])
        rel_pos = pt['x'] - xc
        dx = jnp.real(rel_pos)
        dy = jnp.imag(rel_pos)
        # Indices
        ptcl_x_rows = jnp.arange(current_row_offset, current_row_offset + n_nodes)
        ptcl_y_rows = jnp.arange(current_row_offset + n_nodes, current_row_offset + 2 * n_nodes)
        col_ux = 3 * i
        col_uy = 3 * i + 1
        col_omega = 3 * i + 2

        bc_sqr = bc_sqr.at[ptcl_x_rows, col_ux].set(-1.0)
        bc_sqr = bc_sqr.at[ptcl_y_rows, col_uy].set(-1.0)
        bc_sqr = bc_sqr.at[ptcl_x_rows, col_omega].set(-dy)
        bc_sqr = bc_sqr.at[ptcl_y_rows, col_omega].set(dx)

        w_row = pt['ws'][None, :] # shape (1, n_nodes)
        z_row = jnp.zeros_like(w_row)
        f_block = jnp.block([
            [w_row, z_row],
            [z_row, w_row]
        ])
        F_blocks.append(f_block)
        t_block = jnp.concatenate([(-dy * pt['ws'])[None, :], (dx * pt['ws'])[None, :]], axis=1)
        T_blocks.append(t_block)

        current_row_offset += 2 * n_nodes

    # Final assembly
    bc_gamma_mat = jnp.zeros((total_rows, total_cols))
    bc_gamma_mat = bc_gamma_mat.at[:, :total_rows].set(A)
    bc_gamma_mat = bc_gamma_mat.at[2*Nsx+2*Nobsx_total:, total_rows:].set(bc_sqr)

    # Net force zero and net torque zero
    T_p2p = -jnp.eye(T_p2p.shape[0])/2. + T_p2p # active particle self-to-self SL traction jump
    T3 = jnp.hstack([T_o2p, T_p2p]) 
    fMat_ptcl = - T3 # This actually includes both obstacles and swimmer traction (evaluated on swimmer only)
    fMat_wall = - A21_T

    intF_op = jax.scipy.linalg.block_diag(*F_blocks)
    intT_op = jax.scipy.linalg.block_diag(*T_blocks)

    mid_zeros = jnp.zeros((2 * Nptx_total, 3 * num_ptcl))
    big_mat = jnp.concatenate([fMat_wall, fMat_ptcl, mid_zeros], axis=1)
    intF = intF_op @ big_mat
    intT = intT_op @ big_mat

    # --- Full ELS matrix ---
    E = jnp.block([[bc_gamma_mat],
                    [intF],
                   [intT]])


    return E, bc_gamma_mat, intF, intT

@jit
def evalsol_all(t, s, obs_cell, ptcl_cell, mu, edens):
    N_wall = s['x'].size
    N_obs_tot = sum([obs['x'].size for obs in obs_cell.values()])
    N_ptcl_tot = sum([ptcl['x'].size for ptcl in ptcl_cell.values()])
    wall_dens = edens[:2*N_wall]
    obs_dens = edens[2*N_wall:2*(N_wall+N_obs_tot)]
    ptcl_dens = edens[2*(N_wall+N_obs_tot):2*(N_wall+N_obs_tot+N_ptcl_tot)]

    # wall and proxy to target
    [u, p, _] = StoDLP(t, s, mu, wall_dens, self=False, near=True, panel=True)
    # u = u @ wall_dens
    # p = p @ wall_dens

    # TODO: use vmap when number of particles increase dramatically.
    # Obstacle to target
    current_offset = 0
    if obs_cell:
        for obs in obs_cell.values():
            num_dof = obs['x'].size * 2 
            this_obs_dens = obs_dens[current_offset : current_offset + num_dof]
            [uSo, pSo, _] = StoSLP(t, obs, mu, this_obs_dens, self=False, near=True, panel=False)
            [uDo, pDo, _] = StoDLP(t, obs, mu, this_obs_dens, self=False, near=True, panel=False)
            # u = u + (uSo + uDo) @ this_obs_dens
            # p = p + (pSo + pDo) @ this_obs_dens
            u = u + (uSo + uDo)
            p = p + (pSo + pDo)
            current_offset += num_dof
    # Swimmers to target
    current_offset = 0
    if ptcl_cell:
        for pt in ptcl_cell.values():
            num_dof = pt['x'].size * 2 
            this_ptcl_dens = ptcl_dens[current_offset : current_offset + num_dof]
            [uSp, pSp, _] = StoSLP(t, pt, mu, this_ptcl_dens, self=False, near=True, panel=False)
            [uDp, pDp, _] = StoDLP(t, pt, mu, this_ptcl_dens, self=False, near=True, panel=False)
            # u = u + (uSp + uDp) @ this_ptcl_dens
            # p = p + (pSp + pDp) @ this_ptcl_dens
            u = u + (uSp + uDp)
            p = p + (pSp + pDp)
            current_offset += num_dof

    if len(u.shape) > 1 and u.shape[1] == 1:
        u = u.ravel()
        p = p.ravel()
    
    return u,p
