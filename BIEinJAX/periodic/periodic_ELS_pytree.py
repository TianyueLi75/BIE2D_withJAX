"""
JAX-based implementation of the Extended Linear System (ELS) matrix 
for singly-periodic solve and evaluation for 2D Stokes problems
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

# ----------------------------------------------------------------------
# --- Summing over near sources -------------------------------
# ----------------------------------------------------------------------
@partial(jit, static_argnums=(0,4))
def srcsum(kernel, t, s, mu, self=False):
    """ kernel is a Python-callable kernel function and must be static for the same reason """
    trlist = jnp.atleast_1d(s['trlist'])
    n_tr = trlist.size
    if n_tr % 2 != 1:
        raise Exception("ERROR: trlist should contain odd number of values, so that two sides are symmetric.")
    mid_tr = n_tr // 2 # index of translation that returns the original copy, where self-interact matrix will be used if self=True

    [u,p,T] = kernel(t, {**s, 'x': s['x']+trlist[0]}, mu, jnp.array([]),False)

    for i in range(1,n_tr):
        if i == mid_tr:
            [tempu, tempp, tempT] = kernel(t, s, mu, jnp.array([]), self)
        else:
            [tempu,tempp,tempT] = kernel(t, {**s, 'x': s['x']+trlist[i]}, mu, jnp.array([]), False)
        
        u = u + tempu
        p = p + tempp
        T = T + tempT
 
    return u, p, T

@partial(jit, static_argnums=(0,4,5))
def srcsum_eval(kernel, t, s, mu, self=False, panel=False):
    """ kernel is a Python-callable kernel function and must be static for the same reason """
    trlist = jnp.atleast_1d(s['trlist'])
    n_tr = trlist.size
    if n_tr % 2 != 1:
        raise Exception("ERROR: trlist should contain odd number of values, so that two sides are symmetric.")
    mid_tr = n_tr // 2 # index of translation that returns the original copy, where self-interact matrix will be used if self=True

    [u,p,T] = kernel(t, {**s, 'x': s['x']+trlist[0]}, mu, jnp.array([]),False)

    for i in range(1,n_tr):
        if i == mid_tr:
            if not self:
                # center copy but not self-to-self, use near eval
                [tempu, tempp, tempT] = kernel(t, s, mu, jnp.array([]), self, True, panel)
                # [tempu, tempp, tempT] = kernel(t, s, mu, jnp.array([]), self, False, panel)
            else:
                # center copy self to self
                [tempu, tempp, tempT] = kernel(t, s, mu, jnp.array([]), self)
        else:
            [tempu,tempp,tempT] = kernel(t, {**s, 'x': s['x']+trlist[i]}, mu, jnp.array([]), False)
        
        u = u + tempu
        p = p + tempp
        T = T + tempT
 
    return u, p, T

# ----------------------------------------------------------------------
# --- ELS matrices -------------------------------
# ----------------------------------------------------------------------
# Updated notation: ptcl_cell contains only active particle, so obs_cell are the passive particles.
@jit
def ELSmatrix_wrapper(s, obs_cell, ptcl_cell, P, L, R, peri_len, mu):
    if not ptcl_cell: # passive or empty channel
        if not obs_cell:
            E, A, B, C, Q = ELSmatrix(s, P, L, R, peri_len, mu)
        else:
            E, A, B, C, Q = ELSmatrix_ptcl(s, obs_cell, P, L, R, peri_len, mu)
        intF = jnp.array([])
        intT = jnp.array([])
    elif ptcl_cell:
        if not obs_cell: # only active ptcl and channel
            E, A, B, intF, intT, C, Q = ELSmatrix_rbm(s, ptcl_cell, P, L, R, peri_len, mu)
        else: 
            E, A, B, intF, intT, C, Q = ELSmatrix_rbm_obs(s, obs_cell, ptcl_cell, P, L, R, peri_len, mu)
    return E, A, B, intF, intT, C, Q

@jit
def ELSmatrix(s, P, L, R, peri_len, mu):
    Nsx = s['x'].size
    s['trlist'] = jnp.array([-peri_len,0, peri_len])
    [Asrcsum,_,_] = srcsum(StoDLP, s, s, mu, True)
    A = -jnp.eye(2*Nsx) / 2 + Asrcsum

    [B,_,_] = StoSLP(s, P, mu, jnp.array([]), False)

    s['trlist'] = jnp.array([peri_len])
    [CLD, _, TLD] = srcsum(StoDLP, L, s, mu, False)
    s['trlist'] = jnp.array([-peri_len])
    [CRD, _, TRD] = srcsum(StoDLP, R, s, mu) 
    C = jnp.vstack([CRD - CLD, TRD - TLD])

    [QL, _, QLt] = StoSLP(L, P, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(R, P, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # --- Full ELS matrix ---
    E = jnp.block([[A, B],
                   [C, Q]])

    return E, A, B, C, Q

@jit
def ELSmatrix_ptcl(s, ptcl_cell, P, L, R, peri_len, mu):
    """
    A DL formulation on wall (like before) and SL+DL on particle
    Includes close-eval on all non-self interactions.
    """
    # trlist = peri_len*jnp.array([-1,0,1])
    s['trlist'] = jnp.array([-peri_len, 0, peri_len])
    for pt in ptcl_cell.values():
        pt['trlist'] = jnp.array([-peri_len, 0, peri_len])
    Nsx = s['x'].size
    # Wall self-to-self
    [Asrcsum,_,_] = srcsum(StoDLP, s, s, mu, True)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    # Particle to wall from all ptcls in ptcl_cell, summed
    ptcl_keys = ptcl_cell.keys()
    A12_list = [srcsum_eval(StoDLP, s, ptcl_cell[k], mu, False)[0] + srcsum_eval(StoSLP, s, ptcl_cell[k], mu, False)[0] for k in ptcl_keys]
    A12 = jnp.hstack(list(A12_list))
    # Wall to particle
    A21_list = [srcsum_eval(StoDLP, ptcl_cell[k], s, mu, False, True)[0] for k in ptcl_keys]
    A21 = jnp.vstack(A21_list)

    # FROM ptcl_j (with near) to ptcl_k
    def compute_pair_interaction(key_j, key_k):
        def zero_translation_contribution():
            p_j = ptcl_cell[key_j]
            p_k = ptcl_cell[key_k]
            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res_sl, _, _ = StoSLP(p_j, p_j, mu, jnp.array([]), True)
                res_dl, _, _ = StoDLP(p_j, p_j, mu, jnp.array([]), True)
                return res_sl + res_dl

            def cross_case(_):
                # Off-diagonal block: source particle j, target particle k
                # res_sl, _, _ = StoSLP(p_k, p_j, mu, jnp.array([]), False)
                # res_dl, _, _ = StoDLP(p_k, p_j, mu, jnp.array([]), False)
                res_sl, _, _ = stoSLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e') # Use close-eval to handle close-together particles.
                res_dl, _, _ = stoDLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e')
                return res_sl + res_dl
            
            return jax.lax.cond(key_j == key_k, self_case, cross_case, operand=None)

        def periodic_contribution():
            p_j = ptcl_cell[key_j]
            p_k = ptcl_cell[key_k]
            def single_tr_step(tr_val):
                slp, _, _ = StoSLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                dlp, _, _ = StoDLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                return slp + dlp

            # Sum over non-zero translations only
            far_trlist = jnp.array([s['trlist'][0], s['trlist'][2]])
            results = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results, axis=0)

        return zero_translation_contribution() + periodic_contribution()

    A22_dict = {(key_j, key_k): compute_pair_interaction(key_j, key_k) for key_j in ptcl_cell.keys() for key_k in ptcl_cell.keys()}
    grid = []
    for key_k in ptcl_keys:
        row = []
        for key_j in ptcl_keys:
            if (key_j,key_k) in A22_dict:
                Akj_block = A22_dict[(key_j, key_k)]
            else:
                raise Exception(f"ERROR: interaction between particles {key_j} and {key_k} is not stored.")
            row.append(Akj_block)
        grid.append(row)

    A22 = jnp.block(grid)
    A22 = jnp.eye(A22.shape[0]) / 2. + A22 
    A = jnp.block([[A11, A12],
                   [A21, A22]])

    # From proxy to wall and pt
    [B1,_,_] = StoSLP(s, P, mu, jnp.array([]))
    B2_list = [StoSLP(ptcl_cell[k], P, mu, jnp.array([]), False)[0] for k in ptcl_keys]
    B2 = jnp.vstack(B2_list)
    B = jnp.vstack([B1, B2])

    s['trlist'] = jnp.array([peri_len])
    [CLD, _, TLD] = srcsum(StoDLP, L, s, mu, False)
    s['trlist'] = jnp.array([-peri_len])
    [CRD, _, TRD] = srcsum(StoDLP, R, s, mu, False) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])

    C2_u = []
    C2_T = []
    for k in ptcl_keys:
        ptcl_cell[k]['trlist'] = jnp.array([peri_len])
        C2L_sl_u,_,C2L_sl_T = srcsum(StoSLP, L, ptcl_cell[k], mu, False)
        C2L_dl_u,_,C2L_dl_T = srcsum(StoDLP, L, ptcl_cell[k], mu, False)
        ptcl_cell[k]['trlist'] = jnp.array([-peri_len])
        C2R_dl_u,_,C2R_dl_T = srcsum(StoDLP, R, ptcl_cell[k], mu, False)
        C2R_sl_u,_,C2R_sl_T = srcsum(StoSLP, R, ptcl_cell[k], mu, False)
        u_combined = C2R_dl_u + C2R_sl_u - C2L_dl_u - C2L_sl_u
        T_combined = C2R_dl_T + C2R_sl_T - C2L_dl_T - C2L_sl_T
        C2_u.append(u_combined)
        C2_T.append(T_combined)
    C2 = jnp.vstack([
        jnp.hstack(C2_u),
        jnp.hstack(C2_T)
    ])

    # -------- Suggested vectorized way to avoid the loop above. skip for now.
    # ptcl_stack = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *[ptcl_cell[k] for k in ptcl_keys])
    # def compute_ptcl_contribution(ptcl):
    #     # DL and SL for Right and Left
    #     r_dl = srcsum(StoDLP, R, ptcl, mu, False)
    #     r_sl = srcsum(StoSLP, R, ptcl, mu, False)
    #     l_dl = srcsum(StoDLP, L, ptcl, mu, False)
    #     l_sl = srcsum(StoSLP, L, ptcl, mu, False)
    #     u = r_dl[0] - l_dl[0] + r_sl[0] - l_sl[0]
    #     t = r_dl[2] - l_dl[2] + r_sl[2] - l_sl[2]
    #     return u, t
    # v_compute = jax.vmap(compute_ptcl_contribution, in_axes=0)
    # u_all, t_all = v_compute(ptcl_stack)
    # C2 = jnp.vstack([
    #     jnp.hstack(u_all), 
    #     jnp.hstack(t_all)
    # ])
    # -------------------------------------------------------------

    C = jnp.hstack([C1, C2])

    [QL, _, QLt] = StoSLP(L, P, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(R, P, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # --- Full ELS matrix ---
    E = jnp.block([[A, B],
                   [C, Q]])

    return E, A, B, C, Q

@jit
def ELSmatrix_rbm(s, ptcl_cell, P, L, R, peri_len, mu):
    """
    A DL formulation on wall (like before) and SL+DL on particle -- Rigid body motion
    Close-eval included in all non-self interactions.
    """
    # trlist = peri_len*jnp.array([-1,0,1])
    s['trlist'] = jnp.array([-peri_len, 0, peri_len])
    for pt in ptcl_cell.values():
        pt['trlist'] = jnp.array([-peri_len, 0, peri_len])
    Nsx = s['x'].size
    # Wall self-to-self
    [Asrcsum,_,_] = srcsum(StoDLP, s, s, mu, True)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    # Particle to wall from all ptcls in ptcl_cell, summed
    ptcl_keys = ptcl_cell.keys()
    num_ptcl = len(ptcl_keys)
    A12_list = [srcsum_eval(StoDLP, s, ptcl_cell[k], mu, False)[0] + srcsum_eval(StoSLP, s, ptcl_cell[k], mu, False)[0] for k in ptcl_keys]
    A12 = jnp.hstack(list(A12_list))
    # Wall to particle
    A21_tuples = [srcsum_eval(StoDLP, ptcl_cell[k], s, mu, False, True) for k in ptcl_keys]
    A21_u_list, _, A21_T_list = zip(*A21_tuples)
    A21 = jnp.vstack(A21_u_list)
    A21_T = jnp.vstack(A21_T_list)

    # FROM ptcl_j (with near) to ptcl_k
    def compute_pair_interaction(key_j, key_k):
        def zero_translation_contribution():
            p_j = ptcl_cell[key_j]
            p_k = ptcl_cell[key_k]
            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res_sl, _, res_slT = StoSLP(p_j, p_j, mu, jnp.array([]), True)
                res_dl, _, _ = StoDLP(p_j, p_j, mu, jnp.array([]), True) # self-to-self DL traction is always 0.
                return res_sl + res_dl, res_slT

            def cross_case(_):
                # Off-diagonal block: source particle j, target particle k
                # res_sl, _, res_sl_T = StoSLP(p_k, p_j, mu, jnp.array([]), False)
                # res_dl, _, res_dl_T = StoDLP(p_k, p_j, mu, jnp.array([]), False)
                res_sl, _, res_sl_T = stoSLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e') # Use close-eval to handle close-together particles.
                res_dl, _, res_dl_T = stoDLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e')
                return res_sl + res_dl, res_sl_T + res_dl_T
            
            return jax.lax.cond(key_j == key_k, self_case, cross_case, operand=None)

        def periodic_contribution():
            p_j = ptcl_cell[key_j]
            p_k = ptcl_cell[key_k]
            def single_tr_step(tr_val):
                slp, _, slpT = StoSLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                dlp, _, dlpT = StoDLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                return slp + dlp, slpT + dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([s['trlist'][0], s['trlist'][2]])
            results_u, results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p

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

    # From proxy to wall and pt
    [B1,_,_] = StoSLP(s, P, mu, jnp.array([]))
    B2_tuples = [StoSLP(ptcl_cell[k], P, mu, jnp.array([]), False) for k in ptcl_keys]
    B2_u_list, _, B2_T_list = zip(*B2_tuples)
    B2 = jnp.vstack(B2_u_list)
    B2_T = jnp.vstack(B2_T_list)
    B = jnp.vstack([B1, B2])

    s['trlist'] = jnp.array([peri_len])
    [CLD, _, TLD] = srcsum(StoDLP, L, s, mu, False)
    s['trlist'] = jnp.array([-peri_len])
    [CRD, _, TRD] = srcsum(StoDLP, R, s, mu, False) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])

    C2_u = []
    C2_T = []
    for k in ptcl_keys:
        ptcl_cell[k]['trlist'] = jnp.array([peri_len])
        C2L_sl_u,_,C2L_sl_T = srcsum(StoSLP, L, ptcl_cell[k], mu, False)
        C2L_dl_u,_,C2L_dl_T = srcsum(StoDLP, L, ptcl_cell[k], mu, False)
        ptcl_cell[k]['trlist'] = jnp.array([-peri_len])
        C2R_dl_u,_,C2R_dl_T = srcsum(StoDLP, R, ptcl_cell[k], mu, False)
        C2R_sl_u,_,C2R_sl_T = srcsum(StoSLP, R, ptcl_cell[k], mu, False)
        u_combined = C2R_dl_u + C2R_sl_u - C2L_dl_u - C2L_sl_u
        T_combined = C2R_dl_T + C2R_sl_T - C2L_dl_T - C2L_sl_T
        C2_u.append(u_combined)
        C2_T.append(T_combined)
    C2 = jnp.vstack([
        jnp.hstack(C2_u),
        jnp.hstack(C2_T)
    ])

    # -------- Suggested vectorized way to avoid the loop above. skip for now.
    # ptcl_stack = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *[ptcl_cell[k] for k in ptcl_keys])
    # def compute_ptcl_contribution(ptcl):
    #     # DL and SL for Right and Left
    #     r_dl = srcsum(StoDLP, R, ptcl, mu, False)
    #     r_sl = srcsum(StoSLP, R, ptcl, mu, False)
    #     l_dl = srcsum(StoDLP, L, ptcl, mu, False)
    #     l_sl = srcsum(StoSLP, L, ptcl, mu, False)
    #     u = r_dl[0] - l_dl[0] + r_sl[0] - l_sl[0]
    #     t = r_dl[2] - l_dl[2] + r_sl[2] - l_sl[2]
    #     return u, t
    # v_compute = jax.vmap(compute_ptcl_contribution, in_axes=0)
    # u_all, t_all = v_compute(ptcl_stack)
    # C2 = jnp.vstack([
    #     jnp.hstack(u_all), 
    #     jnp.hstack(t_all)
    # ])
    # -------------------------------------------------------------

    C = jnp.hstack([C1, C2])

    [QL, _, QLt] = StoSLP(L, P, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(R, P, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

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
    fMat_prx = - B2_T

    intF_op = jax.scipy.linalg.block_diag(*F_blocks)
    intT_op = jax.scipy.linalg.block_diag(*T_blocks)

    mid_zeros = jnp.zeros((2 * Nptx_total, 3 * num_ptcl))
    big_mat = jnp.concatenate([fMat_wall, fMat_ptcl, mid_zeros, fMat_prx], axis=1)
    intF = intF_op @ big_mat
    intT = intT_op @ big_mat

    # --- Full ELS matrix ---
    E = jnp.block([[bc_gamma_mat, B],
                    [intF],
                   [intT],
                   [C, jnp.zeros((C.shape[0],3*num_ptcl)), Q]])


    return E, bc_gamma_mat, B, intF, intT, C, Q

@jit
def ELSmatrix_rbm_obs(s, obs_cell, ptcl_cell, P, L, R, peri_len, mu):
    """
    A DL formulation on wall (like before)
     SL+DL formulation on obstacles (passive and often fixed particle) 
     and SL+DL on active particle with Rigid body motion
    Close-eval included in all non-self interactions.
    """
    s['trlist'] = jnp.array([-peri_len, 0, peri_len])
    for obs in obs_cell.values():
        obs['trlist'] = jnp.array([-peri_len, 0, peri_len])
    for pt in ptcl_cell.values():
        pt['trlist'] = jnp.array([-peri_len, 0, peri_len])
    Nsx = s['x'].size
    # Wall self-to-self
    [Asrcsum,_,_] = srcsum(StoDLP, s, s, mu, True)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    # Obstacle to wall from all obs in obs_cell, summed
    obs_keys = obs_cell.keys()
    num_obs = len(obs_keys)
    A12_obs_list = [srcsum_eval(StoDLP, s, obs_cell[k], mu, False)[0] + srcsum_eval(StoSLP, s, obs_cell[k], mu, False)[0] for k in obs_keys]
    # Particle to wall from all ptcls in ptcl_cell, summed
    ptcl_keys = ptcl_cell.keys()
    num_ptcl = len(ptcl_keys)
    A12_ptcl_list = [srcsum_eval(StoDLP, s, ptcl_cell[k], mu, False)[0] + srcsum_eval(StoSLP, s, ptcl_cell[k], mu, False)[0] for k in ptcl_keys]
    A12_obs_list.extend(A12_ptcl_list) 
    A12 = jnp.hstack(A12_obs_list) 
    # Wall to obstacle
    A21_obs_tuples = [srcsum_eval(StoDLP, obs_cell[k], s, mu, False, True) for k in obs_keys]
    A21_obs_u_list, _, _ = zip(*A21_obs_tuples) # don't need traction on obstacles
    A21_obs_u_list = list(A21_obs_u_list) # zip outputs tuples so switch to list
    # Wall to particle
    A21_ptcl_tuples = [srcsum_eval(StoDLP, ptcl_cell[k], s, mu, False, True) for k in ptcl_keys]
    A21_ptcl_u_list, _, A21_ptcl_T_list = zip(*A21_ptcl_tuples)
    A21_ptcl_u_list = list(A21_ptcl_u_list)
    A21_obs_u_list.extend(A21_ptcl_u_list) 
    A21 = jnp.vstack(A21_obs_u_list)
    A21_T = jnp.vstack(A21_ptcl_T_list)

    # FROM ptcl_j (with near) to ptcl_k 
    def compute_p2p(key_j, key_k):
        def zero_translation_contribution():
            p_j = ptcl_cell[key_j]
            p_k = ptcl_cell[key_k]
            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res_sl, _, res_slT = StoSLP(p_j, p_j, mu, jnp.array([]), True)
                res_dl, _, _ = StoDLP(p_j, p_j, mu, jnp.array([]), True) # self-to-self DL traction is always 0.
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

        def periodic_contribution():
            p_j = ptcl_cell[key_j]
            p_k = ptcl_cell[key_k]
            def single_tr_step(tr_val):
                slp, _, slpT = StoSLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                dlp, _, dlpT = StoDLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                return slp + dlp, slpT + dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([s['trlist'][0], s['trlist'][2]])
            results_u, results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p
    
    # FROM ptcl_j (with near) to obs_k (no self)
    def compute_p2o(key_j, key_k):
        def zero_translation_contribution():
            p_j = ptcl_cell[key_j]
            p_k = obs_cell[key_k]
            # Off-diagonal block: source particle j, target particle k
            # res_sl, _, res_sl_T = StoSLP(p_k, p_j, mu, jnp.array([]), False)
            # res_dl, _, res_dl_T = StoDLP(p_k, p_j, mu, jnp.array([]), False)
            res_sl, _, res_sl_T = stoSLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e') # Use close-eval to handle close-together particles.
            res_dl, _, res_dl_T = stoDLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e')
            return res_sl + res_dl, res_sl_T + res_dl_T

        def periodic_contribution():
            p_j = ptcl_cell[key_j]
            p_k = obs_cell[key_k]
            def single_tr_step(tr_val):
                slp, _, slpT = StoSLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                dlp, _, dlpT = StoDLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                return slp + dlp, slpT + dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([s['trlist'][0], s['trlist'][2]])
            results_u, results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p
    
    # FROM obs_j (with near) to ptcl_k  (no self)
    def compute_o2p(key_j, key_k):
        def zero_translation_contribution():
            p_j = obs_cell[key_j]
            p_k = ptcl_cell[key_k]
            # Off-diagonal block: source particle j, target particle k
            # res_sl, _, res_sl_T = StoSLP(p_k, p_j, mu, jnp.array([]), False)
            # res_dl, _, res_dl_T = StoDLP(p_k, p_j, mu, jnp.array([]), False)
            res_sl, _, res_sl_T = stoSLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e') # Use close-eval to handle close-together particles.
            res_dl, _, res_dl_T = stoDLP_closeglobal(p_k, p_j, mu, jnp.eye(2*p_j['x'].size), 'e')
            return res_sl + res_dl, res_sl_T + res_dl_T

        def periodic_contribution():
            p_j = obs_cell[key_j]
            p_k = ptcl_cell[key_k]
            def single_tr_step(tr_val):
                slp, _, slpT = StoSLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                dlp, _, dlpT = StoDLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                return slp + dlp, slpT + dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([s['trlist'][0], s['trlist'][2]])
            results_u, results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p
    
    # FROM ptcl_j (with near) to ptcl_k 
    def compute_o2o(key_j, key_k):
        def zero_translation_contribution():
            p_j = obs_cell[key_j]
            p_k = obs_cell[key_k]
            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res_sl, _, res_slT = StoSLP(p_j, p_j, mu, jnp.array([]), True)
                res_dl, _, _ = StoDLP(p_j, p_j, mu, jnp.array([]), True) # self-to-self DL traction is always 0.
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

        def periodic_contribution():
            p_j = obs_cell[key_j]
            p_k = obs_cell[key_k]
            def single_tr_step(tr_val):
                slp, _, slpT = StoSLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                dlp, _, dlpT = StoDLP(p_k, {**p_j, 'x': p_j['x']+tr_val}, mu, jnp.array([]), False)
                return slp + dlp, slpT + dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([s['trlist'][0], s['trlist'][2]])
            results_u, results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p

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

    # From proxy to wall and obs and pt
    [B1,_,_] = StoSLP(s, P, mu, jnp.array([]))
    B2_obs_tuples = [StoSLP(obs_cell[k], P, mu, jnp.array([]), False) for k in obs_keys]
    B2_obs_u_list, _, _ = zip(*B2_obs_tuples) # don't need traction on obstacles
    B2_obs_u_list = list(B2_obs_u_list)
    B2_ptcl_tuples = [StoSLP(ptcl_cell[k], P, mu, jnp.array([]), False) for k in ptcl_keys]
    B2_ptcl_u_list, _, B2_ptcl_T_list = zip(*B2_ptcl_tuples)
    B2_ptcl_u_list = list(B2_ptcl_u_list)
    B2_obs_u_list.extend(B2_ptcl_u_list)
    B2 = jnp.vstack(B2_obs_u_list)
    B2_T = jnp.vstack(B2_ptcl_T_list)
    B = jnp.vstack([B1, B2])

    s['trlist'] = jnp.array([peri_len])
    [CLD, _, TLD] = srcsum(StoDLP, L, s, mu, False)
    s['trlist'] = jnp.array([-peri_len])
    [CRD, _, TRD] = srcsum(StoDLP, R, s, mu, False) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])

    def get_CuT(obj_cell):
        C2_u = []
        C2_T = []
        for k in obj_cell.keys():
            obj_cell[k]['trlist'] = jnp.array([peri_len])
            C2L_sl_u,_,C2L_sl_T = srcsum(StoSLP, L, obj_cell[k], mu, False)
            C2L_dl_u,_,C2L_dl_T = srcsum(StoDLP, L, obj_cell[k], mu, False)
            obj_cell[k]['trlist'] = jnp.array([-peri_len])
            C2R_dl_u,_,C2R_dl_T = srcsum(StoDLP, R, obj_cell[k], mu, False)
            C2R_sl_u,_,C2R_sl_T = srcsum(StoSLP, R, obj_cell[k], mu, False)
            u_combined = C2R_dl_u + C2R_sl_u - C2L_dl_u - C2L_sl_u
            T_combined = C2R_dl_T + C2R_sl_T - C2L_dl_T - C2L_sl_T
            C2_u.append(u_combined)
            C2_T.append(T_combined)
        return C2_u, C2_T
    
    [C2_ptcl_u, C2_ptcl_T] = get_CuT(ptcl_cell)
    [C2_obs_u, C2_obs_T] = get_CuT(obs_cell)

    C2_obs = jnp.vstack([
        jnp.hstack(C2_obs_u),
        jnp.hstack(C2_obs_T),
    ])

    C2_ptcl = jnp.vstack([
        jnp.hstack(C2_ptcl_u),
        jnp.hstack(C2_ptcl_T),
    ])

    # -------- Suggested vectorized way to avoid the loop above. skip for now.
    # ptcl_stack = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *[ptcl_cell[k] for k in ptcl_keys])
    # def compute_ptcl_contribution(ptcl):
    #     # DL and SL for Right and Left
    #     r_dl = srcsum(StoDLP, R, ptcl, mu, False)
    #     r_sl = srcsum(StoSLP, R, ptcl, mu, False)
    #     l_dl = srcsum(StoDLP, L, ptcl, mu, False)
    #     l_sl = srcsum(StoSLP, L, ptcl, mu, False)
    #     u = r_dl[0] - l_dl[0] + r_sl[0] - l_sl[0]
    #     t = r_dl[2] - l_dl[2] + r_sl[2] - l_sl[2]
    #     return u, t
    # v_compute = jax.vmap(compute_ptcl_contribution, in_axes=0)
    # u_all, t_all = v_compute(ptcl_stack)
    # C2 = jnp.vstack([
    #     jnp.hstack(u_all), 
    #     jnp.hstack(t_all)
    # ])
    # -------------------------------------------------------------

    C = jnp.hstack([C1, C2_obs, C2_ptcl])

    [QL, _, QLt] = StoSLP(L, P, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(R, P, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

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
    fMat_prx = - B2_T

    intF_op = jax.scipy.linalg.block_diag(*F_blocks)
    intT_op = jax.scipy.linalg.block_diag(*T_blocks)

    mid_zeros = jnp.zeros((2 * Nptx_total, 3 * num_ptcl))
    big_mat = jnp.concatenate([fMat_wall, fMat_ptcl, mid_zeros, fMat_prx], axis=1)
    intF = intF_op @ big_mat
    intT = intT_op @ big_mat

    # --- Full ELS matrix ---
    E = jnp.block([[bc_gamma_mat, B],
                    [intF],
                   [intT],
                   [C, jnp.zeros((C.shape[0],3*num_ptcl)), Q]])


    return E, bc_gamma_mat, B, intF, intT, C, Q



# ----------------------------------------------------------------------
# --- Evaluation -------------------------------
# ----------------------------------------------------------------------
@jit
def evalsol_wrapper(t, s, obs_cell, ptcl_cell, P, peri_len, mu, edens):
    if not ptcl_cell:
        if not obs_cell:
            u, p = evalsol(t, s, P, peri_len, mu, edens)
        else:
            u, p = evalsol_ptcl_obs(t, s, obs_cell, ptcl_cell, P, peri_len, mu, edens)
    elif ptcl_cell:
        if not obs_cell:
            u, p = evalsol_ptcl(t, s, ptcl_cell, P, peri_len, mu, edens)
        else:
            u, p = evalsol_ptcl_obs(t, s, obs_cell, ptcl_cell, P, peri_len, mu, edens)
    return u, p

@jit
def evalsol(t, s, P, peri_len, mu, edens):
    s['trlist'] = jnp.array([-peri_len, 0, peri_len])
    N_wall = s['x'].size
    wall_dens = edens[:2*N_wall]
    prx_dens = edens[2*N_wall:]

    [u, p, _] = StoSLP(t, P, mu, prx_dens)
    [uD, pD, _] = srcsum_eval(StoDLP, t, s, mu, False, True)
    uD = uD @ wall_dens
    pD = pD @ wall_dens

    u = u+uD
    p = p+pD
    return u,p

@jit
def evalsol_ptcl(t, s, ptcl_cell, P, peri_len, mu, edens):
    N_wall = s['x'].size
    s['trlist'] = jnp.array([-peri_len, 0, peri_len])
    N_ptcl_tot = 0
    for pt in ptcl_cell.values():
        pt['trlist'] = jnp.array([-peri_len, 0, peri_len])
        N_ptcl_tot = N_ptcl_tot + pt['x'].size
    wall_dens = edens[:2*N_wall]
    ptcl_dens = edens[2*N_wall:2*(N_wall+N_ptcl_tot)]
    prx_dens = edens[2*(N_wall+N_ptcl_tot):]

    [u, p, _] = StoSLP(t, P, mu, prx_dens)
    [uD, pD, _] = srcsum_eval(StoDLP, t, s, mu, False, True)
    u = u + uD @ wall_dens
    p = p + pD @ wall_dens

    # TODO: use vmap when number of particles increase dramatically.
    current_offset = 0
    for pt in ptcl_cell.values():
        num_dof = pt['x'].size * 2 
        this_ptcl_dens = ptcl_dens[current_offset : current_offset + num_dof]
        [uSp, pSp, _] = srcsum_eval(StoSLP, t, pt, mu, False, False)
        [uDp, pDp, _] = srcsum_eval(StoDLP, t, pt, mu, False, False)
        u = u + (uSp + uDp) @ this_ptcl_dens
        p = p + (pSp + pDp) @ this_ptcl_dens
        current_offset += num_dof
    
    return u,p

@jit
def evalsol_ptcl_obs(t, s, obs_cell, ptcl_cell, P, peri_len, mu, edens):
    N_wall = s['x'].size
    s['trlist'] = jnp.array([-peri_len, 0, peri_len])
    N_obs_tot = 0
    for obs in obs_cell.values():
        obs['trlist'] = jnp.array([-peri_len, 0, peri_len])
        N_obs_tot = N_obs_tot + obs['x'].size
    N_ptcl_tot = 0
    for pt in ptcl_cell.values():
        pt['trlist'] = jnp.array([-peri_len, 0, peri_len])
        N_ptcl_tot = N_ptcl_tot + pt['x'].size
    wall_dens = edens[:2*N_wall]
    obs_dens = edens[2*N_wall:2*(N_wall+N_obs_tot)]
    ptcl_dens = edens[2*(N_wall+N_obs_tot):2*(N_wall+N_obs_tot+N_ptcl_tot)]
    prx_dens = edens[2*(N_wall+N_obs_tot+N_ptcl_tot):]

    [u, p, _] = StoSLP(t, P, mu, prx_dens)
    [uD, pD, _] = srcsum_eval(StoDLP, t, s, mu, False, True)
    u = u + uD @ wall_dens
    p = p + pD @ wall_dens

    # TODO: use vmap when number of particles increase dramatically.
    current_offset = 0
    for obs in obs_cell.values():
        num_dof = obs['x'].size * 2 
        this_obs_dens = obs_dens[current_offset : current_offset + num_dof]
        [uSo, pSo, _] = srcsum_eval(StoSLP, t, obs, mu, False, False)
        [uDo, pDo, _] = srcsum_eval(StoDLP, t, obs, mu, False, False)
        u = u + (uSo + uDo) @ this_obs_dens
        p = p + (pSo + pDo) @ this_obs_dens
        current_offset += num_dof

    current_offset = 0
    for pt in ptcl_cell.values():
        num_dof = pt['x'].size * 2 
        this_ptcl_dens = ptcl_dens[current_offset : current_offset + num_dof]
        [uSp, pSp, _] = srcsum_eval(StoSLP, t, pt, mu, False, False)
        [uDp, pDp, _] = srcsum_eval(StoDLP, t, pt, mu, False, False)
        u = u + (uSp + uDp) @ this_ptcl_dens
        p = p + (pSp + pDp) @ this_ptcl_dens
        current_offset += num_dof
    
    return u,p

@jit
def evalsol_all(t, s, obs_cell, ptcl_cell, P, peri_len, mu, edens):
    N_wall = s['x'].size
    s['trlist'] = jnp.array([-peri_len, 0, peri_len])
    N_obs_tot = 0
    if obs_cell:
        for obs in obs_cell.values():
            obs['trlist'] = jnp.array([-peri_len, 0, peri_len])
            N_obs_tot = N_obs_tot + obs['x'].size
    N_ptcl_tot = 0
    if ptcl_cell:
        for pt in ptcl_cell.values():
            pt['trlist'] = jnp.array([-peri_len, 0, peri_len])
            N_ptcl_tot = N_ptcl_tot + pt['x'].size
    wall_dens = edens[:2*N_wall]
    obs_dens = edens[2*N_wall:2*(N_wall+N_obs_tot)]
    ptcl_dens = edens[2*(N_wall+N_obs_tot):2*(N_wall+N_obs_tot+N_ptcl_tot)]
    prx_dens = edens[2*(N_wall+N_obs_tot+N_ptcl_tot):]

    # wall and proxy to target
    [u, p, _] = StoSLP(t, P, mu, prx_dens)
    [uD, pD, _] = srcsum_eval(StoDLP, t, s, mu, False, True)
    u = u + uD @ wall_dens
    p = p + pD @ wall_dens

    # TODO: use vmap when number of particles increase dramatically.
    # Obstacle to target
    current_offset = 0
    if obs_cell:
        for obs in obs_cell.values():
            num_dof = obs['x'].size * 2 
            this_obs_dens = obs_dens[current_offset : current_offset + num_dof]
            [uSo, pSo, _] = srcsum_eval(StoSLP, t, obs, mu, False, False)
            [uDo, pDo, _] = srcsum_eval(StoDLP, t, obs, mu, False, False)
            u = u + (uSo + uDo) @ this_obs_dens
            p = p + (pSo + pDo) @ this_obs_dens
            current_offset += num_dof
    # Swimmers to target
    current_offset = 0
    if ptcl_cell:
        for pt in ptcl_cell.values():
            num_dof = pt['x'].size * 2 
            this_ptcl_dens = ptcl_dens[current_offset : current_offset + num_dof]
            [uSp, pSp, _] = srcsum_eval(StoSLP, t, pt, mu, False, False)
            [uDp, pDp, _] = srcsum_eval(StoDLP, t, pt, mu, False, False)
            u = u + (uSp + uDp) @ this_ptcl_dens
            p = p + (pSp + pDp) @ this_ptcl_dens
            current_offset += num_dof
    
    return u,p
