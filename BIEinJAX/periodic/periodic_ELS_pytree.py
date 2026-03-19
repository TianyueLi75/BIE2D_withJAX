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
@jit
def ELSmatrix_wrapper(s, ptcl_cell, P, L, R, peri_len, mu):
    if not ptcl_cell:
        E, A, B, C, Q = ELSmatrix(s, P, L, R, peri_len, mu)
    elif ptcl_cell:
        E, A, B, C, Q = ELSmatrix_ptcl(s, ptcl_cell, P, L, R, peri_len, mu)
    return E, A, B, C, Q

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
    NOTE: DL traction is still unchecked, so may be off when particles come close in rbm. 
            Should enforce min-sep for now.
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



# ----------------------------------------------------------------------
# --- Evaluation -------------------------------
# ----------------------------------------------------------------------
@jit
def evalsol_wrapper(t, s, ptcl_cell, P, peri_len, mu, edens):
    if not ptcl_cell:
        u, p = evalsol(t, s, P, peri_len, mu, edens)
    elif ptcl_cell:
        u, p = evalsol_ptcl(t, s, ptcl_cell, P, peri_len, mu, edens)
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


