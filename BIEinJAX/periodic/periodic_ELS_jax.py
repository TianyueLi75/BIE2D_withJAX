# Making the ELS matrix.

import jax

import jax.numpy as jnp
from jax import jit
import sys, os
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the folder containing the module you want to import
utils_path = os.path.join(current_dir, '..')
# Add the folder to the system path
sys.path.append(utils_path)
from Sto_kernel_utils_jax import *

@jit
def ELSmatrix(sx, snx, sxp, scur, sw, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu):
    # trlist = peri_len*jnp.array([-1,0,1])
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    [Asrcsum,_,_] = srcsum_self(StoDLP_self, StoDLP, trlist, sx, snx, sxp, scur, sw, mu)
    A = -jnp.eye(2*Nsx) / 2 + Asrcsum

    [B,_,_] = StoSLP(sx, snx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))

    [CLD, _, TLD] = srcsum(StoDLP, peri_len, lx, lnx, sx, snx, sxp, jnp.array([]), sw, mu)
    [CRD, _, TRD] = srcsum(StoDLP, -peri_len, rx, rnx, sx, snx, sxp, jnp.array([]), sw, mu) 
    C = jnp.vstack([CRD - CLD, TRD - TLD])

    [QL, _, QLt] = StoSLP(lx, lnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # --- Full ELS matrix ---
    E = jnp.block([[A, B],
                   [C, Q]])

    return E, A, B, C, Q

# @jit
# A DL formulation on wall (like before) and SL+DL on particle
def ELSmatrix_ptcl(sx, snx, sxp, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu):
    # trlist = peri_len*jnp.array([-1,0,1])
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    [Asrcsum,_,_] = srcsum_self(StoDLP_self, StoDLP, trlist, sx, snx, sxp, scur, sw, mu)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    [A12_dl,_,_] = srcsum(StoDLP, trlist, sx, snx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) # from ptcl to s
    [A12_sl,_,_] = srcsum(StoSLP, trlist, sx, snx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    [A21,_,_] = srcsum(StoDLP, trlist, ptx, ptnx, sx, snx, sxp, jnp.array([]), sw, mu) # from s to ptcl
    [A22_dl,_,_] = srcsum_self(StoDLP_self, StoDLP, trlist, ptx, ptnx, ptxp, ptcur, ptw, mu) # double layer self does not have single-particle assumption

    n_tr = trlist.size
    Nptx = len(ptx)
    N_ptcl_double = Nptx/num_ptcl
    N_ptcl = int(N_ptcl_double)
    # jax.debug.print("Num nodes wall = {Nsx}, Num nodes ptcl each = {N_ptcl}", Nsx=Nsx, N_ptcl = N_ptcl)
    # first translation -- all src evals are far.
    ptx1 = ptx + trlist[0]
    [u, _, _] = StoSLP(ptx, ptnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, jnp.array([]))
    for i in range(1,n_tr):
        ptx1 = ptx + trlist[i]
        if i==1: # Have to hard code for JIT: len 3 trlist, middle one is 0.
            for j in range(num_ptcl):
                # Assume all srcs have same discretization orders
                ptx_cur = ptx[j*N_ptcl:(j+1)*N_ptcl]
                ptnx_cur = ptnx[j*N_ptcl:(j+1)*N_ptcl]
                ptxp_cur = ptxp[j*N_ptcl:(j+1)*N_ptcl]
                ptcur_cur = ptcur[j*N_ptcl:(j+1)*N_ptcl]
                ptw_cur = ptw[j*N_ptcl:(j+1)*N_ptcl]
                [tempu, _, _] = StoSLP_self(ptx_cur, ptnx_cur, ptx_cur, ptnx_cur, ptxp_cur, ptcur_cur, ptw_cur, mu, jnp.array([])) # self to self on particle j (no jump condition)
                # jax.debug.print("tempu self {}", tempu[0,0], ordered = True)
                u = u.at[j*N_ptcl:(j+1)*N_ptcl, j*N_ptcl:(j+1)*N_ptcl].add(tempu[:N_ptcl,:N_ptcl]) # x to x
                u = u.at[j*N_ptcl:(j+1)*N_ptcl, (Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl)].add(tempu[:N_ptcl,N_ptcl:]) # y to x
                u = u.at[(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(tempu[N_ptcl:,:N_ptcl]) # x to y
                u = u.at[(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl),(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl)].add(tempu[N_ptcl:,N_ptcl:]) # y to y
                for k in range(num_ptcl):
                    if k != j:
                        tx_cur = ptx[k*N_ptcl:(k+1)*N_ptcl]
                        tnx_cur = ptnx[k*N_ptcl:(k+1)*N_ptcl]
                        [ttu, _, _] = StoSLP(tx_cur, tnx_cur, ptx_cur, ptnx_cur, ptxp_cur, ptcur_cur, ptw_cur, mu, jnp.array([])) # particle j to k
                        # u[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl] += ttu
                        u = u.at[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(ttu[:N_ptcl,:N_ptcl]) # x to x
                        u = u.at[k*N_ptcl:(k+1)*N_ptcl, (Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl)].add(ttu[:N_ptcl,N_ptcl:]) # y to x
                        u = u.at[(Nptx+k*N_ptcl):(Nptx+(k+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(ttu[N_ptcl:,:N_ptcl]) # x to y
                        u = u.at[(Nptx+k*N_ptcl):(Nptx+(k+1)*N_ptcl),(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl)].add(ttu[N_ptcl:,N_ptcl:]) # y to y
        else:
            # right-side copy, all src evals are far.
            [tempu,_,_] = StoSLP(ptx, ptnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, jnp.array([]))
            u = u + tempu
    A22_sl = u

    A22 = jnp.eye(2*len(ptx)) / 2. + A22_dl + A22_sl # CHANGED MAR 2026: avoid flipping normals for close eval. Particle initialization changed accordingly.
    A = jnp.block([[A11, A12_dl+A12_sl],
                   [A21, A22]])

    [B1,_,_] = StoSLP(sx, snx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    [B2,_,_] = StoSLP(ptx, ptnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    B = jnp.vstack([B1, B2])

    [CLD, _, TLD] = srcsum(StoDLP, peri_len, lx, lnx, sx, snx, sxp, jnp.array([]), sw, mu)
    [CRD, _, TRD] = srcsum(StoDLP, -peri_len, rx, rnx, sx, snx, sxp, jnp.array([]), sw, mu) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])
    [CLDp, _, TLDp] = srcsum(StoDLP, peri_len, lx, lnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu)
    [CRDp, _, TRDp] = srcsum(StoDLP, -peri_len, rx, rnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    [CLSp, _, TLSp] = srcsum(StoSLP, peri_len, lx, lnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu)
    [CRSp, _, TRSp] = srcsum(StoSLP, -peri_len, rx, rnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    C = jnp.hstack([C1, C2])

    [QL, _, QLt] = StoSLP(lx, lnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # --- Full ELS matrix ---
    E = jnp.block([[A, B],
                   [C, Q]])

    return E, A, B, C, Q

ELSmatrix_ptcl = jit(ELSmatrix_ptcl, static_argnums=(10))

# Replace only parts of A relying on ptcl position, Bptcl, Cptcl.
def ELSmatrix_ptcl_update(A, B, C, Q, sx, snx, sxp, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu):
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    [A12_dl,_,_] = srcsum(StoDLP, trlist, sx, snx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) # from ptcl to s
    [A12_sl,_,_] = srcsum(StoSLP, trlist, sx, snx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    [A21,_,_] = srcsum(StoDLP, trlist, ptx, ptnx, sx, snx, sxp, jnp.array([]), sw, mu) # from s to ptcl
    n_tr = trlist.size
    Nptx = len(ptx)
    N_ptcl_double = Nptx/num_ptcl
    N_ptcl = int(N_ptcl_double)
    
    # We create a helper that computes the SUM of all translations for a pair (j, k)
    def compute_pair_interaction(j, k):
        
        # Self-interaction case (j == k)
        def self_interaction(_):
            r_start = 2 * Nsx + j * N_ptcl
            c_start = 2 * Nsx + j * N_ptcl
            
            # Slicing must be static or use dynamic_slice
            t_xx = jax.lax.dynamic_slice(A, (r_start, c_start), (N_ptcl, N_ptcl))
            t_xy = jax.lax.dynamic_slice(A, (r_start, c_start + Nptx), (N_ptcl, N_ptcl))
            t_yx = jax.lax.dynamic_slice(A, (r_start + Nptx, c_start), (N_ptcl, N_ptcl))
            t_yy = jax.lax.dynamic_slice(A, (r_start + Nptx, c_start + Nptx), (N_ptcl, N_ptcl))
            
            return jnp.block([[t_xx, t_xy], [t_yx, t_yy]])

        # Translated interaction case (j != k)
        def translated_interaction(_):
            # Vmap over the trlist (i index)
            def single_tr_step(tr_val):
                # Calculate dynamic start indices
                start_j = j * N_ptcl
                start_k = k * N_ptcl
                
                # Replace: ptx_j = (ptx + tr_val)[start_j : start_j + N_ptcl]
                # With:
                ptx_j = jax.lax.dynamic_slice(ptx + tr_val, (start_j,), (N_ptcl,))
                
                # Do the same for all other source inputs
                ptnx_j  = jax.lax.dynamic_slice(ptnx,  (start_j,), (N_ptcl,))
                ptxp_j  = jax.lax.dynamic_slice(ptxp,  (start_j,), (N_ptcl,))
                ptcur_j = jax.lax.dynamic_slice(ptcur, (start_j,), (N_ptcl,))
                ptw_j   = jax.lax.dynamic_slice(ptw,   (start_j,), (N_ptcl,))
                
                # And for the target inputs
                tx_k  = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
                tnx_k = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))
                
                slp, _, _ = StoSLP(tx_k, tnx_k, ptx_j, ptnx_j, ptxp_j, ptcur_j, ptw_j, mu, jnp.array([]))
                dlp, _, _ = StoDLP(tx_k, tnx_k, ptx_j, ptnx_j, ptxp_j, ptcur_j, ptw_j, mu, jnp.array([]))
                
                return slp + dlp
            
            results = jax.vmap(single_tr_step)(trlist)
            return jnp.sum(results, axis=0)

        return jax.lax.cond(j == k, self_interaction, translated_interaction, operand=None)

    # We use vmap to generate the (num_ptcl, num_ptcl, 2*N_ptcl, 2*N_ptcl) tensor
    indices = jnp.arange(num_ptcl)
    grid_ttu = jax.vmap(lambda j: jax.vmap(lambda k: compute_pair_interaction(j, k))(indices))(indices)
    # grid_ttu shape: (num_j, num_k, 2*N_ptcl, 2*N_ptcl)
    t_xx = grid_ttu[:, :, :N_ptcl, :N_ptcl]    # (num_j, num_k, N_ptcl, N_ptcl)
    t_xy = grid_ttu[:, :, :N_ptcl, N_ptcl:]    # (num_j, num_k, N_ptcl, N_ptcl)
    t_yx = grid_ttu[:, :, N_ptcl:, :N_ptcl]    # (num_j, num_k, N_ptcl, N_ptcl)
    t_yy = grid_ttu[:, :, N_ptcl:, N_ptcl:]    # (num_j, num_k, N_ptcl, N_ptcl)
    # 2. Reassemble each quadrant
    def assemble_quadrant(block_tensor):
        # block_tensor shape is (num_j, num_k, N_ptcl_row, N_ptcl_col)
        # To match your loop:
        # Axis 1 (k) determines the large-scale ROW
        # Axis 0 (j) determines the large-scale COLUMN
        # Final shape must be (num_k * N_ptcl, num_j * N_ptcl)
        return block_tensor.transpose(1, 2, 0, 3).reshape(Nptx, Nptx)
    # 3. Build the final 2Nptx x 2Nptx matrix
    u = jnp.block([
        [assemble_quadrant(t_xx), assemble_quadrant(t_xy)],
        [assemble_quadrant(t_yx), assemble_quadrant(t_yy)]
    ])

    A11 = A[:2*Nsx,:2*Nsx]
    Anew = jnp.block( [ [A11,A12_dl+A12_sl], [A21, u]])

    B1 = B[:2*Nsx,:]
    [B2,_,_] = StoSLP(ptx, ptnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    Bnew = jnp.vstack([B1,B2])

    C1 = C[:,:2*Nsx]
    [CLDp, _, TLDp] = srcsum(StoDLP, peri_len, lx, lnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu)
    [CRDp, _, TRDp] = srcsum(StoDLP, -peri_len, rx, rnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    [CLSp, _, TLSp] = srcsum(StoSLP, peri_len, lx, lnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu)
    [CRSp, _, TRSp] = srcsum(StoSLP, -peri_len, rx, rnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    Cnew = jnp.hstack([C1,C2])

    # --- Full ELS matrix ---
    E = jnp.block([[Anew, Bnew],
                   [Cnew, Q]])

    return E, Anew, Bnew, Cnew, Q
ELSmatrix_ptcl_update = jit(ELSmatrix_ptcl_update, static_argnums=(14))

# Replace only parts of A relying on ptcl position, Bptcl, Cptcl.
# DEPRECATED: changing immutable arrays are much harder in jax, so use new version with vmap instead.
def ELSmatrix_ptcl_update_old(A, B, C, Q, sx, snx, sxp, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu):
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    [A12_dl,_,_] = srcsum(StoDLP, trlist, sx, snx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) # from ptcl to s
    [A12_sl,_,_] = srcsum(StoSLP, trlist, sx, snx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    [A21,_,_] = srcsum(StoDLP, trlist, ptx, ptnx, sx, snx, sxp, jnp.array([]), sw, mu) # from s to ptcl
    n_tr = trlist.size
    Nptx = len(ptx)
    N_ptcl_double = Nptx/num_ptcl
    N_ptcl = int(N_ptcl_double)
    u = jnp.zeros((2*Nptx,2*Nptx))
    for j in range(num_ptcl):
        for k in range(num_ptcl):
            if j==k:
                # self-near to self does not change
                ttu= jnp.block([ [A[2*Nsx+j*N_ptcl:2*Nsx+(j+1)*N_ptcl, 2*Nsx+j*N_ptcl:2*Nsx+(j+1)*N_ptcl], A[2*Nsx+j*N_ptcl:2*Nsx+(j+1)*N_ptcl, 2*Nsx+Nptx+j*N_ptcl:2*Nsx+Nptx+(j+1)*N_ptcl]], 
                                          [A[2*Nsx+Nptx+j*N_ptcl:2*Nsx+Nptx+(j+1)*N_ptcl, 2*Nsx+j*N_ptcl:2*Nsx+(j+1)*N_ptcl], A[2*Nsx+Nptx+j*N_ptcl:2*Nsx+Nptx+(j+1)*N_ptcl, 2*Nsx+Nptx+j*N_ptcl:2*Nsx+Nptx+(j+1)*N_ptcl]]])
                u = u.at[j*N_ptcl:(j+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(ttu[:N_ptcl,:N_ptcl]) # x to x
                u = u.at[j*N_ptcl:(j+1)*N_ptcl, (Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl)].add(ttu[:N_ptcl,N_ptcl:]) # y to x
                u = u.at[(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(ttu[N_ptcl:,:N_ptcl]) # x to y
                u = u.at[(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl),(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl)].add(ttu[N_ptcl:,N_ptcl:]) # y to y
                
            else:
                for i in range(0,n_tr):
                    ptx1 = ptx + trlist[i]
                    # Assume all srcs have same discretization orders
                    ptx_cur = ptx1[j*N_ptcl:(j+1)*N_ptcl] # source shifted by trlist[i]
                    ptnx_cur = ptnx[j*N_ptcl:(j+1)*N_ptcl]
                    ptxp_cur = ptxp[j*N_ptcl:(j+1)*N_ptcl]
                    ptcur_cur = ptcur[j*N_ptcl:(j+1)*N_ptcl]
                    ptw_cur = ptw[j*N_ptcl:(j+1)*N_ptcl]
                    tx_cur = ptx[k*N_ptcl:(k+1)*N_ptcl] # target always in center copy
                    tnx_cur = ptnx[k*N_ptcl:(k+1)*N_ptcl]
                    [ttu, _, _] = StoSLP(tx_cur, tnx_cur, ptx_cur, ptnx_cur, ptxp_cur, ptcur_cur, ptw_cur, mu, jnp.array([])) # particle j to k SL
                    [ttuD, _, _] = StoDLP(tx_cur, tnx_cur, ptx_cur, ptnx_cur, ptxp_cur, ptcur_cur, ptw_cur, mu, jnp.array([])) # particle j to k DL
                    ttu = ttu + ttuD
                    
                    u = u.at[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(ttu[:N_ptcl,:N_ptcl]) # x to x
                    u = u.at[k*N_ptcl:(k+1)*N_ptcl, (Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl)].add(ttu[:N_ptcl,N_ptcl:]) # y to x
                    u = u.at[(Nptx+k*N_ptcl):(Nptx+(k+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(ttu[N_ptcl:,:N_ptcl]) # x to y
                    u = u.at[(Nptx+k*N_ptcl):(Nptx+(k+1)*N_ptcl),(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl)].add(ttu[N_ptcl:,N_ptcl:]) # y to y

    A11 = A[:2*Nsx,:2*Nsx]
    Anew = jnp.block( [ [A11,A12_dl+A12_sl], [A21, u]])

    B1 = B[:2*Nsx,:]
    [B2,_,_] = StoSLP(ptx, ptnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    Bnew = jnp.vstack([B1,B2])

    C1 = C[:,:2*Nsx]
    [CLDp, _, TLDp] = srcsum(StoDLP, peri_len, lx, lnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu)
    [CRDp, _, TRDp] = srcsum(StoDLP, -peri_len, rx, rnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    [CLSp, _, TLSp] = srcsum(StoSLP, peri_len, lx, lnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu)
    [CRSp, _, TRSp] = srcsum(StoSLP, -peri_len, rx, rnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    Cnew = jnp.hstack([C1,C2])

    # --- Full ELS matrix ---
    E = jnp.block([[Anew, Bnew],
                   [Cnew, Q]])

    return E, Anew, Bnew, Cnew, Q


@jit
def ELSmatrix_stokeslet(sx, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu):
    # trlist = peri_len*jnp.array([-1,0,1])
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    [Asrcsum,_,_] = srcsum_self(StoDLP_self, StoDLP, trlist, sx, snx, sxp, scur, sw, mu)
    A = -jnp.eye(2*Nsx) / 2 + Asrcsum

    x_stokeslet_near = x_stokeslet + trlist[0] # left copy first
    [A_ptsrc_near,_,_] = StoSLP(sx, snx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
    [CR_ptsrc_near,_,TR_ptsrc_near] = StoSLP(rx, rnx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
    for i in range(1,len(trlist)):
        x_stokeslet_near = x_stokeslet + trlist[i]
        [A_ptsrc_temp,_,_] = StoSLP(sx, snx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
        if i==2: # right copy
            [CL_ptsrc_near,_,TL_ptsrc_near] = StoSLP(lx, lnx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
        A_ptsrc_near += A_ptsrc_temp 
    C_ptsrc_near = jnp.vstack([CR_ptsrc_near-CL_ptsrc_near, TR_ptsrc_near-TL_ptsrc_near])

    [B,_,_] = StoSLP(sx, snx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))

    [CLD, _, TLD] = srcsum(StoDLP, peri_len, lx, lnx, sx, snx, sxp, jnp.array([]), sw, mu)
    [CRD, _, TRD] = srcsum(StoDLP, -peri_len, rx, rnx, sx, snx, sxp, jnp.array([]), sw, mu) 
    C = jnp.vstack([CRD - CLD, TRD - TLD])       

    [QL, _, QLt] = StoSLP(lx, lnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # --- Full ELS matrix ---
    # E = jnp.block([[A, B],
    #                [C, Q]])
    E = jnp.block([[A, A_ptsrc_near, B], 
                   [C, C_ptsrc_near, Q]])

    return E, A, B, C, Q

@jit
# Stokeslet E matrix with constraint that sum of forces is 0.
def ELSmatrix_stokeslet_f0(sx, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu):
    # trlist = peri_len*jnp.array([-1,0,1])
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    [Asrcsum,_,_] = srcsum_self(StoDLP_self, StoDLP, trlist, sx, snx, sxp, scur, sw, mu)
    A = -jnp.eye(2*Nsx) / 2 + Asrcsum

    x_stokeslet_near = x_stokeslet + trlist[0] # left copy first
    [A_ptsrc_near,_,_] = StoSLP(sx, snx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
    [CR_ptsrc_near,_,TR_ptsrc_near] = StoSLP(rx, rnx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
    for i in range(1,len(trlist)):
        x_stokeslet_near = x_stokeslet + trlist[i]
        [A_ptsrc_temp,_,_] = StoSLP(sx, snx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
        if i==2: # right copy
            [CL_ptsrc_near,_,TL_ptsrc_near] = StoSLP(lx, lnx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
        A_ptsrc_near += A_ptsrc_temp 
    C_ptsrc_near = jnp.vstack([CR_ptsrc_near-CL_ptsrc_near, TR_ptsrc_near-TL_ptsrc_near])

    [B,_,_] = StoSLP(sx, snx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))

    [CLD, _, TLD] = srcsum(StoDLP, peri_len, lx, lnx, sx, snx, sxp, jnp.array([]), sw, mu)
    [CRD, _, TRD] = srcsum(StoDLP, -peri_len, rx, rnx, sx, snx, sxp, jnp.array([]), sw, mu) 
    C = jnp.vstack([CRD - CLD, TRD - TLD])       

    [QL, _, QLt] = StoSLP(lx, lnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # sum of F on stokeslet is 0
    num_stokeslet = len(x_stokeslet)
    F = jnp.zeros((2,2*num_stokeslet))
    F = F.at[0,:num_stokeslet].set(jnp.ones((num_stokeslet,))) # sum of x-dir F
    F = F.at[1,num_stokeslet:].set(jnp.ones((num_stokeslet,))) # sum of y-dir F
    
    # --- Full ELS matrix ---
    E = jnp.block([[A, A_ptsrc_near, B], 
                   [C, C_ptsrc_near, Q],
                   [jnp.zeros((2,2*Nsx)), F, jnp.zeros((2,2*len(px)))]])

    return E, A, B, C, Q

# A DL formulation on wall (like before) and SL+DL on particle
# Include RBM with unknown U and Omega; implement net force and net torque zero conditions.
def ELSmatrix_RBM(sx, snx, sxp, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu):
    # trlist = peri_len*jnp.array([-1,0,1])
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    [Asrcsum,_,_] = srcsum_self(StoDLP_self, StoDLP, trlist, sx, snx, sxp, scur, sw, mu)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    [A12_dl,_,_] = srcsum(StoDLP, trlist, sx, snx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) # from ptcl to s
    [A12_sl,_,_] = srcsum(StoSLP, trlist, sx, snx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    [A21,_,A21_T] = srcsum(StoDLP, trlist, ptx, ptnx, sx, snx, sxp, jnp.array([]), sw, mu) # from s to ptcl
    [A22_dl,_,A22_dl_T] = srcsum_self(StoDLP_self, StoDLP, trlist, ptx, ptnx, ptxp, ptcur, ptw, mu) # double layer self does not have single-particle assumption
    
    n_tr = trlist.size
    N_ptcl_double = Nsx/num_ptcl
    # N_ptcl = N_ptcl_double.astype(int)
    N_ptcl = int(N_ptcl_double)
    # first translation -- all src evals are far.
    ptx1 = ptx + trlist[0]
    [u,p,T] = StoSLP(ptx, ptnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, jnp.array([]))
    for i in range(1,n_tr):
        ptx1 = ptx + trlist[i]
        if i==1: # Have to hard code for JIT: len 3 trlist, middle one is 0.
            # [tempu,tempp,tempT] = kernel_self(sx, snx, sx, snx, sxp, scur, sw, mu, jnp.array([]))

            for j in range(num_ptcl):
                # Assume all srcs have same discretization orders
                ptx_cur = ptx[j*N_ptcl:(j+1)*N_ptcl]
                ptnx_cur = ptnx[j*N_ptcl:(j+1)*N_ptcl]
                ptxp_cur = ptxp[j*N_ptcl:(j+1)*N_ptcl]
                ptcur_cur = ptcur[j*N_ptcl:(j+1)*N_ptcl]
                ptw_cur = ptw[j*N_ptcl:(j+1)*N_ptcl]
                [tempu,tempp,tempT] = StoSLP_self(ptx_cur, ptnx_cur, ptx_cur, ptnx_cur, ptxp_cur, ptcur_cur, ptw_cur, mu, jnp.array([])) # self to self on particle j (no jump condition)
                # jax.debug.print("tempu self {}", tempu[0,0], ordered = True)
                u = u.at[j*N_ptcl:(j+1)*N_ptcl, j*N_ptcl:(j+1)*N_ptcl].add(tempu[:N_ptcl,:N_ptcl]) # x to x
                u = u.at[j*N_ptcl:(j+1)*N_ptcl, (Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(tempu[:N_ptcl,N_ptcl:]) # y to x
                u = u.at[(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(tempu[N_ptcl:,:N_ptcl]) # x to y
                u = u.at[(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl),(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(tempu[N_ptcl:,N_ptcl:]) # y to y
                p = p.at[j*N_ptcl:(j+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(tempp[:,:N_ptcl]) # x to p
                p = p.at[j*N_ptcl:(j+1)*N_ptcl,(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(tempp[:,N_ptcl:]) # y to p
                # TODO: Check T's ordering.
                T = T.at[j*N_ptcl:(j+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(tempT[:N_ptcl,:N_ptcl]) # x to x
                T = T.at[j*N_ptcl:(j+1)*N_ptcl, (Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(tempT[:N_ptcl,N_ptcl:]) # y to x
                T = T.at[(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(tempT[N_ptcl:,:N_ptcl]) # x to y
                T = T.at[(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl),(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(tempT[N_ptcl:,N_ptcl:]) # y to y 
                for k in range(num_ptcl):
                    if k != j:
                        tx_cur = ptx[k*N_ptcl:(k+1)*N_ptcl]
                        tnx_cur = ptnx[k*N_ptcl:(k+1)*N_ptcl]
                        [ttu, ttp, ttT] = StoSLP(tx_cur, tnx_cur, ptx_cur, ptnx_cur, ptxp_cur, ptcur_cur, ptw_cur, mu, jnp.array([])) # particle j to k
                        # u[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl] += ttu
                        u = u.at[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(ttu[:N_ptcl,:N_ptcl]) # x to x
                        u = u.at[k*N_ptcl:(k+1)*N_ptcl, (Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(ttu[:N_ptcl,N_ptcl:]) # y to x
                        u = u.at[(Nsx+k*N_ptcl):(Nsx+(k+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(ttu[N_ptcl:,:N_ptcl]) # x to y
                        u = u.at[(Nsx+k*N_ptcl):(Nsx+(k+1)*N_ptcl),(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(ttu[N_ptcl:,N_ptcl:]) # y to y
                        p = p.at[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(ttp[:,:N_ptcl]) # x to p
                        p = p.at[k*N_ptcl:(k+1)*N_ptcl,(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(ttp[:,N_ptcl:]) # y to p
                        # TODO: Check T's ordering.
                        T = T.at[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(ttT[:N_ptcl,:N_ptcl]) # x to x
                        T = T.at[k*N_ptcl:(k+1)*N_ptcl, (Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(ttT[:N_ptcl,N_ptcl:]) # y to x
                        T = T.at[(Nsx+k*N_ptcl):(Nsx+(k+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(ttT[N_ptcl:,:N_ptcl]) # x to y
                        T = T.at[(Nsx+k*N_ptcl):(Nsx+(k+1)*N_ptcl),(Nsx+j*N_ptcl):(Nsx+(j+1)*N_ptcl)].add(ttT[N_ptcl:,N_ptcl:]) # y to y 
        else:
            # right-side copy, all src evals are far.
            [tempu,_,tempT] = StoSLP(ptx, ptnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, jnp.array([]))
            u = u + tempu
            T = T + tempT
    A22_sl = u
    A22_sl_T = T

    A22 = -jnp.eye(2*len(ptx)) / 2. + A22_dl + A22_sl
    A22_T = -jnp.eye(2*len(ptx)) / 2. + A22_dl_T + A22_sl_T
    A = jnp.block([[A11, A12_dl+A12_sl],
                   [A21, A22]])

    [B1,_,_] = StoSLP(sx, snx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    [B2,_,_] = StoSLP(ptx, ptnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    B = jnp.vstack([B1, B2])

    [CLD, _, TLD] = srcsum(StoDLP, peri_len, lx, lnx, sx, snx, sxp, jnp.array([]), sw, mu)
    [CRD, _, TRD] = srcsum(StoDLP, -peri_len, rx, rnx, sx, snx, sxp, jnp.array([]), sw, mu) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])
    [CLDp, _, TLDp] = srcsum(StoDLP, peri_len, lx, lnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu)
    [CRDp, _, TRDp] = srcsum(StoDLP, -peri_len, rx, rnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    [CLSp, _, TLSp] = srcsum(StoSLP, peri_len, lx, lnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu)
    [CRSp, _, TRSp] = srcsum(StoSLP, -peri_len, rx, rnx, ptx, ptnx, ptxp, jnp.array([]), ptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    C = jnp.hstack([C1, C2])

    [QL, _, QLt] = StoSLP(lx, lnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # --- Full ELS matrix ---
    E = jnp.block([[A, B],
                   [C, Q]])

    return E, A, B, C, Q



# Since JAX tracer distinguishes different inputs, need to use self-kernel when inputting self-interactions.
# kernel_self and kernel are Python-callable kernel functions (e.g. StoDLP_self, StoDLP).
# These must be treated as static (compile-time constants) when jitting this routine,
# otherwise JAX will attempt to interpret the function object as an array and raise
# a TypeError. Mark them static via static_argnums so their identity is known at
# compile time and the jitted function can be traced correctly.
def srcsum_self(kernel_self, kernel, trlist, sx, snx, sxp, scur, sw, mu):
    trlist = jnp.atleast_1d(trlist)
    n_tr = trlist.size

    # first translation
    sx1 = sx + trlist[0]
    [u,p,T] = kernel(sx, snx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))
    for i in range(1,n_tr):
        sx1 = sx + trlist[i]
        if i==1: # Have to hard code: len 3 trlist, middle one is 0.
            [tempu,tempp,tempT] = kernel_self(sx, snx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
        else:
            [tempu,tempp,tempT] = kernel(sx, snx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))
        u = u + tempu
        p = p + tempp
        T = T + tempT
    return u,p,T

"""
# To allow for multi-particle interactions, self vs target kernel eval must be distinguished.
def srcsum_self_ptcl(kernel_self, kernel, trlist, sx, snx, sxp, scur, sw, num_src, mu):
    trlist = jnp.atleast_1d(trlist)
    n_tr = trlist.size
    N_tot = sx.size
    # N_ptcl = int(N_tot / num_src)
    N_ptcl_double = N_tot/num_src
    N_ptcl = N_ptcl_double.astype(int)
    # jax.debug.print("debug: N nodes on ptcl is {}", N_ptcl)
    # print("N_ptcl = " , N_ptcl)

    # first translation -- all src evals are far.
    sx1 = sx + trlist[0]
    [u,p,T] = kernel(sx, snx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))
    for i in range(1,n_tr):
        sx1 = sx + trlist[i]
        if i==1: # Have to hard code for JIT: len 3 trlist, middle one is 0.
            # [tempu,tempp,tempT] = kernel_self(sx, snx, sx, snx, sxp, scur, sw, mu, jnp.array([]))

            for j in range(num_src):
                # Assume all srcs have same discretization orders
                sx_cur = sx[j*N_ptcl:(j+1)*N_ptcl]
                snx_cur = snx[j*N_ptcl:(j+1)*N_ptcl]
                sxp_cur = sxp[j*N_ptcl:(j+1)*N_ptcl]
                scur_cur = scur[j*N_ptcl:(j+1)*N_ptcl]
                sw_cur = sw[j*N_ptcl:(j+1)*N_ptcl]
                [tempu,tempp,tempT] = kernel_self(sx_cur, snx_cur, sx_cur, snx_cur, sxp_cur, scur_cur, sw_cur, mu, jnp.array([])) # self to self on particle j (no jump condition)
                # jax.debug.print("tempu self {}", tempu[0,0], ordered = True)
                u = u.at[j*N_ptcl:(j+1)*N_ptcl, j*N_ptcl:(j+1)*N_ptcl].add(tempu[:N_ptcl,:N_ptcl]) # x to x
                u = u.at[j*N_ptcl:(j+1)*N_ptcl, (N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(tempu[:N_ptcl,N_ptcl:]) # y to x
                u = u.at[(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(tempu[N_ptcl:,:N_ptcl]) # x to y
                u = u.at[(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl),(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(tempu[N_ptcl:,N_ptcl:]) # y to y
                p = p.at[j*N_ptcl:(j+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(tempp[:,:N_ptcl]) # x to p
                p = p.at[j*N_ptcl:(j+1)*N_ptcl,(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(tempp[:,N_ptcl:]) # y to p
                # TODO: Check T's ordering.
                T = T.at[j*N_ptcl:(j+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(tempT[:N_ptcl,:N_ptcl]) # x to x
                T = T.at[j*N_ptcl:(j+1)*N_ptcl, (N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(tempT[:N_ptcl,N_ptcl:]) # y to x
                T = T.at[(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(tempT[N_ptcl:,:N_ptcl]) # x to y
                T = T.at[(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl),(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(tempT[N_ptcl:,N_ptcl:]) # y to y 
                for k in range(num_src):
                    if k != j:
                        tx_cur = sx[k*N_ptcl:(k+1)*N_ptcl]
                        tnx_cur = snx[k*N_ptcl:(k+1)*N_ptcl]
                        [ttu, ttp, ttT] = kernel(tx_cur, tnx_cur, sx_cur, snx_cur, sxp_cur, scur_cur, sw_cur, mu, jnp.array([])) # particle j to k
                        # u[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl] += ttu
                        u = u.at[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(ttu[:N_ptcl,:N_ptcl]) # x to x
                        u = u.at[k*N_ptcl:(k+1)*N_ptcl, (N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(ttu[:N_ptcl,N_ptcl:]) # y to x
                        u = u.at[(N_tot+k*N_ptcl):(N_tot+(k+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(ttu[N_ptcl:,:N_ptcl]) # x to y
                        u = u.at[(N_tot+k*N_ptcl):(N_tot+(k+1)*N_ptcl),(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(ttu[N_ptcl:,N_ptcl:]) # y to y
                        p = p.at[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(ttp[:,:N_ptcl]) # x to p
                        p = p.at[k*N_ptcl:(k+1)*N_ptcl,(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(ttp[:,N_ptcl:]) # y to p
                        # TODO: Check T's ordering.
                        T = T.at[k*N_ptcl:(k+1)*N_ptcl,j*N_ptcl:(j+1)*N_ptcl].add(ttT[:N_ptcl,:N_ptcl]) # x to x
                        T = T.at[k*N_ptcl:(k+1)*N_ptcl, (N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(ttT[:N_ptcl,N_ptcl:]) # y to x
                        T = T.at[(N_tot+k*N_ptcl):(N_tot+(k+1)*N_ptcl),j*N_ptcl:(j+1)*N_ptcl].add(ttT[N_ptcl:,:N_ptcl]) # x to y
                        T = T.at[(N_tot+k*N_ptcl):(N_tot+(k+1)*N_ptcl),(N_tot+j*N_ptcl):(N_tot+(j+1)*N_ptcl)].add(ttT[N_ptcl:,N_ptcl:]) # y to y 
        else:
            # right-side copy, all src evals are far.
            [tempu,tempp,tempT] = kernel(sx, snx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))
            u = u + tempu
            p = p + tempp
            T = T + tempT

    return u,p,T
"""

# kernel is a Python-callable kernel function and must be static for the same reason
def srcsum(kernel, trlist, tx, tnx, sx, snx, sxp, scur, sw, mu):
    trlist = jnp.atleast_1d(trlist)
    n_tr = trlist.size

    # first translation
    sx1 = sx + trlist[0]
    [u,p,T] = kernel(tx, tnx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))

    for i in range(1,n_tr):
        sx1 = sx + trlist[i]
        [tempu,tempp,tempT] = kernel(tx, tnx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))
        u = u + tempu
        p = p + tempp
        T = T + tempT

    return u,p,T

# Specific close eval functions for particles, hardcoded: side='e', using either stosl or dl closeglobal.
# stosl close requires looping over the particles.
@jit
def srcsum_ptcl_stodl_closeglobal(trlist, tx, tnx, ptx, ptnx, pta, ptxp, ptcur, ptw, ptwxp, sigma, mu):
    trlist = jnp.atleast_1d(trlist)

    # left copy, all should be far from center copy particles.
    ptx1 = ptx + trlist[0]
    [u,p,_] = StoDLP(tx, tnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, sigma)

    # right copy, also should be far.
    ptx1 = ptx + trlist[-1]
    [tempu,tempp,_] = StoDLP(tx, tnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, sigma)
    u = u + tempu
    p = p + tempp

    ptx1 = ptx
    Nptx = len(ptx)
    num_ptcl = len(pta)
    N_ptcl = int(Nptx/num_ptcl) # num discr. points per ptcl, assumed to be equal.
    for j in range(num_ptcl):
        ptx_cur = ptx1[j*N_ptcl:(j+1)*N_ptcl]
        ptxp_cur = ptxp[j*N_ptcl:(j+1)*N_ptcl]
        ptwxp_cur = ptwxp[j*N_ptcl:(j+1)*N_ptcl]
        sigma_cur = jnp.concatenate([sigma[j*N_ptcl:(j+1)*N_ptcl], sigma[Nptx+j*N_ptcl:Nptx+(j+1)*N_ptcl]]) 

        [tempu,tempp] = stoDLP_closeglobal(tx, ptx_cur, ptxp_cur, ptwxp_cur, sigma_cur, mu)
        u = u + tempu.ravel()
        p = p + tempp.ravel()

    return u,p

@jit
def srcsum_ptcl_stosl_closeglobal(trlist, tx, tnx, ptx, ptnx, ptt, pta, ptcur, ptxp, ptw, sigma, mu):
    trlist = jnp.atleast_1d(trlist)

    # left copy, all should be far from center copy particles.
    ptx1 = ptx + trlist[0]
    [u,p,T] = StoSLP(tx, tnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, sigma)

    # right copy, also should be far.
    ptx1 = ptx + trlist[-1]
    [tempu,tempp,tempT] = StoSLP(tx, tnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, sigma)
    u = u + tempu
    p = p + tempp
    T = T + tempT

    # Center copy, use closeglobal. TODO: hardcoded to only have 3 values in trlist, center copy being 0.
    ptx1 = ptx
    pta1 = pta
    Nptx = len(ptx)
    num_ptcl = len(pta)
    N_ptcl = int(Nptx/num_ptcl) # num discr. points per ptcl, assumed to be equal.
    for j in range(num_ptcl):
        ptx_cur = ptx1[j*N_ptcl:(j+1)*N_ptcl]
        ptxp_cur = ptxp[j*N_ptcl:(j+1)*N_ptcl]
        ptw_cur = ptw[j*N_ptcl:(j+1)*N_ptcl]
        ptt_cur = ptt[j*N_ptcl:(j+1)*N_ptcl]
        pta_cur = pta1[j]
        sigma_cur = jnp.concatenate([sigma[j*N_ptcl:(j+1)*N_ptcl], sigma[Nptx+j*N_ptcl:Nptx+(j+1)*N_ptcl]]) # TODO: size correct?

        [tempu,tempp,tempT] = stoSLP_closeglobal(tx, tnx, ptx_cur, ptxp_cur, ptt_cur, pta_cur, ptw_cur, sigma_cur, mu)
        
        u = u + tempu.ravel()
        p = p + tempp.ravel()
        T = T + tempT.ravel()

    return u,p,T

# Source sum for panel-based discr. on wall (since only DL available, a more specific function)
@jit
def srcsum_wall(trlist, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sws, mu):
    trlist = jnp.atleast_1d(trlist)

    # first translation
    sx1 = sx + trlist[0]
    [u,p,_] = StoDLP(tx, tnx, sx1, snx, sxp, scur, sws, mu, jnp.array([]))

    # Hardcode trlist 3 entries, L, middle, R
    sx1 = sx + trlist[1]
    sxlo1 = sxlo + trlist[1]
    sxhi1 = sxhi + trlist[1]
    sigma_real = jnp.eye(2*sx.shape[0]) # TODO: is it necessary to form the matrix here?
    [tempu,tempp] = stoDLP_closepanel(tx, tnx, sx1, sxp, sws, scur, sxlo1, sxhi1, sigma_real, mu,'i')
    u = u + tempu
    p = p + tempp

    sx1 = sx + trlist[2]
    [tempu,tempp,_] = StoDLP(tx, tnx, sx1, snx, sxp, scur, sws, mu, jnp.array([]))
    u = u + tempu
    p = p + tempp
    
    return u,p

# Wrap the above functions with jax.jit after definition to avoid decorator-time
# TypeError on some JAX versions where calling `jit(...)` as a decorator at
# import time can raise "jit() missing 1 required positional argument: 'fun'".
srcsum_self = jit(srcsum_self, static_argnums=(0,1))
srcsum = jit(srcsum, static_argnums=(0,))
# srcsum_self_ptcl = jit(srcsum_self_ptcl, static_argnums=(0,1,8))

def sumblk(At, M, trlist):
    n = len(trlist)
    d = int(At.shape[0] / M / n)
    # print(f"\n debug: d from srcsum2 is {d}")
    A = jnp.zeros((M*d, At.shape[1]))
    print("in sumblk, size of matrix is ", A.shape)
    print(A)
    if d==1:
        ii = jnp.arange(M)
    else:
        ii = jnp.concatenate([jnp.arange(M), n*M + jnp.arange(M)])
    for i in range(n):
        A = A + At[ii + (i-1)*M,:]
    return A

def srcsum2(kernel, trlist, tx, tnx, sx, snx, sxp, scur, mu, dens):
    n = len(trlist)
    M = len(tx) # number of targets
    ttx = jnp.array([])
    for i in range(n):
        ttx = jnp.append(ttx, tx-trlist[i])
    
    ttnx = jnp.tile(tnx, (n,1))
    print("repeating target in srcsum 2: ",ttx.shape," ", ttnx.shape," ")

    [At, Bt, _] = kernel(ttx, ttnx, sx, snx, sxp, scur, mu, dens)
    print(At)
    print(Bt)

    # TODO: use to_ndarray instead..
    At = jnp.tile(At, (1,1)).T
    Bt = jnp.tile(Bt, (1,1)).T

    A = sumblk(At,M,trlist)
    B = sumblk(Bt,M,trlist)
    return A,B

def evalsol(tx, tnx, sx, snx, sxp, scur, px, pxp, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    wall_dens = edens[:2*N_wall]
    prx_dens = edens[2*N_wall:]

    [u, p, _] = StoSLP(tx, tnx, px, jnp.array([]), pxp, jnp.array([]), mu, prx_dens)
    [uD, pD] = srcsum2(StoDLP, trlist, tx, tnx, sx, snx, sxp, scur, mu, wall_dens) 
    # TODO: use to_ndarray.
    u = jnp.tile(u, (1,1)).T
    p = jnp.tile(p, (1,1)).T

    u = u+uD
    p = p+pD
    return u,p

@jit
def evalsol2(tx, tnx, sx, snx, sxp, scur, sw, px, pxp, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    wall_dens = edens[:2*N_wall]
    prx_dens = edens[2*N_wall:]

    [u, p, _] = StoSLP(tx, tnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, prx_dens)
    [uD, pD, _] = srcsum(StoDLP, trlist, tx, tnx, sx, snx, sxp, scur, sw, mu)
    uD = uD @ wall_dens
    pD = pD @ wall_dens

    u = u+uD
    p = p+pD
    return u,p

@jit
def evalsol2_panel(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, px, pxp, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    wall_dens = edens[:2*N_wall]
    prx_dens = edens[2*N_wall:]

    [u, p, _] = StoSLP(tx, tnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, prx_dens)
    [uD, pD] = srcsum_wall(trlist, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu)
    uD = uD @ wall_dens
    pD = pD @ wall_dens

    u = u+uD
    p = p+pD
    return u,p

@jit
def evalsol_ptcl_panel(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptcur, ptw, ptwxp, px, pxp, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    N_ptcl = len(ptx)
    wall_dens = edens[:2*N_wall]
    ptcl_dens = edens[2*N_wall:2*(N_wall+N_ptcl)]
    prx_dens = edens[2*(N_wall+N_ptcl):]

    [u, p, _] = StoSLP(tx, tnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, prx_dens)
    [uD, pD] = srcsum_wall(trlist, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu)

    [uSp, pSp, _] = srcsum_ptcl_stosl_closeglobal(trlist,tx,tnx,ptx,ptnx,ptt,pta,ptcur,ptxp,ptw,ptcl_dens,mu)
    [uDp, pDp] = srcsum_ptcl_stodl_closeglobal(trlist,tx,tnx,ptx,ptnx,pta,ptxp,ptcur,ptw,ptwxp,ptcl_dens,mu)

    uD = uD @ wall_dens
    pD = pD @ wall_dens
    uDSp = uDp + uSp
    pDSp = pDp + pSp

    u = u+uD+uDSp
    p = p+pD+pDSp
    return u,p

@jit
def evalsol_ptcl(tx, tnx, sx, snx, sxp, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, px, pxp, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    N_ptcl = len(ptx)
    wall_dens = edens[:2*N_wall]
    ptcl_dens = edens[2*N_wall:2*(N_wall+N_ptcl)]
    prx_dens = edens[2*(N_wall+N_ptcl):]

    [u, p, _] = StoSLP(tx, tnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, prx_dens)
    [uD, pD, _] = srcsum(StoDLP, trlist, tx, tnx, sx, snx, sxp, scur, sw, mu) # no need for DL traction so fill in tnx with snx
    [uDp, pDp, _] = srcsum(StoDLP, trlist, tx, tnx, ptx, ptnx, ptxp, ptcur, ptw, mu)
    [uSp, pSp, _] = srcsum(StoSLP, trlist, tx, tnx, ptx, ptnx, ptxp, ptcur, ptw, mu)
    uD = uD @ wall_dens
    pD = pD @ wall_dens
    uDSp = (uDp + uSp) @ ptcl_dens
    pDSp = (pDp + pSp) @ ptcl_dens

    u = u+uD+uDSp
    p = p+pD+pDSp
    return u,p

@jit
def evalsol_stokeslet_panel(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    N_stokeslet = len(x_stokeslet)
    wall_dens = edens[:2*N_wall]
    f_stokeslet = edens[2*N_wall:2*(N_wall+N_stokeslet)]
    prx_dens = edens[2*(N_wall+N_stokeslet):]

    [u, p, _] = StoSLP(tx, tnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, prx_dens)
    [uD, pD] = srcsum_wall(trlist, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu) # no need for DL traction so fill in tnx with snx
    x_stokeslet_near = x_stokeslet + trlist[0] # left copy first
    [u_ptsrc, p_ptsrc, _] = StoSLP(tx, tnx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
    for i in range(1,len(trlist)):
        x_stokeslet_near = x_stokeslet + trlist[i]
        [u_ptsrc_temp, p_ptsrc_temp, _] = StoSLP(tx, tnx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
        u_ptsrc += u_ptsrc_temp 
        p_ptsrc += p_ptsrc_temp 
    uD = uD @ wall_dens
    pD = pD @ wall_dens
    uD_sto = u_ptsrc @ f_stokeslet
    pD_sto = p_ptsrc @ f_stokeslet

    u = u+uD+uD_sto
    p = p+pD+pD_sto
    return u,p

@jit
def evalsol_stokeslet(tx, tnx, sx, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    N_stokeslet = len(x_stokeslet)
    wall_dens = edens[:2*N_wall]
    f_stokeslet = edens[2*N_wall:2*(N_wall+N_stokeslet)]
    prx_dens = edens[2*(N_wall+N_stokeslet):]

    [u, p, _] = StoSLP(tx, tnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, prx_dens)
    [uD, pD, _] = srcsum(StoDLP, trlist, tx, tnx, sx, snx, sxp, scur, sw, mu) # no need for DL traction so fill in tnx with snx
    x_stokeslet_near = x_stokeslet + trlist[0] # left copy first
    [u_ptsrc, p_ptsrc, _] = StoSLP(tx, tnx, x_stokeslet, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
    for i in range(1,len(trlist)):
        x_stokeslet_near = x_stokeslet + trlist[i]
        [u_ptsrc_temp, p_ptsrc_temp, _] = StoSLP(tx, tnx, x_stokeslet_near, jnp.array([]), xp_stokeslet, jnp.array([]), w_stokeslet, mu, jnp.array([]))
        u_ptsrc += u_ptsrc_temp 
        p_ptsrc += p_ptsrc_temp 
    uD = uD @ wall_dens
    pD = pD @ wall_dens
    uD_sto = u_ptsrc @ f_stokeslet
    pD_sto = p_ptsrc @ f_stokeslet

    u = u+uD+uD_sto
    p = p+pD+pD_sto
    return u,p