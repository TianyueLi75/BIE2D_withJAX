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
def ELSmatrix(sx, snx, scur, sw, px, pwt, lx, lnx, rx, rnx, peri_len, mu):
    Nsx = len(sx)
    [Asrcsum,_,_] = srcsum_dl_self(peri_len, sx, snx, scur, sw, mu)
    A = -jnp.eye(2*Nsx) / 2 + Asrcsum

    [B,_,_] = StoSLP(sx, snx, px, pwt, mu, jnp.array([]))

    [CLD, _, TLD] = srcsum_dl(peri_len, lx, lnx, sx, snx, sw, mu)
    [CRD, _, TRD] = srcsum_dl(-peri_len, rx, rnx, sx, snx, sw, mu) 
    C = jnp.vstack([CRD - CLD, TRD - TLD])

    [QL, _, QLt] = StoSLP(lx, lnx, px, pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # --- Full ELS matrix ---
    E = jnp.block([[A, B],
                   [C, Q]])

    return E, A, B, C, Q


def ELSmatrix_ptcl(sx, snx, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu):
    """A DL formulation on wall (like before) and SL+DL on particle"""
    # trlist = peri_len*jnp.array([-1,0,1])
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    [Asrcsum,_,_] = srcsum_dl_self(peri_len, sx, snx, scur, sw, mu)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    [A12_dl,_,_] = srcsum_dl(trlist, sx, snx, ptx, ptnx, ptw, mu) # from ptcl to s
    [A12_sl,_,_] = srcsum_sl(trlist, sx, snx, ptx, ptw, mu) 
    [A21,_,_] = srcsum_dl(trlist, ptx, ptnx, sx, snx, sw, mu) # from s to ptcl
    [A22_dl,_,_] = srcsum_dl_self(peri_len, ptx, ptnx, ptcur, ptw, mu) # double layer self does not have single-particle assumption
    # [A12_sl, _, _] = srcsum_ptcl_stosl_closeglobal(peri_len,sx,snx,ptx,ptt,pta,ptxp,ptw,jnp.eye(ptx.shape[0]),mu)
    # [A12_dl, _] = srcsum_ptcl_stodl_closeglobal(peri_len,tx,tnx,ptx,ptnx,pta,ptxp,ptw,ptwxp,jnp.eye(ptx.shape[0]),mu)


    Nptx = len(ptx)
    N_ptcl_double = Nptx/num_ptcl
    N_ptcl = int(N_ptcl_double)
    # jax.debug.print("Num nodes wall = {Nsx}, Num nodes ptcl each = {N_ptcl}", Nsx=Nsx, N_ptcl = N_ptcl)
    # We create a helper that computes the SUM of all translations for a pair (j, k)
    def compute_pair_interaction(j, k):
        # 1. Define the 0-translation case (Middle of trlist)
        # This handles the Singularity (j==k) or standard SLP (j!=k)
        def zero_translation_contribution():
            start_j = j * N_ptcl
            start_k = k * N_ptcl
            
            # Extract source and target slices
            ptx_j   = jax.lax.dynamic_slice(ptx, (start_j,), (N_ptcl,))
            ptnx_j  = jax.lax.dynamic_slice(ptnx, (start_j,), (N_ptcl,))
            ptxp_j  = jax.lax.dynamic_slice(ptxp, (start_j,), (N_ptcl,))
            ptcur_j = jax.lax.dynamic_slice(ptcur, (start_j,), (N_ptcl,))
            ptw_j   = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
            
            tx_k    = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
            tnx_k   = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))

            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res, _, _ = StoSLP_self(ptx_j, ptnx_j, ptxp_j, ptcur_j, ptw_j, mu, jnp.array([]))
                return res

            def cross_case(_):
                # Off-diagonal block: Particle j interacting with particle k
                res, _, _ = StoSLP(tx_k, tnx_k, ptx_j, ptw_j, mu, jnp.array([]))
                return res
            
            return jax.lax.cond(j == k, self_case, cross_case, operand=None)

        # 2. Define the far-field periodic images (trlist[0] and trlist[2])
        # These are ALWAYS standard StoSLP because source and target are separated by +/- L
        def periodic_contribution():
            def single_tr_step(tr_val):
                start_j = j * N_ptcl
                start_k = k * N_ptcl
                
                # Shifted source
                ptx_j_shifted = jax.lax.dynamic_slice(ptx + tr_val, (start_j,), (N_ptcl,))
                ptw_j         = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
                
                # Static target
                tx_k  = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
                tnx_k = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))
                
                slp, _, _ = StoSLP(tx_k, tnx_k, ptx_j_shifted, ptw_j, mu, jnp.array([]))
                return slp

            # Sum over non-zero translations only
            far_trlist = jnp.array([trlist[0], trlist[2]])
            results = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results, axis=0)

        # Total = Local Interaction (0-dist) + Periodic Images (+/- L)
        return zero_translation_contribution() + periodic_contribution()

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
        return block_tensor.transpose(1, 2, 0, 3).reshape(Nptx, Nptx)
    # 3. Build the final 2Nptx x 2Nptx matrix
    u = jnp.block([[assemble_quadrant(t_xx), assemble_quadrant(t_xy)],
                   [assemble_quadrant(t_yx), assemble_quadrant(t_yy)]])
    A22_sl = u

    A22 = jnp.eye(2*len(ptx)) / 2. + A22_dl + A22_sl # CHANGED MAR 2026: avoid flipping normals for close eval. Particle initialization changed accordingly.
    A = jnp.block([[A11, A12_dl+A12_sl],
                   [A21, A22]])

    [B1,_,_] = StoSLP(sx, snx, px, pwt, mu, jnp.array([]))
    [B2,_,_] = StoSLP(ptx, ptnx, px, pwt, mu, jnp.array([]))
    B = jnp.vstack([B1, B2])

    [CLD, _, TLD] = srcsum_dl(peri_len, lx, lnx, sx, snx, sw, mu)
    [CRD, _, TRD] = srcsum_dl(-peri_len, rx, rnx, sx, snx, sw, mu) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])
    [CLDp, _, TLDp] = srcsum_dl(peri_len, lx, lnx, ptx, ptnx, ptw, mu)
    [CRDp, _, TRDp] = srcsum_dl(-peri_len, rx, rnx, ptx, ptnx, ptw, mu) 
    [CLSp, _, TLSp] = srcsum_sl(peri_len, lx, lnx, ptx, ptw, mu)
    [CRSp, _, TRSp] = srcsum_sl(-peri_len, rx, rnx, ptx, ptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    C = jnp.hstack([C1, C2])

    [QL, _, QLt] = StoSLP(lx, lnx, px, pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # --- Full ELS matrix ---
    E = jnp.block([[A, B],
                   [C, Q]])

    return E, A, B, C, Q



def ELSmatrix_ptcl_near(sx, snx, sxp, sxlo, sxhi, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptcur, ptw, ptwxp, px, pwt, lx, lnx, rx, rnx, peri_len, mu):
    """A DL formulation on wall (like before) and SL+DL on particle"""
    # trlist = peri_len*jnp.array([-1,0,1])
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    num_ptcl = len(pta)
    [Asrcsum,_,_] = srcsum_dl_self(peri_len, sx, snx, scur, sw, mu)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    # [A12_dl,_,_] = srcsum_dl(trlist, sx, snx, ptx, ptnx, ptw, mu) # from ptcl to s
    # [A12_sl,_,_] = srcsum_sl(trlist, sx, snx, ptx, ptw, mu) 
    # [A21,_,_] = srcsum_dl(trlist, ptx, ptnx, sx, snx, sw, mu) # from s to ptcl
    # [A22_dl,_,_] = srcsum_dl_self(peri_len, ptx, ptnx, ptcur, ptw, mu) # double layer self does not have single-particle assumption
    [A12_sl, _, _] = srcsum_ptcl_stosl_closeglobal(peri_len,sx,snx,ptx,ptt,pta,ptxp,ptw,jnp.eye(ptx.shape[0]),mu)
    [A12_dl, _] = srcsum_ptcl_stodl_closeglobal(peri_len,sx,snx,ptx,ptnx,pta,ptxp,ptw,ptwxp,jnp.eye(ptx.shape[0]),mu)
    [A21,_,_] = srcsum_wall_closepanel(peri_len, ptx, ptnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu)

    Nptx = len(ptx)
    N_ptcl = Nptx // num_ptcl
    # We create a helper that computes the SUM of all translations for a pair (j, k)
    def compute_pair_interaction(j, k):
        # 1. Define the 0-translation case (Middle of trlist)
        # This handles the Singularity (j==k) or standard SLP (j!=k)
        def zero_translation_contribution():
            start_j = j * N_ptcl
            start_k = k * N_ptcl
            
            # Extract source and target slices
            ptx_j   = jax.lax.dynamic_slice(ptx, (start_j,), (N_ptcl,))
            ptnx_j  = jax.lax.dynamic_slice(ptnx, (start_j,), (N_ptcl,))
            ptxp_j  = jax.lax.dynamic_slice(ptxp, (start_j,), (N_ptcl,))
            ptcur_j = jax.lax.dynamic_slice(ptcur, (start_j,), (N_ptcl,))
            ptw_j   = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
            ptwxp_j   = jax.lax.dynamic_slice(ptwxp, (start_j,), (N_ptcl,))
            ptt_j = jax.lax.dynamic_slice(ptt, (start_j,), (N_ptcl,))
            pta_j   = jax.lax.dynamic_slice(pta, (j,), (1,))
            
            tx_k    = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
            tnx_k   = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))

            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res, _, _ = StoSLP_self(ptx_j, ptnx_j, ptxp_j, ptcur_j, ptw_j, mu, jnp.array([]))
                res, _, _ = StoDLP_self(ptx_j, ptnx_j, ptcur_j, ptw_j, mu, jnp.array([]))
                return res

            def cross_case(_):
                # Off-diagonal block: Particle j interacting with particle k
                res, _, _ = stoSLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptt_j, pta_j, ptw_j, jnp.eye(ptx_j.shape[0]), mu)
                res, _, _ = stoSLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptwxp_j, jnp.eye(ptx_j.shape[0]), mu)
                return res
            
            return jax.lax.cond(j == k, self_case, cross_case, operand=None)

        # 2. Define the far-field periodic images (trlist[0] and trlist[2])
        # These are ALWAYS standard StoSLP because source and target are separated by +/- L
        def periodic_contribution():
            def single_tr_step(tr_val):
                start_j = j * N_ptcl
                start_k = k * N_ptcl
                
                # Shifted source
                ptx_j_shifted = jax.lax.dynamic_slice(ptx + tr_val, (start_j,), (N_ptcl,))
                ptw_j         = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
                
                # Static target
                tx_k  = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
                tnx_k = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))
                
                slp, _, _ = StoSLP(tx_k, tnx_k, ptx_j_shifted, ptw_j, mu, jnp.array([]))
                return slp

            # Sum over non-zero translations only
            far_trlist = jnp.array([trlist[0], trlist[2]])
            results = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results, axis=0)

        # Total = Local Interaction (0-dist) + Periodic Images (+/- L)
        return zero_translation_contribution() + periodic_contribution()

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
        return block_tensor.transpose(1, 2, 0, 3).reshape(Nptx, Nptx)
    # 3. Build the final 2Nptx x 2Nptx matrix
    u = jnp.block([[assemble_quadrant(t_xx), assemble_quadrant(t_xy)],
                   [assemble_quadrant(t_yx), assemble_quadrant(t_yy)]])
    A22 = u # now includes both SL and DL, since close-eval needs to go ptcl-by-ptcl

    A22 = jnp.eye(2*len(ptx)) / 2. + A22 
    A = jnp.block([[A11, A12_dl+A12_sl],
                   [A21, A22]])

    [B1,_,_] = StoSLP(sx, snx, px, pwt, mu, jnp.array([]))
    [B2,_,_] = StoSLP(ptx, ptnx, px, pwt, mu, jnp.array([]))
    B = jnp.vstack([B1, B2])

    [CLD, _, TLD] = srcsum_dl(peri_len, lx, lnx, sx, snx, sw, mu)
    [CRD, _, TRD] = srcsum_dl(-peri_len, rx, rnx, sx, snx, sw, mu) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])
    [CLDp, _, TLDp] = srcsum_dl(peri_len, lx, lnx, ptx, ptnx, ptw, mu)
    [CRDp, _, TRDp] = srcsum_dl(-peri_len, rx, rnx, ptx, ptnx, ptw, mu) 
    [CLSp, _, TLSp] = srcsum_sl(peri_len, lx, lnx, ptx, ptw, mu)
    [CRSp, _, TRSp] = srcsum_sl(-peri_len, rx, rnx, ptx, ptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    C = jnp.hstack([C1, C2])

    [QL, _, QLt] = StoSLP(lx, lnx, px, pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # --- Full ELS matrix ---
    E = jnp.block([[A, B],
                   [C, Q]])

    return E, A, B, C, Q


def ELSmatrix_ptcl_update(A, B, C, Q, sx, snx, sw, ptx, ptnx, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu):
    """Replace only parts of A relying on ptcl position, Bptcl, Cptcl."""
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    [A12_dl,_,_] = srcsum_dl(trlist, sx, snx, ptx, ptnx, ptw, mu) # from ptcl to s
    [A12_sl,_,_] = srcsum_sl(trlist, sx, snx, ptx, ptw, mu) 
    [A21,_,_] = srcsum_dl(trlist, ptx, ptnx, sx, snx, sw, mu) # from s to ptcl
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
                ptnx_j   = jax.lax.dynamic_slice(ptnx,   (start_j,), (N_ptcl,))
                ptw_j   = jax.lax.dynamic_slice(ptw,   (start_j,), (N_ptcl,))
                
                # And for the target inputs
                tx_k  = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
                tnx_k = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))
                
                slp, _, _ = StoSLP(tx_k, tnx_k, ptx_j, ptw_j, mu, jnp.array([]))
                dlp, _, _ = StoDLP(tx_k, tnx_k, ptx_j, ptnx_j, ptw_j, mu, jnp.array([]))
                
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
        return block_tensor.transpose(1, 2, 0, 3).reshape(Nptx, Nptx)
    # 3. Build the final 2Nptx x 2Nptx matrix
    u = jnp.block([
        [assemble_quadrant(t_xx), assemble_quadrant(t_xy)],
        [assemble_quadrant(t_yx), assemble_quadrant(t_yy)]
    ])

    A11 = A[:2*Nsx,:2*Nsx]
    Anew = jnp.block( [ [A11,A12_dl+A12_sl], [A21, u]])

    B1 = B[:2*Nsx,:]
    [B2,_,_] = StoSLP(ptx, ptnx, px, pwt, mu, jnp.array([]))
    Bnew = jnp.vstack([B1,B2])

    C1 = C[:,:2*Nsx]
    [CLDp, _, TLDp] = srcsum_dl(peri_len, lx, lnx, ptx, ptnx, ptw, mu)
    [CRDp, _, TRDp] = srcsum_dl(-peri_len, rx, rnx, ptx, ptnx, ptw, mu) 
    [CLSp, _, TLSp] = srcsum_sl(peri_len, lx, lnx, ptx, ptw, mu)
    [CRSp, _, TRSp] = srcsum_sl(-peri_len, rx, rnx, ptx, ptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    Cnew = jnp.hstack([C1,C2])

    # --- Full ELS matrix ---
    E = jnp.block([[Anew, Bnew],
                   [Cnew, Q]])

    return E, Anew, Bnew, Cnew, Q

@jit
def ELSmatrix_stokeslet_f0(sx, snx, scur, sw, x_stokeslet, w_stokeslet, px, pwt, lx, lnx, rx, rnx, peri_len, mu):
    """ Stokeslet E matrix with constraint that sum of forces is 0."""
    
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)

    [Asrcsum,_,_] = srcsum_dl_self(peri_len, sx, snx, scur, sw, mu)
    A = -jnp.eye(2*Nsx) / 2 + Asrcsum

    x_stokeslet_near = x_stokeslet + trlist[0] # left copy first
    [A_ptsrc_near,_,_] = StoSLP(sx, snx, x_stokeslet_near, w_stokeslet, mu, jnp.array([]))
    [CR_ptsrc_near,_,TR_ptsrc_near] = StoSLP(rx, rnx, x_stokeslet_near, w_stokeslet, mu, jnp.array([]))
    for i in range(1,len(trlist)):
        x_stokeslet_near = x_stokeslet + trlist[i]
        [A_ptsrc_temp,_,_] = StoSLP(sx, snx, x_stokeslet_near, w_stokeslet, mu, jnp.array([]))
        if i==2: # right copy
            [CL_ptsrc_near,_,TL_ptsrc_near] = StoSLP(lx, lnx, x_stokeslet_near, w_stokeslet, mu, jnp.array([]))
        A_ptsrc_near += A_ptsrc_temp 
    C_ptsrc_near = jnp.vstack([CR_ptsrc_near-CL_ptsrc_near, TR_ptsrc_near-TL_ptsrc_near])

    [B,_,_] = StoSLP(sx, snx, px, pwt, mu, jnp.array([]))

    [CLD, _, TLD] = srcsum_dl(peri_len, lx, lnx, sx, snx, sw, mu)
    [CRD, _, TRD] = srcsum_sl(-peri_len, rx, rnx, sx, sw, mu) 
    C = jnp.vstack([CRD - CLD, TRD - TLD])       

    [QL, _, QLt] = StoSLP(lx, lnx, px, pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, pwt, mu, jnp.array([]))
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


def ELSmatrix_rbm(sx, snx, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu):
    """
    A DL formulation on wall (like before) and SL+DL on particle
    Include RBM with unknown U and Omega; implement net force and net torque zero conditions.
    """
    trlist = jnp.array([-peri_len, 0, peri_len])

    Nsx = len(sx)
    # TODO: include near-eval!
    [Asrcsum,_,_] = srcsum_dl_self(peri_len, sx, snx, scur, sw, mu)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    [A12_dl,_,_] = srcsum_dl(trlist, sx, snx, ptx, ptnx, ptw, mu) # from ptcl to s
    [A12_sl,_,_] = srcsum_sl(trlist, sx, snx, ptx, ptw, mu) 
    [A21,_,A21_T] = srcsum_dl(trlist, ptx, ptnx, sx, snx, sw, mu) # from s to ptcl
    [A22_dl,_,_] = srcsum_dl_self(peri_len, ptx, ptnx, ptcur, ptw, mu) # double layer self does not have single-particle assumption
    # Ptcl near nbr contribution to DL traction
    ptx_l = ptx + trlist[0]
    [_,_,A22_dlT_1] = StoDLP(ptx,ptnx,ptx_l,ptnx,ptw,mu,jnp.array([]))
    ptx_r = ptx + trlist[2]
    [_,_,A22_dlT_2] = StoDLP(ptx,ptnx,ptx_r,ptnx,ptw,mu,jnp.array([]))

    Nptx = len(ptx)
    N_ptcl_double = Nptx/num_ptcl
    N_ptcl = int(N_ptcl_double)

    # We create a helper that computes the SUM of all translations for a pair (j, k)
    def compute_pair_interaction(j, k):
        # 1. Define the 0-translation case (Middle of trlist)
        # This handles the Singularity (j==k) or standard SLP (j!=k)
        def zero_translation_contribution():
            start_j = j * N_ptcl
            start_k = k * N_ptcl
            
            # Extract source and target slices
            ptx_j   = jax.lax.dynamic_slice(ptx, (start_j,), (N_ptcl,))
            ptnx_j  = jax.lax.dynamic_slice(ptnx, (start_j,), (N_ptcl,))
            ptxp_j  = jax.lax.dynamic_slice(ptxp, (start_j,), (N_ptcl,))
            ptcur_j = jax.lax.dynamic_slice(ptcur, (start_j,), (N_ptcl,))
            ptw_j   = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
            
            tx_k    = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
            tnx_k   = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))

            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res, _, res_T = StoSLP_self(ptx_j, ptnx_j, ptxp_j, ptcur_j, ptw_j, mu, jnp.array([]))
                return res, res_T

            def cross_case(_):
                # Off-diagonal block: Particle j interacting with particle k
                res, _, res_T = StoSLP(tx_k, tnx_k, ptx_j, ptw_j, mu, jnp.array([]))
                return res, res_T
            
            return jax.lax.cond(j == k, self_case, cross_case, operand=None)

        # 2. Define the far-field periodic images (trlist[0] and trlist[2])
        # These are ALWAYS standard StoSLP because source and target are separated by +/- L
        def periodic_contribution():
            def single_tr_step(tr_val):
                start_j = j * N_ptcl
                start_k = k * N_ptcl
                
                # Shifted source
                ptx_j_shifted = jax.lax.dynamic_slice(ptx + tr_val, (start_j,), (N_ptcl,))
                ptw_j         = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
                
                # Static target
                tx_k  = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
                tnx_k = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))
                
                slp, _, slpT = StoSLP(tx_k, tnx_k, ptx_j_shifted, ptw_j, mu, jnp.array([]))
                return slp, slpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([trlist[0], trlist[2]])
            results_u,results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        # Total = Local Interaction (0-dist) + Periodic Images (+/- L)
        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p

    # We use vmap to generate the (num_ptcl, num_ptcl, 2*N_ptcl, 2*N_ptcl) tensor
    indices = jnp.arange(num_ptcl)
    grid_ttu, grid_ttT = jax.vmap(lambda j: jax.vmap(lambda k: compute_pair_interaction(j, k))(indices))(indices)
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
    u = jnp.block([[assemble_quadrant(t_xx), assemble_quadrant(t_xy)],
                   [assemble_quadrant(t_yx), assemble_quadrant(t_yy)]])
    # Same for T
    t_xx_T = grid_ttT[:, :, :N_ptcl, :N_ptcl]
    t_xy_T = grid_ttT[:, :, :N_ptcl, N_ptcl:]
    t_yx_T = grid_ttT[:, :, N_ptcl:, :N_ptcl]
    t_yy_T = grid_ttT[:, :, N_ptcl:, N_ptcl:]
    T = jnp.block([[assemble_quadrant(t_xx_T), assemble_quadrant(t_xy_T)],
                   [assemble_quadrant(t_yx_T), assemble_quadrant(t_yy_T)]])
    A22_sl = u
    A22_slT = T

    A22 = jnp.eye(2*len(ptx)) / 2. + A22_dl + A22_sl # CHANGED MAR 2026: avoid flipping normals for close eval. Particle initialization changed accordingly.
    A = jnp.block([[A11, A12_dl+A12_sl],
                   [A21, A22]])

    [B1,_,_] = StoSLP(sx, snx, px, pwt, mu, jnp.array([]))
    [B2,_,B2_T] = StoSLP(ptx, ptnx, px, pwt, mu, jnp.array([]))
    B = jnp.vstack([B1, B2])

    [CLD, _, TLD] = srcsum_dl(peri_len, lx, lnx, sx, snx, sw, mu)
    [CRD, _, TRD] = srcsum_dl(-peri_len, rx, rnx, sx, snx, sw, mu) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])
    [CLDp, _, TLDp] = srcsum_dl(peri_len, lx, lnx, ptx, ptnx, ptw, mu)
    [CRDp, _, TRDp] = srcsum_dl(-peri_len, rx, rnx, ptx, ptnx, ptw, mu) 
    [CLSp, _, TLSp] = srcsum_sl(peri_len, lx, lnx, ptx, ptw, mu)
    [CRSp, _, TRSp] = srcsum_sl(-peri_len, rx, rnx, ptx, ptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    C = jnp.hstack([C1, C2])

    [QL, _, QLt] = StoSLP(lx, lnx, px, pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # Rigid body motion adding to boundary conditions
    total_rows = 2*Nsx + 2*Nptx
    total_cols = total_rows + 3 * num_ptcl
    nodes_per_ptcl = Nptx // num_ptcl
    bc_sqr = jnp.zeros((2*Nptx,3*num_ptcl))
    ptx_2d = ptx.reshape(num_ptcl,nodes_per_ptcl) # ASSUMES same discr. on each ptcl
    # X - Xc
    Xc = jnp.mean(ptx_2d, axis=1, keepdims=1)
    XmXc = ptx_2d - Xc
    XmXc_flat = XmXc.reshape(-1)
    
    row_indices = jnp.arange(Nptx)
    col_ux = jnp.repeat(jnp.arange(0, num_ptcl * 2, 2), nodes_per_ptcl)
    col_uy = jnp.repeat(jnp.arange(1, num_ptcl * 2, 2), nodes_per_ptcl)
    bc_sqr = bc_sqr.at[row_indices, col_ux].set(-1.0)
    bc_sqr = bc_sqr.at[row_indices + Nptx, col_uy].set(-1.0)
    col_omega = jnp.repeat(jnp.arange(num_ptcl * 2, 3 * num_ptcl), nodes_per_ptcl)
    bc_sqr = bc_sqr.at[row_indices, col_omega].set(-jnp.imag(XmXc_flat))
    bc_sqr = bc_sqr.at[row_indices + Nptx, col_omega].set(jnp.real(XmXc_flat))

    bc_gamma_mat = jnp.zeros((total_rows, total_cols))
    bc_gamma_mat = bc_gamma_mat.at[:, :total_rows].set(A)
    bc_gamma_mat = bc_gamma_mat.at[2*Nsx:, total_rows:].set(bc_sqr)

    # Net force zero and net torque zero conditions
    T3 = A22_slT + A22_dlT_1 + A22_dlT_2
    fMat_ptcl = - ( -jnp.eye(2*Nptx)/2. + T3)
    fMat_wall = - A21_T
    fMat_prx = - B2_T
    
    ptcl_cell_w = ptw.reshape(num_ptcl, nodes_per_ptcl) 
    WF = jax.scipy.linalg.block_diag(*[w[None, :] for w in ptcl_cell_w])
    tx_vals = -ptcl_cell_w * jnp.imag(XmXc)
    ty_vals =  ptcl_cell_w * jnp.real(XmXc)
    WTx = jax.scipy.linalg.block_diag(*[t[None, :] for t in tx_vals])
    WTy = jax.scipy.linalg.block_diag(*[t[None, :] for t in ty_vals])
    # MATLAB: [WFx, 0; 0, WFy]
    intF_op = jax.scipy.linalg.block_diag(WF, WF)
    # MATLAB: [WTx, WTy]
    intT_op = jnp.concatenate([WTx, WTy], axis=1)
    mid_zeros = jnp.zeros((2 * Nptx, 3 * num_ptcl))
    big_mat = jnp.concatenate([fMat_wall, fMat_ptcl, mid_zeros, fMat_prx], axis=1)

    intF = intF_op @ big_mat
    intT = intT_op @ big_mat

    # --- Full ELS matrix ---
    E = jnp.block([[bc_gamma_mat, B],
                   [intF],
                   [intT],
                   [C, jnp.zeros((C.shape[0],3*num_ptcl)), Q]])

    return E, bc_gamma_mat, B, intF,intT, C, Q



@jit
def ELSmatrix_rbm_near(sx, snx, sxp, sxlo, sxhi, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptcur, ptw, ptwxp, px, pwt, lx, lnx, rx, rnx, peri_len, mu):
    """
    A DL formulation on wall (like before) and SL+DL on particle
    Include RBM with unknown U and Omega; implement net force and net torque zero conditions.
    """
    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    num_ptcl = len(pta)
    [Asrcsum,_,_] = srcsum_dl_self(peri_len, sx, snx, scur, sw, mu)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    # [A12_dl,_,_] = srcsum_dl(trlist, sx, snx, ptx, ptnx, ptw, mu) # from ptcl to s
    # [A12_sl,_,_] = srcsum_sl(trlist, sx, snx, ptx, ptw, mu) 
    # [A21,_,A21_T] = srcsum_dl(trlist, ptx, ptnx, sx, snx, sw, mu) # from s to ptcl
    # [A22_dl,_,_] = srcsum_dl_self(peri_len, ptx, ptnx, ptcur, ptw, mu) # double layer self does not have single-particle assumption
    # # Ptcl near nbr contribution to DL traction
    # ptx_l = ptx + trlist[0]
    # [_,_,A22_dlT_1] = StoDLP(ptx,ptnx,ptx_l,ptnx,ptw,mu,jnp.array([]))
    # ptx_r = ptx + trlist[2]
    # [_,_,A22_dlT_2] = StoDLP(ptx,ptnx,ptx_r,ptnx,ptw,mu,jnp.array([]))
    [A12_sl, _, _] = srcsum_ptcl_stosl_closeglobal(peri_len,sx,snx,ptx,ptt,pta,ptxp,ptw,jnp.eye(2*ptx.shape[0]),mu)
    [A12_dl, _] = srcsum_ptcl_stodl_closeglobal(peri_len,sx,snx,ptx,ptnx,pta,ptxp,ptw,ptwxp,jnp.eye(2*ptx.shape[0]),mu)
    [A21,_,A21_T] = srcsum_wall_closepanel(peri_len, ptx, ptnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu)


    Nptx = len(ptx)
    N_ptcl_double = Nptx/num_ptcl
    N_ptcl = int(N_ptcl_double)

    # We create a helper that computes the SUM of all translations for a pair (j, k)
    def compute_pair_interaction(j, k):
        # 1. Define the 0-translation case (Middle of trlist)
        # This handles the Singularity (j==k) or standard SLP (j!=k)
        def zero_translation_contribution():
            start_j = j * N_ptcl
            start_k = k * N_ptcl
            
            # Extract source and target slices
            ptx_j   = jax.lax.dynamic_slice(ptx, (start_j,), (N_ptcl,))
            ptnx_j  = jax.lax.dynamic_slice(ptnx, (start_j,), (N_ptcl,))
            ptxp_j  = jax.lax.dynamic_slice(ptxp, (start_j,), (N_ptcl,))
            ptcur_j = jax.lax.dynamic_slice(ptcur, (start_j,), (N_ptcl,))
            ptw_j   = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
            ptwxp_j   = jax.lax.dynamic_slice(ptwxp, (start_j,), (N_ptcl,))
            ptt_j = jax.lax.dynamic_slice(ptt, (start_j,), (N_ptcl,))
            pta_j   = jax.lax.dynamic_slice(pta, (j,), (1,))
            
            tx_k    = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
            tnx_k   = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))

            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res_sl, _, res_slT = StoSLP_self(ptx_j, ptnx_j, ptxp_j, ptcur_j, ptw_j, mu, jnp.array([]))
                res_dl, _, _ = StoDLP_self(ptx_j, ptnx_j, ptcur_j, ptw_j, mu, jnp.array([]))
                return res_sl+res_dl, res_slT # NOTE: self-to-self DL traction contributes 0, self-eval doesn't handle singularity.

            def cross_case(_):
                # Off-diagonal block: Particle j interacting with particle k
                res_sl, _, res_slT = stoSLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptt_j, pta_j, ptw_j, jnp.eye(ptx_j.shape[0]), mu)
                res_dl, _, res_dlT = stoDLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptwxp_j, jnp.eye(ptx_j.shape[0]), mu)
                return res_sl+res_dl, res_slT+res_dlT
            
            return jax.lax.cond(j == k, self_case, cross_case, operand=None)

        # 2. Define the far-field periodic images (trlist[0] and trlist[2])
        # These are ALWAYS standard StoSLP because source and target are separated by +/- L
        def periodic_contribution():
            def single_tr_step(tr_val):
                start_j = j * N_ptcl
                start_k = k * N_ptcl
                
                # Shifted source
                ptx_j_shifted = jax.lax.dynamic_slice(ptx + tr_val, (start_j,), (N_ptcl,))
                ptnx_j         = jax.lax.dynamic_slice(ptnx, (start_j,), (N_ptcl,))
                ptw_j         = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
                
                # Static target
                tx_k  = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
                tnx_k = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))
                
                slp, _, slpT = StoSLP(tx_k, tnx_k, ptx_j_shifted, ptw_j, mu, jnp.array([]))
                dlp, _, dlpT = StoDLP(tx_k, tnx_k, ptx_j_shifted, ptnx_j, ptw_j, mu, jnp.array([]))
                return slp+dlp, slpT+dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([trlist[0], trlist[2]])
            results_u,results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        # Total = Local Interaction (0-dist) + Periodic Images (+/- L)
        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p

    # We use vmap to generate the (num_ptcl, num_ptcl, 2*N_ptcl, 2*N_ptcl) tensor
    indices = jnp.arange(num_ptcl)
    grid_ttu, grid_ttT = jax.vmap(lambda j: jax.vmap(lambda k: compute_pair_interaction(j, k))(indices))(indices)
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
    u = jnp.block([[assemble_quadrant(t_xx), assemble_quadrant(t_xy)],
                   [assemble_quadrant(t_yx), assemble_quadrant(t_yy)]])
    # Same for T
    t_xx_T = grid_ttT[:, :, :N_ptcl, :N_ptcl]
    t_xy_T = grid_ttT[:, :, :N_ptcl, N_ptcl:]
    t_yx_T = grid_ttT[:, :, N_ptcl:, :N_ptcl]
    t_yy_T = grid_ttT[:, :, N_ptcl:, N_ptcl:]
    T = jnp.block([[assemble_quadrant(t_xx_T), assemble_quadrant(t_xy_T)],
                   [assemble_quadrant(t_yx_T), assemble_quadrant(t_yy_T)]])
    A22 = u
    A22_T = T

    A22 = jnp.eye(2*len(ptx)) / 2. + A22
    A = jnp.block([[A11, A12_dl+A12_sl],
                   [A21, A22]])

    [B1,_,_] = StoSLP(sx, snx, px, pwt, mu, jnp.array([]))
    [B2,_,B2_T] = StoSLP(ptx, ptnx, px, pwt, mu, jnp.array([]))
    B = jnp.vstack([B1, B2])

    [CLD, _, TLD] = srcsum_dl(peri_len, lx, lnx, sx, snx, sw, mu)
    [CRD, _, TRD] = srcsum_dl(-peri_len, rx, rnx, sx, snx, sw, mu) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])
    [CLDp, _, TLDp] = srcsum_dl(peri_len, lx, lnx, ptx, ptnx, ptw, mu)
    [CRDp, _, TRDp] = srcsum_dl(-peri_len, rx, rnx, ptx, ptnx, ptw, mu) 
    [CLSp, _, TLSp] = srcsum_sl(peri_len, lx, lnx, ptx, ptw, mu)
    [CRSp, _, TRSp] = srcsum_sl(-peri_len, rx, rnx, ptx, ptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    C = jnp.hstack([C1, C2])

    [QL, _, QLt] = StoSLP(lx, lnx, px, pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # Rigid body motion adding to boundary conditions
    total_rows = 2*Nsx + 2*Nptx
    total_cols = total_rows + 3 * num_ptcl
    nodes_per_ptcl = Nptx // num_ptcl
    bc_sqr = jnp.zeros((2*Nptx,3*num_ptcl))
    ptx_2d = ptx.reshape(num_ptcl,nodes_per_ptcl) # ASSUMES same discr. on each ptcl
    # X - Xc
    Xc = jnp.mean(ptx_2d, axis=1, keepdims=1)
    XmXc = ptx_2d - Xc
    XmXc_flat = XmXc.reshape(-1)
    
    row_indices = jnp.arange(Nptx)
    col_ux = jnp.repeat(jnp.arange(0, num_ptcl * 2, 2), nodes_per_ptcl)
    col_uy = jnp.repeat(jnp.arange(1, num_ptcl * 2, 2), nodes_per_ptcl)
    bc_sqr = bc_sqr.at[row_indices, col_ux].set(-1.0)
    bc_sqr = bc_sqr.at[row_indices + Nptx, col_uy].set(-1.0)
    col_omega = jnp.repeat(jnp.arange(num_ptcl * 2, 3 * num_ptcl), nodes_per_ptcl)
    bc_sqr = bc_sqr.at[row_indices, col_omega].set(-jnp.imag(XmXc_flat))
    bc_sqr = bc_sqr.at[row_indices + Nptx, col_omega].set(jnp.real(XmXc_flat))

    bc_gamma_mat = jnp.zeros((total_rows, total_cols))
    bc_gamma_mat = bc_gamma_mat.at[:, :total_rows].set(A)
    bc_gamma_mat = bc_gamma_mat.at[2*Nsx:, total_rows:].set(bc_sqr)

    # Net force zero and net torque zero conditions
    T3 = A22_T
    fMat_ptcl = - ( -jnp.eye(2*Nptx)/2. + T3)
    fMat_wall = - A21_T
    fMat_prx = - B2_T
    
    ptcl_cell_w = ptw.reshape(num_ptcl, nodes_per_ptcl) 
    WF = jax.scipy.linalg.block_diag(*[w[None, :] for w in ptcl_cell_w])
    tx_vals = -ptcl_cell_w * jnp.imag(XmXc)
    ty_vals =  ptcl_cell_w * jnp.real(XmXc)
    WTx = jax.scipy.linalg.block_diag(*[t[None, :] for t in tx_vals])
    WTy = jax.scipy.linalg.block_diag(*[t[None, :] for t in ty_vals])
    # MATLAB: [WFx, 0; 0, WFy]
    intF_op = jax.scipy.linalg.block_diag(WF, WF)
    # MATLAB: [WTx, WTy]
    intT_op = jnp.concatenate([WTx, WTy], axis=1)
    mid_zeros = jnp.zeros((2 * Nptx, 3 * num_ptcl))
    big_mat = jnp.concatenate([fMat_wall, fMat_ptcl, mid_zeros, fMat_prx], axis=1)

    intF = intF_op @ big_mat
    intT = intT_op @ big_mat

    # --- Full ELS matrix ---
    E = jnp.block([[bc_gamma_mat, B],
                   [intF],
                   [intT],
                   [C, jnp.zeros((C.shape[0],3*num_ptcl)), Q]])

    return E, bc_gamma_mat, B, intF,intT, C, Q


def get_reorder_indices(N_obs, N_ptcl):
    N_tot = N_obs + N_ptcl
    # Kernel order: [x_obs, x_ptcl, y_obs, y_ptcl]
    # Desired order: [x_obs, y_obs, x_ptcl, y_ptcl]
    
    new_indices = jnp.concatenate([
        jnp.arange(N_obs),                   # x_obs
        jnp.arange(N_tot, N_tot + N_obs),     # y_obs
        jnp.arange(N_obs, N_tot),             # x_ptcl
        jnp.arange(N_tot + N_obs, 2 * N_tot)  # y_ptcl
    ])
    return new_indices

# A DL formulation on wall (like before) and SL+DL on particle
# Include RBM with unknown U and Omega; implement net force and net torque zero conditions.
# Also includes obstacle particles.
def ELSmatrix_rbm_obs(sx, snx, sxp, sxlo, sxhi, scur, sw, obsx, obsnx, obst, obsa, obsxp, obscur, obsw, obswxp, num_obs, ptx, ptnx, ptt, pta, ptxp, ptcur, ptw, ptwxp, num_ptcl, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu):

    trlist = jnp.array([-peri_len, 0, peri_len])
    Nsx = len(sx)
    # n_tr = trlist.size
    Nptx = len(ptx)
    N_ptcl = Nptx // num_ptcl
    Nobsx = len(obsx)
    N_obs = Nobsx // num_obs

    jax.debug.print("Nobsx = {a}, N_obs = {b}, Nptx = {c}, N_ptcl = {d}", a=Nobsx, b=N_obs, c=Nptx, d = N_ptcl)

    # [Asrcsum,_,_] = srcsum_self(StoDLP_self, StoDLP, trlist, sx, snx, sxp, scur, sw, mu)
    [Asrcsum,_,_] = srcsum_dl_self(peri_len, sx, snx, scur, sw, mu)
    A11 = -jnp.eye(2*Nsx) / 2. + Asrcsum
    # Combine obs and ptc for some calculations that 
    # don't distinguish between rigid body and passive particles
    # a-pt-_: "all-particle-..."
    aptx = jnp.concatenate([obsx,ptx])
    aptnx = jnp.concatenate([obsnx,ptnx])
    aptw = jnp.concatenate([obsw,ptw])
    # For reordering in this case
    idx = get_reorder_indices(Nobsx, Nptx)
    # ----------------------------------------------
    [A12_sl_p, _, _] = srcsum_ptcl_stosl_closeglobal(peri_len,sx,snx,ptx,ptt,pta,ptxp,ptw,jnp.eye(2*ptx.shape[0]),mu) # closeglobal assumes all items in ptx, etc have same N_nodes_per_obj.
    [A12_dl_p, _, _] = srcsum_ptcl_stodl_closeglobal(peri_len,sx,snx,ptx,ptnx,pta,ptxp,ptw,ptwxp,jnp.eye(2*ptx.shape[0]),mu)
    [A12_sl_o, _, _] = srcsum_ptcl_stosl_closeglobal(peri_len,sx,snx,obsx,obst,obsa,obsxp,obsw,jnp.eye(2*obsx.shape[0]),mu)
    [A12_dl_o, _, _] = srcsum_ptcl_stodl_closeglobal(peri_len,sx,snx,obsx,obsnx,obsa,obsxp,obsw,obswxp,jnp.eye(2*obsx.shape[0]),mu)
    A12_p = A12_dl_p + A12_sl_p
    A12_o = A12_dl_o + A12_sl_o
    A12 = jnp.hstack([A12_o, A12_p])
    [A21,_,A21_T] = srcsum_wall_closepanel(peri_len, ptx, ptnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu)
    [A21_o,_,_] = srcsum_wall_closepanel(peri_len, obsx, obsnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu)
    A21 = jnp.vstack([A21_o, A21])

    Nptx = len(ptx)
    N_ptcl_double = Nptx/num_ptcl
    N_ptcl = int(N_ptcl_double)

    # FROM ptcl_j (with near) to ptcl_k 
    def compute_p2p(j, k):
        def zero_translation_contribution():
            start_j = j * N_ptcl
            start_k = k * N_ptcl
            
            # Extract source and target slices
            ptx_j   = jax.lax.dynamic_slice(ptx, (start_j,), (N_ptcl,))
            ptnx_j  = jax.lax.dynamic_slice(ptnx, (start_j,), (N_ptcl,))
            ptxp_j  = jax.lax.dynamic_slice(ptxp, (start_j,), (N_ptcl,))
            ptcur_j = jax.lax.dynamic_slice(ptcur, (start_j,), (N_ptcl,))
            ptw_j   = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
            ptwxp_j   = jax.lax.dynamic_slice(ptwxp, (start_j,), (N_ptcl,))
            ptt_j = jax.lax.dynamic_slice(ptt, (start_j,), (N_ptcl,))
            pta_j   = jax.lax.dynamic_slice(pta, (j,), (1,))
            
            tx_k    = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
            tnx_k   = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))

            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res_sl, _, res_slT = StoSLP_self(ptx_j, ptnx_j, ptxp_j, ptcur_j, ptw_j, mu, jnp.array([]))
                res_dl, _, _ = StoDLP_self(ptx_j, ptnx_j, ptcur_j, ptw_j, mu, jnp.array([]))
                return res_sl+res_dl, res_slT # NOTE: self-to-self DL traction contributes 0, self-eval doesn't handle singularity.

            def cross_case(_):
                # Off-diagonal block: Particle j interacting with particle k
                res_sl, _, res_slT = stoSLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptt_j, pta_j, ptw_j, jnp.eye(2*ptx_j.shape[0]), mu)
                res_dl, _, res_dlT = stoDLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptwxp_j, jnp.eye(2*ptx_j.shape[0]), mu)
                return res_sl+res_dl, res_slT+res_dlT
            
            return jax.lax.cond(j == k, self_case, cross_case, operand=None)

        def periodic_contribution():
            def single_tr_step(tr_val):
                start_j = j * N_ptcl
                start_k = k * N_ptcl
                
                # Shifted source
                ptx_j_shifted = jax.lax.dynamic_slice(ptx + tr_val, (start_j,), (N_ptcl,))
                ptnx_j         = jax.lax.dynamic_slice(ptnx, (start_j,), (N_ptcl,))
                ptw_j         = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
                
                # Static target
                tx_k  = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
                tnx_k = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))
                
                slp, _, slpT = StoSLP(tx_k, tnx_k, ptx_j_shifted, ptw_j, mu, jnp.array([]))
                dlp, _, dlpT = StoDLP(tx_k, tnx_k, ptx_j_shifted, ptnx_j, ptw_j, mu, jnp.array([]))
                return slp+dlp, slpT+dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([trlist[0], trlist[2]])
            results_u,results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p
    
    # FROM ptcl_j (with near) to obs_k (no self)
    def compute_p2o(j, k):
        def zero_translation_contribution():
            start_j = j * N_ptcl
            start_k = k * N_obs
            
            # Extract source and target slices
            ptx_j   = jax.lax.dynamic_slice(ptx, (start_j,), (N_ptcl,))
            ptxp_j  = jax.lax.dynamic_slice(ptxp, (start_j,), (N_ptcl,))
            ptw_j   = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
            ptwxp_j   = jax.lax.dynamic_slice(ptwxp, (start_j,), (N_ptcl,))
            ptt_j = jax.lax.dynamic_slice(ptt, (start_j,), (N_ptcl,))
            pta_j   = jax.lax.dynamic_slice(pta, (j,), (1,))
            
            tx_k    = jax.lax.dynamic_slice(obsx,  (start_k,), (N_obs,))
            tnx_k   = jax.lax.dynamic_slice(obsnx, (start_k,), (N_obs,))

            res_sl, _, res_slT = stoSLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptt_j, pta_j, ptw_j, jnp.eye(2*ptx_j.shape[0]), mu)
            res_dl, _, res_dlT = stoDLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptwxp_j, jnp.eye(2*ptx_j.shape[0]), mu)
            return res_sl+res_dl, res_slT+res_dlT

        def periodic_contribution():
            def single_tr_step(tr_val):
                start_j = j * N_ptcl
                start_k = k * N_obs
                
                # Shifted source
                ptx_j_shifted = jax.lax.dynamic_slice(ptx + tr_val, (start_j,), (N_ptcl,))
                ptnx_j         = jax.lax.dynamic_slice(ptnx, (start_j,), (N_ptcl,))
                ptw_j         = jax.lax.dynamic_slice(ptw, (start_j,), (N_ptcl,))
                
                # Static target
                tx_k  = jax.lax.dynamic_slice(obsx,  (start_k,), (N_obs,))
                tnx_k = jax.lax.dynamic_slice(obsnx, (start_k,), (N_obs,))
                
                slp, _, slpT = StoSLP(tx_k, tnx_k, ptx_j_shifted, ptw_j, mu, jnp.array([]))
                dlp, _, dlpT = StoDLP(tx_k, tnx_k, ptx_j_shifted, ptnx_j, ptw_j, mu, jnp.array([]))
                return slp+dlp, slpT+dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([trlist[0], trlist[2]])
            results_u,results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p
    
    # FROM obs_j (with near) to ptcl_k  (no self)
    def compute_o2p(j, k):
        def zero_translation_contribution():
            start_j = j * N_obs
            start_k = k * N_ptcl
            
            # Extract source and target slices
            ptx_j   = jax.lax.dynamic_slice(obsx, (start_j,), (N_obs,))
            ptxp_j  = jax.lax.dynamic_slice(obsxp, (start_j,), (N_obs,))
            ptw_j   = jax.lax.dynamic_slice(obsw, (start_j,), (N_obs,))
            ptwxp_j   = jax.lax.dynamic_slice(obswxp, (start_j,), (N_obs,))
            ptt_j = jax.lax.dynamic_slice(obst, (start_j,), (N_obs,))
            pta_j   = jax.lax.dynamic_slice(obsa, (j,), (1,))
            
            tx_k    = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
            tnx_k   = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))

            res_sl, _, res_slT = stoSLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptt_j, pta_j, ptw_j, jnp.eye(2*ptx_j.shape[0]), mu)
            res_dl, _, res_dlT = stoDLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptwxp_j, jnp.eye(2*ptx_j.shape[0]), mu)
            return res_sl+res_dl, res_slT+res_dlT

        def periodic_contribution():
            def single_tr_step(tr_val):
                start_j = j * N_obs
                start_k = k * N_ptcl
                
                # Shifted source
                ptx_j_shifted = jax.lax.dynamic_slice(obsx + tr_val, (start_j,), (N_obs,))
                ptnx_j         = jax.lax.dynamic_slice(obsnx, (start_j,), (N_obs,))
                ptw_j         = jax.lax.dynamic_slice(obsw, (start_j,), (N_obs,))
                
                # Static target
                tx_k  = jax.lax.dynamic_slice(ptx,  (start_k,), (N_ptcl,))
                tnx_k = jax.lax.dynamic_slice(ptnx, (start_k,), (N_ptcl,))
                
                slp, _, slpT = StoSLP(tx_k, tnx_k, ptx_j_shifted, ptw_j, mu, jnp.array([]))
                dlp, _, dlpT = StoDLP(tx_k, tnx_k, ptx_j_shifted, ptnx_j, ptw_j, mu, jnp.array([]))
                return slp+dlp, slpT+dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([trlist[0], trlist[2]])
            results_u,results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p
    
    # FROM ptcl_j (with near) to ptcl_k 
    def compute_o2o(j, k):
        def zero_translation_contribution():
            start_j = j * N_obs
            start_k = k * N_obs
            
            # Extract source and target slices
            ptx_j   = jax.lax.dynamic_slice(obsx, (start_j,), (N_obs,))
            ptnx_j  = jax.lax.dynamic_slice(obsnx, (start_j,), (N_obs,))
            ptxp_j  = jax.lax.dynamic_slice(obsxp, (start_j,), (N_obs,))
            ptcur_j = jax.lax.dynamic_slice(obscur, (start_j,), (N_obs,))
            ptw_j   = jax.lax.dynamic_slice(obsw, (start_j,), (N_obs,))
            ptwxp_j   = jax.lax.dynamic_slice(obswxp, (start_j,), (N_obs,))
            ptt_j = jax.lax.dynamic_slice(obst, (start_j,), (N_obs,))
            pta_j   = jax.lax.dynamic_slice(obsa, (j,), (1,))
            
            tx_k    = jax.lax.dynamic_slice(obsx,  (start_k,), (N_obs,))
            tnx_k   = jax.lax.dynamic_slice(obsnx, (start_k,), (N_obs,))

            def self_case(_):
                # Diagonal block: Particle interacting with itself (No jump)
                res_sl, _, res_slT = StoSLP_self(ptx_j, ptnx_j, ptxp_j, ptcur_j, ptw_j, mu, jnp.array([]))
                res_dl, _, _ = StoDLP_self(ptx_j, ptnx_j, ptcur_j, ptw_j, mu, jnp.array([]))
                return res_sl+res_dl, res_slT 

            def cross_case(_):
                # Off-diagonal block: Particle j interacting with particle k
                res_sl, _, res_slT = stoSLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptt_j, pta_j, ptw_j, jnp.eye(2*ptx_j.shape[0]), mu)
                res_dl, _, res_dlT = stoDLP_closeglobal(tx_k, tnx_k, ptx_j, ptxp_j, ptwxp_j, jnp.eye(2*ptx_j.shape[0]), mu)
                return res_sl+res_dl, res_slT+res_dlT
            
            return jax.lax.cond(j == k, self_case, cross_case, operand=None)

        def periodic_contribution():
            def single_tr_step(tr_val):
                start_j = j * N_obs
                start_k = k * N_obs
                
                # Shifted source
                ptx_j_shifted = jax.lax.dynamic_slice(obsx + tr_val, (start_j,), (N_obs,))
                ptnx_j         = jax.lax.dynamic_slice(obsnx, (start_j,), (N_obs,))
                ptw_j         = jax.lax.dynamic_slice(obsw, (start_j,), (N_obs,))
                
                # Static target
                tx_k  = jax.lax.dynamic_slice(obsx,  (start_k,), (N_obs,))
                tnx_k = jax.lax.dynamic_slice(obsnx, (start_k,), (N_obs,))
                
                slp, _, slpT = StoSLP(tx_k, tnx_k, ptx_j_shifted, ptw_j, mu, jnp.array([]))
                dlp, _, dlpT = StoDLP(tx_k, tnx_k, ptx_j_shifted, ptnx_j, ptw_j, mu, jnp.array([]))
                return slp+dlp, slpT+dlpT

            # Sum over non-zero translations only
            far_trlist = jnp.array([trlist[0], trlist[2]])
            results_u,results_T = jax.vmap(single_tr_step)(far_trlist)
            return jnp.sum(results_u, axis=0),jnp.sum(results_T, axis=0)

        u_0, T_0 = zero_translation_contribution()
        u_p, T_p = periodic_contribution()
        return u_0 + u_p, T_0 + T_p

    # We use vmap to generate the (num_ptcl, num_ptcl, 2*N_ptcl, 2*N_ptcl) tensor
    indices_ptcl = jnp.arange(num_ptcl)
    indices_obs = jnp.arange(num_obs)
    grid_up2p, grid_Tp2p = jax.vmap(lambda j: jax.vmap(lambda k: compute_p2p(j, k))(indices_ptcl))(indices_ptcl)
    grid_up2o, grid_Tp2o = jax.vmap(lambda j: jax.vmap(lambda k: compute_p2o(j, k))(indices_obs))(indices_ptcl)
    grid_uo2p, grid_To2p = jax.vmap(lambda j: jax.vmap(lambda k: compute_o2p(j, k))(indices_ptcl))(indices_obs)
    grid_uo2o, grid_To2o = jax.vmap(lambda j: jax.vmap(lambda k: compute_o2o(j, k))(indices_obs))(indices_obs)

    # 2. Reassemble each quadrant
    def assemble_quadrant(grid, Nsrc_per_obj, Nsrc_tot_obj, Ntrg_per_obj, Ntrg_tot_obj): # for ptcl: N_per_obj = N_ptcl, N_tot_obj = Nptx

        t_xx = grid[:, :, :Ntrg_per_obj, :Nsrc_per_obj]    # (num_j, num_k, Ntrg_per_obj, Nsrc_per_obj)
        t_xy = grid[:, :, :Ntrg_per_obj, Nsrc_per_obj:]    
        t_yx = grid[:, :, Ntrg_per_obj:, :Nsrc_per_obj]    
        t_yy = grid[:, :, Ntrg_per_obj:, Nsrc_per_obj:]    
        
        grid_xx = t_xx.transpose(1, 2, 0, 3).reshape(Ntrg_tot_obj, Nsrc_tot_obj)
        grid_xy = t_xy.transpose(1, 2, 0, 3).reshape(Ntrg_tot_obj, Nsrc_tot_obj)
        grid_yx = t_yx.transpose(1, 2, 0, 3).reshape(Ntrg_tot_obj, Nsrc_tot_obj)
        grid_yy = t_yy.transpose(1, 2, 0, 3).reshape(Ntrg_tot_obj, Nsrc_tot_obj)
        return jnp.block([[grid_xx, grid_xy],
                        [grid_yx, grid_yy]])
    # 3. Build the final 2Nptx x 2Nptx matrix
    u_p2p = assemble_quadrant(grid_up2p, N_ptcl, Nptx, N_ptcl, Nptx)
    u_p2o = assemble_quadrant(grid_up2o, N_ptcl, Nptx, N_obs, Nobsx)
    u_o2p = assemble_quadrant(grid_uo2p, N_obs, Nobsx, N_ptcl, Nptx)
    u_o2o = assemble_quadrant(grid_uo2o, N_obs, Nobsx, N_obs, Nobsx)
    T_p2p = assemble_quadrant(grid_Tp2p, N_ptcl, Nptx, N_ptcl, Nptx)
    # T_p2o = assemble_quadrant(grid_Tp2o, N_ptcl, Nptx, N_obs, Nobsx)
    T_o2p = assemble_quadrant(grid_To2p, N_obs, Nobsx, N_ptcl, Nptx)
    # T_o2o = assemble_quadrant(grid_To2o, N_obs, Nobsx, N_obs, Nobsx)


    A22 = jnp.block([[u_o2o, u_p2o],
                    [u_o2p, u_p2p]])
    A22_T = jnp.hstack([T_o2p, T_p2p])
    
    A22 = jnp.eye(2*len(ptx)+2*len(obsx)) / 2. + A22
    A = jnp.block([[A11, A12],
                   [A21, A22]])
    
    [B1,_,_] = StoSLP(sx, snx, px, pwt, mu, jnp.array([]))
    [B2,_,B2_T] = StoSLP(aptx, aptnx, px, pwt, mu, jnp.array([]))
    B2 = B2[idx,:]
    B2_T = B2_T[idx,:]
    B2_T = B2_T[2*Nobsx:,:]
    B = jnp.vstack([B1, B2])

    [CLD, _, TLD] = srcsum_dl(peri_len, lx, lnx, sx, snx, sw, mu)
    [CRD, _, TRD] = srcsum_dl(-peri_len, rx, rnx, sx, snx, sw, mu) 
    C1 = jnp.vstack([CRD - CLD, TRD - TLD])
    [CLDp, _, TLDp] = srcsum_dl(peri_len, lx, lnx, aptx, aptnx, aptw, mu)
    [CRDp, _, TRDp] = srcsum_dl(-peri_len, rx, rnx, aptx, aptnx, aptw, mu) 
    [CLSp, _, TLSp] = srcsum_sl(peri_len, lx, lnx, aptx, aptw, mu)
    [CRSp, _, TRSp] = srcsum_sl(-peri_len, rx, rnx, aptx, aptw, mu) 
    C2 = jnp.vstack([CRDp - CLDp + CRSp - CLSp, TRDp - TLDp + TRSp - TLSp])
    C2 = C2[:,idx] # swap source order
    C = jnp.hstack([C1, C2])

    [QL, _, QLt] = StoSLP(lx, lnx, px, pwt, mu, jnp.array([]))
    [QR, _, QRt ]= StoSLP(rx, rnx, px, pwt, mu, jnp.array([]))
    Q = jnp.vstack([QR - QL, QRt - QLt])

    # Rigid body motion adding to boundary conditions
    total_rows = 2*Nsx + 2*Nobsx + 2*Nptx
    total_cols = total_rows + 3 * num_ptcl
    bc_sqr = jnp.zeros((2*Nptx,3*num_ptcl))
    ptx_2d = ptx.reshape(num_ptcl,N_ptcl) # ASSUMES same discr. on each ptcl
    # X - Xc
    Xc = jnp.mean(ptx_2d, axis=1, keepdims=1)
    XmXc = ptx_2d - Xc
    XmXc_flat = XmXc.reshape(-1)
    
    row_indices = jnp.arange(Nptx)
    col_ux = jnp.repeat(jnp.arange(0, num_ptcl), N_ptcl)
    col_uy = jnp.repeat(jnp.arange(num_ptcl, 2 * num_ptcl), N_ptcl)
    # col_ux = jnp.repeat(jnp.arange(0, num_ptcl * 2, 2), N_ptcl)
    # col_uy = jnp.repeat(jnp.arange(1, num_ptcl * 2, 2), N_ptcl)
    bc_sqr = bc_sqr.at[row_indices, col_ux].set(-1.0)
    bc_sqr = bc_sqr.at[row_indices + Nptx, col_uy].set(-1.0)
    col_omega = jnp.repeat(jnp.arange(num_ptcl * 2, 3 * num_ptcl), N_ptcl)
    bc_sqr = bc_sqr.at[row_indices, col_omega].set(-jnp.imag(XmXc_flat))
    bc_sqr = bc_sqr.at[row_indices + Nptx, col_omega].set(jnp.real(XmXc_flat))

    bc_gamma_mat = jnp.zeros((total_rows, total_cols))
    bc_gamma_mat = bc_gamma_mat.at[:, :total_rows].set(A)
    bc_gamma_mat = bc_gamma_mat.at[2*Nsx+2*Nobsx:, total_rows:].set(bc_sqr)

    # Net force zero and net torque zero conditions
    T3 = A22_T
    ptcl_eye = jnp.hstack([jnp.zeros((2*Nptx,2*Nobsx)), jnp.eye(2*Nptx)/2.])
    fMat_ptcl = - ( -ptcl_eye + T3)
    fMat_wall = - A21_T
    fMat_prx = - B2_T
    
    ptcl_cell_w = ptw.reshape(num_ptcl, N_ptcl) 
    WF = jax.scipy.linalg.block_diag(*[w[None, :] for w in ptcl_cell_w])
    tx_vals = -ptcl_cell_w * jnp.imag(XmXc)
    ty_vals =  ptcl_cell_w * jnp.real(XmXc)
    WTx = jax.scipy.linalg.block_diag(*[t[None, :] for t in tx_vals])
    WTy = jax.scipy.linalg.block_diag(*[t[None, :] for t in ty_vals])
    # MATLAB: [WFx, 0; 0, WFy]
    intF_op = jax.scipy.linalg.block_diag(WF, WF)
    # MATLAB: [WTx, WTy]
    intT_op = jnp.concatenate([WTx, WTy], axis=1)
    mid_zeros = jnp.zeros((2 * Nptx, 3 * num_ptcl))
    big_mat = jnp.concatenate([fMat_wall, fMat_ptcl, mid_zeros, fMat_prx], axis=1)

    intF = intF_op @ big_mat
    intT = intT_op @ big_mat

    # --- Full ELS matrix ---
    E = jnp.block([[bc_gamma_mat, B],
                   [intF],
                   [intT],
                   [C, jnp.zeros((C.shape[0],3*num_ptcl)), Q]])

    return E, bc_gamma_mat, B, intF,intT, C, Q
ELSmatrix_rbm_obs = jit(ELSmatrix_rbm_obs, static_argnums=(15,24))

# def srcsum_self(kernel_self, kernel, trlist, sx, snx, sxp, scur, sw, mu):
#     trlist = jnp.atleast_1d(trlist)
#     n_tr = trlist.size
#     # first translation
#     sx1 = sx + trlist[0]
#     [u,p,T] = kernel(sx, snx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))
#     for i in range(1,n_tr):
#         sx1 = sx + trlist[i]
#         if i==1: # Have to hard code: len 3 trlist, middle one is 0.
#             [tempu,tempp,tempT] = kernel_self(sx, snx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
#         else:
#             [tempu,tempp,tempT] = kernel(sx, snx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))
#         u = u + tempu
#         p = p + tempp
#         T = T + tempT
#     return u,p,T

@jit
def srcsum_dl(trlist, tx, tnx, sx, snx, sw, mu):
    trlist = jnp.atleast_1d(trlist)
    n_tr = trlist.size

    # first translation
    sx1 = sx + trlist[0]
    [u,p,T] = StoDLP(tx, tnx, sx1, snx, sw, mu, jnp.array([]))

    for i in range(1,n_tr):
        sx1 = sx + trlist[i]
        [tempu,tempp,tempT] = StoDLP(tx, tnx, sx1, snx, sw, mu, jnp.array([]))
        u = u + tempu
        p = p + tempp
        T = T + tempT

    return u,p,T

@jit
def srcsum_sl(trlist, tx, tnx, sx, sw, mu):
    trlist = jnp.atleast_1d(trlist)
    n_tr = trlist.size

    # first translation
    sx1 = sx + trlist[0]
    [u,p,T] = StoSLP(tx, tnx, sx1, sw, mu, jnp.array([]))

    for i in range(1,n_tr):
        sx1 = sx + trlist[i]
        [tempu,tempp,tempT] = StoSLP(tx, tnx, sx1, sw, mu, jnp.array([]))
        u = u + tempu
        p = p + tempp
        T = T + tempT

    return u,p,T

@jit
def srcsum_sl_self(peri_len, sx, snx,sxp, scur, sw, mu):
    sx1 = sx - peri_len
    [u,p,T] = StoSLP(sx, snx, sx1, sw, mu, jnp.array([]))

    [tempu,tempp,tempT] = StoSLP_self(sx, snx, sxp, scur, sw, mu, jnp.array([]))
    u = u + tempu
    p = p + tempp
    T = T + tempT

    sx1 = sx + peri_len
    [tempu,tempp,tempT] = StoSLP(sx, snx, sx1, sw, mu, jnp.array([]))
    u = u + tempu
    p = p + tempp
    T = T + tempT

    return u,p,T

@jit
def srcsum_dl_self(peri_len, sx, snx, scur, sw, mu):
    sx1 = sx - peri_len
    [u,p,T] = StoDLP(sx, snx, sx1, snx, sw, mu, jnp.array([]))

    [tempu,tempp,tempT] = StoDLP_self(sx, snx, scur, sw, mu, jnp.array([]))
    u = u + tempu
    p = p + tempp
    T = T + tempT

    sx1 = sx + peri_len
    [tempu,tempp,tempT] = StoDLP(sx, snx, sx1, snx, sw, mu, jnp.array([]))
    u = u + tempu
    p = p + tempp
    T = T + tempT

    return u,p,T

# def srcsum(kernel, trlist, tx, tnx, sx, snx, sxp, scur, sw, mu):
#     """ kernel is a Python-callable kernel function and must be static for the same reason """
#     trlist = jnp.atleast_1d(trlist)
#     n_tr = trlist.size

#     # first translation
#     sx1 = sx + trlist[0]
#     [u,p,T] = kernel(tx, tnx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))

#     for i in range(1,n_tr):
#         sx1 = sx + trlist[i]
#         [tempu,tempp,tempT] = kernel(tx, tnx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))
#         u = u + tempu
#         p = p + tempp
#         T = T + tempT
 
#     return u,p,T


@jit
def srcsum_ptcl_stodl_closeglobal(peri_len, tx, tnx, ptx, ptnx, pta, ptxp, ptw, ptwxp, sigma, mu):
    """
    Specific close eval functions for particles, hardcoded: side='e', using either stosl or dl closeglobal.
    stosl close requires looping over the particles.
    """
    if len(sigma.shape) > 1:
        mat = True
    else:
        mat = False

    # left copy, all should be far from center copy particles.
    ptx1 = ptx - peri_len
    [u,p,T] = StoDLP(tx, tnx, ptx1, ptnx, ptw, mu, sigma)

    # right copy, also should be far.
    ptx1 = ptx + peri_len
    [tempu,tempp,tempT] = StoDLP(tx, tnx, ptx1, ptnx, ptw, mu, sigma)
    u = u + tempu
    p = p + tempp
    T = T + tempT

    ptx1 = ptx
    Nptx = len(ptx)
    num_ptcl = len(pta)
    N_ptcl = int(Nptx/num_ptcl) # num discr. points per ptcl, assumed to be equal.
    for j in range(num_ptcl):
        ptx_cur = ptx1[j*N_ptcl:(j+1)*N_ptcl]
        ptxp_cur = ptxp[j*N_ptcl:(j+1)*N_ptcl]
        ptwxp_cur = ptwxp[j*N_ptcl:(j+1)*N_ptcl]
        # sigma_cur = jnp.concatenate([sigma[j*N_ptcl:(j+1)*N_ptcl], sigma[Nptx+j*N_ptcl:Nptx+(j+1)*N_ptcl]]) 
        if mat:
            sigma_cur = jnp.concatenate([sigma[j*N_ptcl:(j+1)*N_ptcl,:], sigma[Nptx+j*N_ptcl:Nptx+(j+1)*N_ptcl,:]])
        else:
            sigma_cur = jnp.concatenate([sigma[j*N_ptcl:(j+1)*N_ptcl], sigma[Nptx+j*N_ptcl:Nptx+(j+1)*N_ptcl]])


        [tempu,tempp,tempT] = stoDLP_closeglobal(tx, tnx, ptx_cur, ptxp_cur, ptwxp_cur, sigma_cur, mu)
        
        if mat:
            u = u + tempu
            p = p + tempp
            T = T + tempT
        else:
            u = u + tempu.ravel()
            p = p + tempp.ravel()
            T = T + tempT.ravel()

    return u,p,T

@jit
def srcsum_ptcl_stosl_closeglobal(peri_len, tx, tnx, ptx, ptt, pta, ptxp, ptw, sigma, mu):
    if len(sigma.shape) > 1:
        mat = True
    else:
        mat = False

    # left copy, all should be far from center copy particles.
    ptx1 = ptx - peri_len
    [u,p,T] = StoSLP(tx, tnx, ptx1, ptw, mu, sigma)

    # right copy, also should be far.
    ptx1 = ptx + peri_len
    [tempu,tempp,tempT] = StoSLP(tx, tnx, ptx1, ptw, mu, sigma)
    u = u + tempu
    p = p + tempp
    T = T + tempT

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
        if mat:
            sigma_cur = jnp.concatenate([sigma[j*N_ptcl:(j+1)*N_ptcl,:], sigma[(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl),:]])
        else:
            sigma_cur = jnp.concatenate([sigma[j*N_ptcl:(j+1)*N_ptcl], sigma[(Nptx+j*N_ptcl):(Nptx+(j+1)*N_ptcl)]])

        [tempu,tempp,tempT] = stoSLP_closeglobal(tx, tnx, ptx_cur, ptxp_cur, ptt_cur, pta_cur, ptw_cur, sigma_cur, mu)
        
        if mat:
            u = u + tempu
            p = p + tempp
            T = T + tempT
        else:
            u = u + tempu.ravel()
            p = p + tempp.ravel()
            T = T + tempT.ravel()

    return u,p,T


@jit
def srcsum_wall_closepanel(peri_len, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sws, mu):
    """ Source sum for panel-based discr. on wall (since only DL available, a more specific function)"""
    # first translation
    sx1 = sx - peri_len
    [u,p,T] = StoDLP(tx, tnx, sx1, snx, sws, mu, jnp.array([]))

    # Hardcode trlist 3 entries, L, middle, R
    sigma_real = jnp.eye(2*sx.shape[0])
    [tempu,tempp,tempT] = stoDLP_closepanel(tx, tnx, sx, sxp, sws, scur, sxlo, sxhi, sigma_real, mu)
    u = u + tempu
    p = p + tempp
    T = T + tempT

    sx1 = sx + peri_len
    [tempu,tempp,tempT] = StoDLP(tx, tnx, sx1, snx, sws, mu, jnp.array([]))
    u = u + tempu
    p = p + tempp
    T = T + tempT
    
    return u,p,T

# ----------------------------------------------------------------------
# --- EVAL functions ----------------------------------------------
# --- Now always for panel-based walls and global quadr on particles; 
# ---           exterior problem for ptcl and interior problem for wall;
# ---           DL only on wall and SL+DL on ptcl.
# ---     always include near-eval.
# ----------------------------------------------------------------------
@jit
def evalsol(tx, tnx, sx, snx, sw, px, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    wall_dens = edens[:2*N_wall]
    prx_dens = edens[2*N_wall:]

    [u, p, _] = StoSLP(tx, tnx, px, pwt, mu, prx_dens)
    [uD, pD, _] = srcsum_dl(trlist, tx, tnx, sx, snx, sw, mu)
    uD = uD @ wall_dens
    pD = pD @ wall_dens

    u = u+uD
    p = p+pD
    return u,p

@jit
def evalsol_panel(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, px, pwt, peri_len, mu, edens):
    N_wall = len(sx)
    wall_dens = edens[:2*N_wall]
    prx_dens = edens[2*N_wall:]

    [u, p, _] = StoSLP(tx, tnx, px, pwt, mu, prx_dens)
    [uD, pD, _] = srcsum_wall_closepanel(peri_len, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu)
    uD = uD @ wall_dens
    pD = pD @ wall_dens

    u = u+uD
    p = p+pD
    return u,p


@jit
def evalsol_ptcl(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pwt, peri_len, mu, edens):
    """ Used to be 'evalsol_ptcl_panel' """
    N_wall = len(sx)
    N_ptcl = len(ptx)
    wall_dens = edens[:2*N_wall]
    ptcl_dens = edens[2*N_wall:2*(N_wall+N_ptcl)]
    prx_dens = edens[2*(N_wall+N_ptcl):]

    [u, p, _] = StoSLP(tx, tnx, px, pwt, mu, prx_dens)
    [uD, pD, _] = srcsum_wall_closepanel(peri_len, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu)

    [uSp, pSp, _] = srcsum_ptcl_stosl_closeglobal(peri_len,tx,tnx,ptx,ptt,pta,ptxp,ptw,ptcl_dens,mu)
    [uDp, pDp] = srcsum_ptcl_stodl_closeglobal(peri_len,tx,tnx,ptx,ptnx,pta,ptxp,ptw,ptwxp,ptcl_dens,mu)

    uD = uD @ wall_dens
    pD = pD @ wall_dens
    uDSp = uDp + uSp
    pDSp = pDp + pSp

    u = u+uD+uDSp
    p = p+pD+pDSp
    return u,p

@jit
def evalsol_stokeslet(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, x_stokeslet, w_stokeslet, px, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    N_stokeslet = len(x_stokeslet)
    wall_dens = edens[:2*N_wall]
    f_stokeslet = edens[2*N_wall:2*(N_wall+N_stokeslet)]
    prx_dens = edens[2*(N_wall+N_stokeslet):]

    [u, p, _] = StoSLP(tx, tnx, px, pwt, mu, prx_dens)
    [uD, pD, _] = srcsum_wall_closepanel(peri_len, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu) # no need for DL traction so fill in tnx with snx
    x_stokeslet_near = x_stokeslet + trlist[0] # left copy first
    [u_ptsrc, p_ptsrc, _] = StoSLP(tx, tnx, x_stokeslet_near, w_stokeslet, mu, jnp.array([]))
    for i in range(1,len(trlist)):
        x_stokeslet_near = x_stokeslet + trlist[i]
        [u_ptsrc_temp, p_ptsrc_temp, _] = StoSLP(tx, tnx, x_stokeslet_near, w_stokeslet, mu, jnp.array([]))
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
def evalsol_rbm(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pwt, peri_len, mu, edens):
    N_wall = len(sx)
    N_ptcl = len(ptx)
    wall_dens = edens[:2*N_wall]
    ptcl_dens = edens[2*N_wall:2*(N_wall+N_ptcl)]
    Nskip = len(edens) - 2*(N_wall+N_ptcl+len(px)) # number of entries for U and Omega for each particle
    prx_dens = edens[2*(N_wall+N_ptcl)+Nskip:]
    
    [u, p, _] = StoSLP(tx, tnx, px, pwt, mu, prx_dens)
    [uD, pD, _] = srcsum_wall_closepanel(peri_len, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu)

    [uSp, pSp, _] = srcsum_ptcl_stosl_closeglobal(peri_len,tx,tnx,ptx,ptt,pta,ptxp,ptw,ptcl_dens,mu)
    [uDp, pDp, _] = srcsum_ptcl_stodl_closeglobal(peri_len,tx,tnx,ptx,ptnx,pta,ptxp,ptw,ptwxp,ptcl_dens,mu)
    # trlist = jnp.array([-peri_len,0,peri_len])
    # [uSp, pSp, _] = srcsum_sl(trlist,tx,tnx,ptx,ptw,mu)
    # [uDp, pDp, _] = srcsum_dl(trlist,tx,tnx,ptx,ptnx,ptw,mu)

    uD = uD @ wall_dens
    pD = pD @ wall_dens
    uDSp = uDp + uSp
    pDSp = pDp + pSp
    # uDSp = (uDp + uSp) @ ptcl_dens
    # pDSp = (pDp + pSp) @ ptcl_dens

    u = u+uD+uDSp
    p = p+pD+pDSp
    return u,p

@jit
def evalsol_rbm_obs(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, obsx, obsnx, obst, obsa, obsxp, obsw, obswxp, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pxp, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    Nsx = len(sx)
    Nptx = len(ptx)
    Nobsx = len(obsx)
    wall_dens = edens[:2*Nsx]
    obs_dens = edens[2*Nsx:2*Nsx+2*Nobsx]
    ptcl_dens = edens[2*Nsx+2*Nobsx:2*(Nsx+Nobsx+Nptx)]
    Nskip = len(edens) - 2*(Nsx+Nobsx+Nptx+len(px)) # number of entries for U and Omega for each particle
    prx_dens = edens[2*(Nsx+Nobsx+Nptx)+Nskip:]

    # jax.debug.print("norm of wall dens: {a}, norm of obs dens: {b}, norm of ptcl dens: {c}, norm of prx dens = {d}", a=jnp.linalg.norm(wall_dens), b=jnp.linalg.norm(obs_dens), c=jnp.linalg.norm(ptcl_dens), d=jnp.linalg.norm(prx_dens))

    [u, p, _] = StoSLP(tx, tnx, px, pwt, mu, prx_dens) # source = proxy
    [uD, pD, _] = srcsum_wall_closepanel(peri_len, tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, mu) # source = walls

    [uSp, pSp, _] = srcsum_ptcl_stosl_closeglobal(peri_len,tx,tnx,ptx,ptt,pta,ptxp,ptw,ptcl_dens,mu) # source = ptcls, SL
    [uDp, pDp, _] = srcsum_ptcl_stodl_closeglobal(peri_len,tx,tnx,ptx,ptnx,pta,ptxp,ptw,ptwxp,ptcl_dens,mu) # source = ptcls, DL

    [uSo, pSo, _] = srcsum_ptcl_stosl_closeglobal(peri_len,tx,tnx,obsx,obst,obsa,obsxp,obsw,obs_dens,mu) # source = obstacle, SL
    [uDo, pDo, _] = srcsum_ptcl_stodl_closeglobal(peri_len,tx,tnx,obsx,obsnx,obsa,obsxp,obsw,obswxp,obs_dens,mu) # source = obstacle, DL

    uD = uD @ wall_dens
    pD = pD @ wall_dens
    uDSp = uDp + uSp
    pDSp = pDp + pSp
    uDSo = uDo + uSo
    pDSo = pDo + pSo

    u = u+uD+uDSp+uDSo
    p = p+pD+pDSp+pDSo
    return u,p


ELSmatrix_ptcl = jit(ELSmatrix_ptcl, static_argnums=(9))
ELSmatrix_ptcl_update = jit(ELSmatrix_ptcl_update, static_argnums=(10))
ELSmatrix_rbm = jit(ELSmatrix_rbm, static_argnums=(9))
# srcsum_self = jit(srcsum_self, static_argnums=(0,1))
# srcsum = jit(srcsum, static_argnums=(0,))