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
    # [A22_dl,_,_] = srcsum_self_ptcl(StoDLP_self, StoDLP, trlist, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, mu) 
    [A22_dl,_,_] = srcsum_self(StoDLP_self, StoDLP, trlist, ptx, ptnx, ptxp, ptcur, ptw, mu) # double layer self does not have single-particle assumption
    # [A22_sl,_,_] = srcsum_self_ptcl(StoSLP_self, StoSLP, trlist, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, mu) # to enable multi-particle interactions.
    
    # Choco Jan 2026: move all of srcsum_self_ptcl in main frame of ELSmatrix to allow JIT, since another layer of static num_ptcl is no-go.
    #################################
    n_tr = trlist.size
    N_ptcl_double = Nsx/num_ptcl
    N_ptcl = N_ptcl_double.astype(int)
    # first translation -- all src evals are far.
    ptx1 = ptx + trlist[0]
    [u,_,_] = StoSLP(ptx, ptnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, jnp.array([]))
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
            [tempu,_,_] = StoSLP(ptx, ptnx, ptx1, ptnx, ptxp, ptcur, ptw, mu, jnp.array([]))
            u = u + tempu
    A22_sl = u
    ###################################

    A22 = -jnp.eye(2*len(ptx)) / 2. + A22_dl + A22_sl
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
            # print("i = 1")
            [tempu,tempp,tempT] = kernel_self(sx, snx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
        else:
            # print("i!= 1")
            [tempu,tempp,tempT] = kernel(sx, snx, sx1, snx, sxp, scur, sw, mu, jnp.array([]))
        u = u + tempu
        p = p + tempp
        T = T + tempT
        # print(u.shape)
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
def evalsol2(tx, tnx, sx, snx, sxp, scur, sw, px, pxp, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    wall_dens = edens[:2*N_wall]
    prx_dens = edens[2*N_wall:]

    [u, p, _] = StoSLP(tx, jnp.array([]), px, jnp.array([]), pxp, jnp.array([]), mu, prx_dens)
    [uD, pD, _] = srcsum(StoDLP, trlist, tx, tnx, sx, snx, sxp, scur, sw, mu) # no need for DL traction so fill in tnx with snx
    uD = uD @ wall_dens
    pD = pD @ wall_dens

    u = u+uD
    p = p+pD
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
def evalsol_stokeslet(tx, tnx, sx, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, peri_len, mu, edens):
    trlist = peri_len*jnp.array([-1,0,1])
    N_wall = len(sx)
    N_stokeslet = len(x_stokeslet)
    wall_dens = edens[:2*N_wall]
    f_stokeslet = edens[2*N_wall:2*(N_wall+N_stokeslet)]
    prx_dens = edens[2*(N_wall+N_stokeslet):]

    [u, p, _] = StoSLP(tx, tnx, px, jnp.array([]), pxp, jnp.array([]), pwt, mu, prx_dens)
    [uD, pD, _] = srcsum(StoDLP, trlist, tx, tnx, sx, snx, sxp, scur, sw, mu) # no need for DL traction so fill in tnx with snx
    # TODO stokeslet to target all near.
    # [uDp, pDp, _] = srcsum(StoDLP, trlist, tx, tnx, x_stokeslet, ptnx, ptxp, ptcur, ptw, mu)
    # [uSp, pSp, _] = srcsum(StoSLP, trlist, tx, tnx, x_, ptnx, ptxp, ptcur, ptw, mu)
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