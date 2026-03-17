# A test script to observe the update routine for only part of the ELSmatrix, 
#       when only the particle positions change per (RL or time) iteration.
#       Compared to the runtime for remaking the ELSmatrix of whole extended system at each iteration.
# 
# Choco Li, March 2026

import time
from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.io import savemat
import sys, os
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the folder containing the module you want to import
utils_path = os.path.join(current_dir, '..')
peri_path = os.path.join(current_dir, '../periodic')
# Add the folder to the system path
sys.path.append(utils_path)
sys.path.append(peri_path)
from periodic.structure_jax import *
from periodic.periodic_ELS_jax import *

# jax.clear_caches()

# Set discretization parameters
Np_wall = 40 # number of panels
p_wall = 10 # GL grid order on each panel
N_wall = Np_wall * p_wall # total number of discr. points on EACH wall
N_ptcl = 200 # total number of discr. points on EACH particle (global quadr)
N_side = 100
N_prx = 2*N_side
peri_len = 2*jnp.pi

# Set up channel walls
Z_top = lambda t : peri_len / (2*jnp.pi) * (2*jnp.pi - t) + 1j*(1 + 0.3*jnp.sin(2*jnp.pi - t)) # NEW: wrong in rescaling t using peri_len, should rescale x instead. t always [0,2pi]
Zp_top = lambda t : -peri_len / (2*jnp.pi) - 1j*(0.3*jnp.cos(2*jnp.pi-t))
Zpp_top = lambda t : -1j*0.3*jnp.sin(2*jnp.pi-t)
Z_bot = lambda t : peri_len / (2*jnp.pi) * t + 1j*(-1 + 0.3*jnp.sin(t))
Zp_bot = lambda t : peri_len / (2*jnp.pi) + 1j*(0.3*jnp.cos(t))
Zpp_bot = lambda t : -1j*0.3*jnp.sin(t)
[sx,sxp,snx,scur,sw,swxp,sxlo,sxhi] = channel_wall_glpanels(Z_top,Np_wall,p_wall,Zp_top,Zpp_top)
[sx2,sxp2,snx2,scur2,sw2,swxp2,sxlo2,sxhi2] = channel_wall_glpanels(Z_bot,Np_wall,p_wall,Zp_bot,Zpp_bot)
# Combine top and bottom walls into one Wall object.
sx = jnp.concatenate([sx,sx2])
sxp = jnp.concatenate([sxp,sxp2])
snx = jnp.concatenate([snx,snx2])
scur = jnp.concatenate([scur,scur2])
sw = jnp.concatenate([sw,sw2])
swxp = jnp.concatenate([swxp,swxp2])
sxlo = jnp.concatenate([sxlo,sxlo2])
sxhi = jnp.concatenate([sxhi,sxhi2])
vis(sx, snx, True)

# Add particle 
num_ptcl = 2 # number of particles on the interior, for self eval.
Z_ptcl = lambda t : 1 + 0.3*jnp.cos(t) + 1j*(0.3*jnp.sin(t)+0.25)
Zp_ptcl = lambda t : - 0.3*jnp.sin(t) + 1j*0.3*jnp.cos(t)
Zpp_ptcl = lambda t : - 0.3*jnp.cos(t) - 1j*0.3*jnp.sin(t)
[ptx,ptxp,ptnx,ptcur,ptw] = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl) # CHANGED MAR 2026: no flipping normals for close eval, Ematrix changed accordingly.
# Add another particle
Z_ptcl = lambda t : 5 + 0.2*jnp.cos(t) + 1j*(0.2*jnp.sin(t)+0.)
Zp_ptcl = lambda t : - 0.2*jnp.sin(t) + 1j*0.2*jnp.cos(t)
Zpp_ptcl = lambda t : - 0.2*jnp.cos(t) - 1j*0.2*jnp.sin(t)
[ptx2,ptxp2,ptnx2,ptcur2,ptw2] = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
# Combine particle info, ASSUMES same discr on each ptcl!
ptx = jnp.concatenate([ptx,ptx2])
ptxp = jnp.concatenate([ptxp,ptxp2])
ptnx = jnp.concatenate([ptnx,ptnx2])
ptcur = jnp.concatenate([ptcur,ptcur2])
ptw = jnp.concatenate([ptw,ptw2])


# Set up side walls 
[lx,lnx] = side_wall(0., 2, N_side)
[rx,rnx] = side_wall(0.+peri_len, 2, N_side)
lx = lx - 1j
rx = rx - 1j

# Set up proxy sources
R = 1.1 * peri_len
[px,pxp,pnx,pwt] = proxy(R, peri_len, N_prx)

mu = 0.7

# Make ELS matrix -- first iteration, call update to compile
[E,A,B,C,Q] = ELSmatrix_ptcl(sx, snx, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu)
[E2,A2,B2,C2,Q2] = ELSmatrix_ptcl_update(A,B,C,Q, sx, snx, sw, ptx, ptnx, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu)

# NEXT, update particle position using vslip (arbitrarily just move the particles in +x direction for now)
Z_ptcl = lambda t : 1 + 0.1 + 0.3*jnp.cos(t) + 1j*(0.3*jnp.sin(t)+0.25)
Zp_ptcl = lambda t : - 0.3*jnp.sin(t) + 1j*0.3*jnp.cos(t)
Zpp_ptcl = lambda t : - 0.3*jnp.cos(t) - 1j*0.3*jnp.sin(t)
[ptx,ptxp,ptnx,ptcur,ptw] = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl) 
# Add another particle
Z_ptcl = lambda t : 5 + 0.1 + 0.2*jnp.cos(t) + 1j*(0.2*jnp.sin(t)+0.)
Zp_ptcl = lambda t : - 0.2*jnp.sin(t) + 1j*0.2*jnp.cos(t)
Zpp_ptcl = lambda t : - 0.2*jnp.cos(t) - 1j*0.2*jnp.sin(t)
[ptx2,ptxp2,ptnx2,ptcur2,ptw2] = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
# Combine particle info, ASSUMES same discr on each ptcl!
ptx = jnp.concatenate([ptx,ptx2])
ptxp = jnp.concatenate([ptxp,ptxp2])
ptnx = jnp.concatenate([ptnx,ptnx2])
ptcur = jnp.concatenate([ptcur,ptcur2])
ptw = jnp.concatenate([ptw,ptw2])

# Compare time of making ELSmatrix from scratch vs updating only part of the block.
start1 = time.perf_counter()
[E1,A1,B1,C1,Q1] = ELSmatrix_ptcl(sx, snx, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu)
Q1.block_until_ready()
end1 = time.perf_counter()
start2 = time.perf_counter()
[E2,A2,B2,C2,Q2] = ELSmatrix_ptcl_update(A, B, C, Q, sx, snx, sw, ptx, ptnx, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu)
Q2.block_until_ready()
end2 = time.perf_counter()
print(f"norm of difference between E2 (made whole) and E2 (sub out) = {jnp.linalg.norm(E1-E2):.3g}.")
print(f"Time to run whole matrix build: {end1-start1:.3g}, run only update: {end2-start2:.3g}")
