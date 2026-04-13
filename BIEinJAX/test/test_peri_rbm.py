# A test script to solve 2D Stokes periodic rigid body mobility problem
# Choco Li Mar 16, 2026

from jax import config
config.update("jax_enable_x64", True)

import time
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

@jit
def get_vslip(B1, B2, ptcl_coords_2d, ptcl_tang):
    """
    ptcl_coords: Array of shape (num_particles, num_nodes) 
                 containing complex coordinates.
    ptcl_tang; 1d array (num_ptcl * num_nodes), ptcl fast dimension slow.
    """
    xc = jnp.mean(ptcl_coords_2d, axis=1, keepdims=True)
    xmxc_tot = (ptcl_coords_2d - xc).reshape(-1)
    theta = jnp.atan2(jnp.imag(xmxc_tot), jnp.real(xmxc_tot))
    u_theta = B1 * jnp.sin(theta) + B2 * jnp.sin(theta) * jnp.cos(theta)
    # vslip = -u_theta * jnp.sin(theta) + 1j*u_theta * jnp.cos(theta)
    if len(ptcl_tang) == 0:
        print("WARNING: no particle tangent given when computing vslip, will assume circle.")
        vslip = -u_theta * jnp.sin(theta) + 1j* u_theta * jnp.cos(theta)
    else:
        vslip = u_theta * jnp.real(ptcl_tang) + 1j* u_theta * jnp.imag(ptcl_tang)
    return vslip


# Set discretization parameters
Np_wall = 10 # number of panels
p_wall = 10 # GL grid order on each panel
N_wall = Np_wall * p_wall # total number of discr. points on EACH wall
N_ptcl = 50 # total number of discr. points on EACH particle (global quadr)
N_obs = 40 # total number of discr. points on EACH obstacle (global quadr)
N_side = 40
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
# vis(sx, snx, True)

# Add particle 
num_ptcl = 2 # number of particles on the interior, for self eval.
if num_ptcl > 0:
    Z_ptcl = lambda t : 1 + 0.3*jnp.cos(t-jnp.pi/7) + 1j*(0.3*jnp.sin(t)+0.25)
    Zp_ptcl = lambda t : - 0.3*jnp.sin(t-jnp.pi/7) + 1j*0.3*jnp.cos(t)
    Zpp_ptcl = lambda t : - 0.3*jnp.cos(t-jnp.pi/7) - 1j*0.3*jnp.sin(t)
    [ptx,ptxp,ptnx,ptcur,ptw] = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
    ptwxp = 2*jnp.pi/N_ptcl * ptxp
    ptt = jnp.linspace(0, 2 * jnp.pi, N_ptcl, endpoint=False)
    pta = jnp.array([1+0.25j])
    # r_list = jnp.array([0.3])
    if num_ptcl > 1:
        # Add another particle
        Z_ptcl = lambda t : 5 + 0.2*jnp.cos(t) + 1j*(0.2*jnp.sin(t)+0.)
        Zp_ptcl = lambda t : - 0.2*jnp.sin(t) + 1j*0.2*jnp.cos(t)
        Zpp_ptcl = lambda t : - 0.2*jnp.cos(t) - 1j*0.2*jnp.sin(t)
        [ptx2,ptxp2,ptnx2,ptcur2,ptw2] = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
        ptwxp2 = 2*jnp.pi/N_ptcl * ptxp2
        ptt2 = jnp.linspace(0, 2 * jnp.pi, N_ptcl, endpoint=False)
        pta2 = jnp.array([5+0j])
        # Combine particle info, ASSUMES same discr on each ptcl!
        # TODO later: can allow for differences if use vstack and pads. Will need to change ptcl all to all in that case.
        ptx = jnp.concatenate([ptx,ptx2])
        ptxp = jnp.concatenate([ptxp,ptxp2])
        ptnx = jnp.concatenate([ptnx,ptnx2])
        ptcur = jnp.concatenate([ptcur,ptcur2])
        ptw = jnp.concatenate([ptw,ptw2])
        ptt = jnp.concatenate([ptt,ptt2])
        pta = jnp.concatenate([pta,pta2])
        ptwxp = jnp.concatenate([ptwxp,ptwxp2])
        # r_list = jnp.array([0.3, 0.2])
    # vis(ptx, ptnx, True)
    # Xc_list = pta

num_obs = 2
if num_obs > 0:
    Z_obs = lambda t : 3.6 + 0.24*jnp.cos(t) + 1j*(0.24*jnp.sin(t)-0.4)
    Zp_obs = lambda t : -0.24*jnp.sin(t) + 1j*0.24*jnp.cos(t)
    Zpp_obs = lambda t : -0.24*jnp.cos(t) - 1j*0.24*jnp.sin(t)
    # Z_obs = lambda t : 3.6 + 0.2*jnp.cos(t-jnp.pi/8.) + 1j*(0.24*jnp.sin(t)-0.4) # new test: rotated ellipse
    # Zp_obs = lambda t : -0.2*jnp.sin(t-jnp.pi/8.) + 1j*0.24*jnp.cos(t)
    # Zpp_obs = lambda t : -0.2*jnp.cos(t-jnp.pi/8.) - 1j*0.24*jnp.sin(t)
    [obsx,obsxp,obsnx,obscur,obsw] = channel_wall_func(Z_obs,N_obs,Zp_obs,Zpp_obs)
    obswxp = 2*jnp.pi/N_obs * obsxp
    obst = jnp.linspace(0, 2 * jnp.pi, N_obs, endpoint=False)
    obsa = jnp.array([3.6-0.4j])
    # vis(obsx, obsnx, True)
    if num_obs > 1:
        # Add another particle
        Z_obs = lambda t : 2.18 + 0.198*jnp.cos(t) + 1j*(0.198*jnp.sin(t)+0.76)
        Zp_obs = lambda t : - 0.198*jnp.sin(t) + 1j*0.198*jnp.cos(t)
        Zpp_obs = lambda t : - 0.198*jnp.cos(t) - 1j*0.198*jnp.sin(t)
        [obsx2,obsxp2,obsnx2,obscur2,obsw2] = channel_wall_func(Z_obs,N_obs,Zp_obs, Zpp_obs)
        obswxp2 = 2*jnp.pi/N_obs * obsxp2
        obst2 = jnp.linspace(0, 2 * jnp.pi, N_obs, endpoint=False)
        obsa2 = jnp.array([2.18+0.76j])

        # Combine particle info, ASSUMES same discr on each ptcl!
        obsx = jnp.concatenate([obsx,obsx2])
        obsxp = jnp.concatenate([obsxp,obsxp2])
        obsnx = jnp.concatenate([obsnx,obsnx2])
        obscur = jnp.concatenate([obscur,obscur2])
        obsw = jnp.concatenate([obsw,obsw2])
        obst = jnp.concatenate([obst,obst2])
        obsa = jnp.concatenate([obsa,obsa2])
        obswxp = jnp.concatenate([obswxp,obswxp2])
else:
    obsx = jnp.array([])
    obsnx = jnp.array([])
    obsxp = jnp.array([])
    obscur = jnp.array([])
    obsw = jnp.array([])
    obswxp = jnp.array([])
    obst = jnp.array([])
    obsa = jnp.array([])

# vis(obsx, obsnx, True, True)

# Set up side walls 
[lx,lnx] = side_wall(0., 2, N_side)
[rx,rnx] = side_wall(0.+peri_len, 2, N_side)
lx = lx - 1j
rx = rx - 1j
# vis(lx, lnx, True)
# vis(rx, rnx, True)

# Set up proxy sources
R = 1.1 * peri_len
[px,pxp,pnx,pwt] = proxy(R, peri_len, N_prx)
# vis(px, pnx, True, True)

mu = 0.7

# Make ELS matrix
# [E,bc_gamma_mat,B,intF,intT,C,Q] = ELSmatrix_rbm(sx, snx, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu)
# Q.block_until_ready()
# start_setup = time.perf_counter()
# [E,bc_gamma_mat,B,intF,intT,C,Q] = ELSmatrix_rbm(sx, snx, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu)
[E,bc_gamma_mat,B,intF,intT,C,Q] = ELSmatrix_rbm_obs(sx, snx, sxp, sxlo, sxhi, scur, sw, obsx, obsnx, obst, obsa, obsxp, obscur, obsw, obswxp, num_obs, ptx, ptnx, ptt, pta, ptxp, ptcur, ptw, ptwxp, num_ptcl, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu)
# end_setup = time.perf_counter()

tx = jnp.array([2+0.2j,4+0.1j])
tnx = jnp.array([1+0j,1+0j])
# tx = jnp.array([2+0.2j])
# tnx = jnp.array([1+0j])

# # Slip velocity parameters
# B1 = 1.23; B2 = -0.73
# vrhs = jnp.zeros((sx.size*2,))
# ptx_2d = ptx.reshape((num_ptcl,N_ptcl))
# vrhs_ptcl_cpx = get_vslip(B1,B2,ptx_2d,1j*ptnx)
# vrhs_ptcl = jnp.concatenate([jnp.real(vrhs_ptcl_cpx),jnp.imag(vrhs_ptcl_cpx)])
# jump = 0
# Tjump = -jump * jnp.concatenate([jnp.real(rnx), jnp.imag(rnx)])
# erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((3*num_ptcl,)), jnp.zeros((2*N_side,)), Tjump])
# # Solve for density
# [edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15)

N_nodes_wall = len(sx)
N_nodes_ptcls = len(ptx)
N_nodes_obs = len(obsx)
# start_solve3 = time.perf_counter()
B1 = 1.23; B2 = -0.73
vrhs = jnp.zeros((N_nodes_wall*2+N_nodes_obs*2,))
ptx_2d = ptx.reshape((num_ptcl,N_ptcl))
vrhs_ptcl_cpx = get_vslip(B1,B2,ptx_2d,1j*ptnx)
vrhs_ptcl = jnp.concatenate([jnp.real(vrhs_ptcl_cpx),jnp.imag(vrhs_ptcl_cpx)])
jump = 0
Tjump = -jump * jnp.concatenate([jnp.real(rnx), jnp.imag(rnx)])
erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((3*num_ptcl,)), jnp.zeros((2*N_side,)), Tjump]) 

# Solve for density
[edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15)
# end_solve3 = time.perf_counter()
print(f'resid norm = {resid[0]:.3g}')
wall_dens = edens[:2*N_nodes_wall]
obs_dens = edens[2*N_nodes_wall:2*(N_nodes_wall+N_nodes_obs)] 
ptcl_dens = edens[2*(N_nodes_wall+N_nodes_obs):2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls)] 
UOmega_all = edens[2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls):2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls)+3*num_ptcl]
prx_dens = edens[2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls) + 3*num_ptcl:]
wall_dens_norm = jnp.linalg.norm(wall_dens)
obs_dens_norm = jnp.linalg.norm(obs_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
prx_dens_norm = jnp.linalg.norm(prx_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of obstacle density = {obs_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}')
# TODO: print statement below just for 2 particle case for now..
print(f"Mobility problem for B1={B1},B2={B2}, is U1 = [{UOmega_all[0]:.3g},{UOmega_all[1]:.3g}], Omega1 = {UOmega_all[2]:.3g} U2 = [{UOmega_all[3]:.3g},{UOmega_all[4]:.3g}], Omega2 = {UOmega_all[5]:.3g}")

# # Evaluate at target
# [ut, pt] = evalsol_rbm(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pwt, peri_len, mu, edens)
# ut.block_until_ready()

# start_eval = time.perf_counter()
# [ut, pt] = evalsol_rbm(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pwt, peri_len, mu, edens)
[ut, pt] = evalsol_rbm_obs(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, obsx, obsnx, obst, obsa, obsxp, obsw, obswxp, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pxp, pwt, peri_len, mu, edens)
# ut.block_until_ready()
# end_eval = time.perf_counter()
print('u velocity at zt = {:.12g}, {:.12g}, p at first trg = {:.12g}, at second = {:.12g}'.format(ut[0], ut[1], pt[0], pt[1]))
# print(f"TIMING RESULTS: setup time: {end_setup - start_setup:.3g}, solve time: {end_solve - start_solve:.3g}, eval time: {end_eval - start_eval:.3g}.\n")

def plot_streamlines_total(edens, nxg=140, ng=70, ypad=0.5, density=1.3, buffer_factor=1.0):
    xg = np.linspace(0.0, peri_len, nxg)
    yg = np.linspace(-1.0-ypad, 1.0+ypad, ng)
    X, Y = np.meshgrid(xg, yg)

    dx = (xg.max() - xg.min()) / (nxg - 1)
    dy = (yg.max() - yg.min()) / (ng - 1)
    delta = buffer_factor * min(dx, dy)

    Xj = jnp.array(xg)
    YT = np.array(jnp.imag(Z_top(2*jnp.pi-Xj))) # TODO: hardcoded t->2pi-t flip for now.
    # YT = np.array(jnp.imag(Z_top(Xj)))
    YB = np.array(jnp.imag(Z_bot(Xj)))

    inside = (Y >= (YB[None, :] + delta)) & (Y <= (YT[None, :] - delta))

    def point_in_poly(xq, yq, polyx, polyy):
        xq_flat = xq.ravel()
        yq_flat = yq.ravel()
        inside_flat = np.zeros_like(xq_flat, dtype=bool)
        n = len(polyx)
        if n == 0:
            return inside_flat.reshape(xq.shape)
        j = n - 1
        for i in range(n):
            xi, yi = polyx[i], polyy[i]
            xj, yj = polyx[j], polyy[j]
            crossing = ((yi > yq_flat) != (yj > yq_flat)) & \
                       (xq_flat < (xj - xi) * (yq_flat - yi) / (yj - yi) + xi)
            inside_flat ^= crossing
            j = i
        return inside_flat.reshape(xq.shape)

    # block out particles using their true boundary shapes when available
    hole = np.zeros_like(X, dtype=bool)
    for ptidx in range(num_ptcl):
        ptx_here = ptx[(ptidx*N_ptcl):(ptidx+1)*N_ptcl]
        boundary = np.array(ptx_here)
        hole |= point_in_poly(X, Y, np.real(boundary), np.imag(boundary))
    for obsidx in range(num_obs):
        obsx_here = obsx[(obsidx*N_obs):(obsidx+1)*N_obs]
        boundary = np.array(obsx_here)
        hole |= point_in_poly(X, Y, np.real(boundary), np.imag(boundary))

    inside = inside & (~hole)

    tx_inside = (X[inside] + 1j*Y[inside]).astype(np.complex128)
    tx_jax = jnp.array(tx_inside)
    tnx_jax = jnp.ones_like(tx_jax) + 0j

    # u_tot, _ = evalsol_rbm(tx_jax, tnx_jax, sx, sxlo, sxhi, snx, sxp, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pwt, peri_len, mu, edens)
    u_tot, _ = evalsol_rbm_obs(tx_jax, tnx_jax, sx, sxlo, sxhi, snx, sxp, scur, sw, obsx, obsnx, obst, obsa, obsxp, obsw, obswxp, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pxp, pwt, peri_len, mu, edens)

    M = tx_inside.size
    ux = u_tot[:M]
    uy = u_tot[M:2*M]

    Ux = np.full_like(X, np.nan, dtype=float)
    Uy = np.full_like(Y, np.nan, dtype=float)
    Ux[inside] = np.real(ux)
    Uy[inside] = np.real(uy)

    plt.figure(figsize=(10, 3))
    speed = np.sqrt(Ux**2 + Uy**2)
    plt.pcolormesh(X, Y, speed, shading="auto")
    plt.streamplot(X, Y, Ux, Uy, density=density, linewidth=0.8, arrowsize=0.8)

    tt = np.linspace(0.0, peri_len, 800)
    zt = np.array(Z_top(jnp.array(tt)))
    zb = np.array(Z_bot(jnp.array(tt)))
    plt.plot(np.real(zt), np.imag(zt), 'k', lw=2)
    plt.plot(np.real(zb), np.imag(zb), 'k', lw=2)

    plt.axis("equal")
    plt.xlim(0, peri_len)
    plt.ylim(-1.0-ypad, 1.0+ypad)
    plt.colorbar(label="|u|")
    plt.title("Flow visualization, with close evaluation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

plot_streamlines_total(edens, density=4)