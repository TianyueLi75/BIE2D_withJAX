# A test script to solve 2D Stokes periodic rigid body mobility problem
# Choco Li Mar 16, 2026

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

@jit
def get_vslip(B1, B2, ptcl_coords_2d):
    """
    ptcl_coords: Array of shape (num_particles, num_nodes) 
                 containing complex coordinates.
    """
    xc = jnp.mean(ptcl_coords_2d, axis=1, keepdims=True)
    xmxc_tot = (ptcl_coords_2d - xc).reshape(-1)
    theta = jnp.atan2(jnp.imag(xmxc_tot), jnp.real(xmxc_tot))
    u_theta = B1 * jnp.sin(theta) + B2 * jnp.sin(theta) * jnp.cos(theta)
    vslip = -u_theta * jnp.sin(theta) + 1j*u_theta * jnp.cos(theta)
    return vslip


# Set discretization parameters
Np_wall = 10 # number of panels
p_wall = 10 # GL grid order on each panel
N_wall = Np_wall * p_wall # total number of discr. points on EACH wall
N_ptcl = 80 # total number of discr. points on EACH particle (global quadr)
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
vis(sx, snx, True)

# Add particle 
num_ptcl = 2 # number of particles on the interior, for self eval.
Z_ptcl = lambda t : 1 + 0.3*jnp.cos(t) + 1j*(0.3*jnp.sin(t)+0.25)
Zp_ptcl = lambda t : - 0.3*jnp.sin(t) + 1j*0.3*jnp.cos(t)
Zpp_ptcl = lambda t : - 0.3*jnp.cos(t) - 1j*0.3*jnp.sin(t)
[ptx,ptxp,ptnx,ptcur,ptw] = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
ptwxp = 2*jnp.pi/N_ptcl * ptxp
ptt = jnp.linspace(0, 2 * jnp.pi, N_ptcl, endpoint=False)
pta = jnp.array([1+0.25j])
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
vis(ptx, ptnx, True)
Xc_list = pta
r_list = jnp.array([0.3, 0.2])

# Set up side walls 
[lx,lnx] = side_wall(0., 2, N_side)
[rx,rnx] = side_wall(0.+peri_len, 2, N_side)
lx = lx - 1j
rx = rx - 1j
vis(lx, lnx, True)
vis(rx, rnx, True)

# Set up proxy sources
R = 1.1 * peri_len
[px,pxp,pnx,pwt] = proxy(R, peri_len, N_prx)
vis(px, pnx, True)

mu = 0.7

# Make ELS matrix
# TODO: debug ELS matrix
[E,bc_gamma_mat,B,intF,intT,C,Q] = ELSmatrix_rbm(sx, snx, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pwt, lx, lnx, rx, rnx, peri_len, mu)

tx = jnp.array([2+0.2j,4+0.1j])
tnx = jnp.array([1+0j,1+0j])
# tx = jnp.array([2+0.2j])
# tnx = jnp.array([1+0j])

# Slip velocity parameters
B1 = 1.23; B2 = -0.73
vrhs = jnp.zeros((sx.size*2,))
ptx_2d = ptx.reshape((num_ptcl,N_ptcl))
vrhs_ptcl_cpx = get_vslip(B1,B2,ptx_2d)
vrhs_ptcl = jnp.concatenate([jnp.real(vrhs_ptcl_cpx),jnp.imag(vrhs_ptcl_cpx)])
jump = 0
Tjump = -jump * jnp.concatenate([jnp.real(rnx), jnp.imag(rnx)])
erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((3*num_ptcl,)), jnp.zeros((2*N_side,)), Tjump])

# Solve for density
[edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15)
print(f'resid norm = {resid[0]:.3g}\n')
wall_dens = edens[:4*N_wall]
ptcl_dens = edens[4*N_wall:4*N_wall+2*num_ptcl*N_ptcl] # will just be a zero slice if num_ptcl = 0
U = edens[4*N_wall+2*num_ptcl*N_ptcl:4*N_wall+2*num_ptcl*N_ptcl+2*num_ptcl]
Omega = edens[4*N_wall+2*num_ptcl*N_ptcl+2*num_ptcl:4*N_wall+2*num_ptcl*N_ptcl+3*num_ptcl]
prx_dens = edens[4*N_wall+2*num_ptcl*N_ptcl+3*num_ptcl:]
wall_dens_norm = jnp.linalg.norm(wall_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
prx_dens_norm = jnp.linalg.norm(prx_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}')
print(f"Mobility problem for B1={B1},B2={B2}, is U1 = [{U[0]:.3g},{U[1]:.3g}], U2 = [{U[2]:.3g},{U[3]:.3g}], Omega1 = {Omega[0]:.3g}, Omega2 = {Omega[1]:.3g}")

# Evaluate at target
[ut, pt] = evalsol_rbm(tx, tnx, sx, sxlo, sxhi, snx, sxp, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pwt, peri_len, mu, edens)
print('u velocity at zt = {:.12g}, {:.12g}, p at first trg = {:.12g}, at second = {:.12g}'.format(ut[0], ut[1], pt[0], pt[1]))

def plot_streamlines_total(edens, Xc_list, r_list, nxg=140, ng=70, ypad=0.5, density=1.3, buffer_factor=1.0):
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

    # block out each particle (hardcoded to be circles for now)
    hole = np.zeros_like(X, dtype=bool)
    for ind in range(len(Xc_list)):
        z = Xc_list[ind]
        r = r_list[ind]
        X0, Y0 = float(np.real(z)), float(np.imag(z))
        hole |= (X - X0)**2 + (Y - Y0)**2 <= r**2
    inside = inside & (~hole)

    tx_inside = (X[inside] + 1j*Y[inside]).astype(np.complex128)
    tx_jax = jnp.array(tx_inside)
    tnx_jax = jnp.ones_like(tx_jax) + 0j

    u_tot, _ = evalsol_rbm(tx_jax, tnx_jax, sx, sxlo, sxhi, snx, sxp, scur, sw, ptx, ptnx, ptt, pta, ptxp, ptw, ptwxp, px, pwt, peri_len, mu, edens)

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

plot_streamlines_total(edens, Xc_list, r_list, density=4)