"""
A test script for setting up, solving, and evaluating in the extended linear system (ELS) 
for periodized 2d Stokes equations. 
Use dictionaries (pytree compatible in JAX) and partially jitted functions 
for a much leaner code.
"""

from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import sys, os
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the folder containing the module you want to import
utils_path = os.path.join(current_dir, '..')
peri_path = os.path.join(current_dir, '../periodic')
# Add the folder to the system path
sys.path.append(utils_path)
sys.path.append(peri_path)
from periodic.structure_pytree import *
from periodic.periodic_ELS_pytree import *

# jax.clear_caches()

@jit
def get_vslip(B1, B2, ptcl_coords):
    """
    ptcl_coords: of a single particle, in complex notation. 
    returns: vslip of a single particle, in stacked x then y.
    """
    xc = jnp.mean(ptcl_coords)
    xmxc = ptcl_coords - xc
    theta = jnp.atan2(jnp.imag(xmxc), jnp.real(xmxc))
    u_theta = B1 * jnp.sin(theta) + B2 * jnp.sin(theta) * jnp.cos(theta)
    vslip = jnp.concatenate([-u_theta * jnp.sin(theta), u_theta * jnp.cos(theta)])
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
U = channel_wall_glpanels(Z_top,Np_wall,p_wall,Zp_top,Zpp_top)
D = channel_wall_glpanels(Z_bot,Np_wall,p_wall,Zp_bot,Zpp_bot)
s = jax.tree_util.tree_map(lambda x,y: jnp.concatenate([x,y],axis=0), U, D)
vis(s['x'], s['nx'], True)

# Add particle 
num_ptcl = 2 # number of particles on the interior, for self eval.
ptcl_cell = {}
if num_ptcl > 0:
    Z_ptcl = lambda t : 1 + 0.3*jnp.cos(t) + 1j*(0.3*jnp.sin(t)+0.25)
    Zp_ptcl = lambda t : - 0.3*jnp.sin(t) + 1j*0.3*jnp.cos(t)
    Zpp_ptcl = lambda t : - 0.3*jnp.cos(t) - 1j*0.3*jnp.sin(t)
    ptcl1 = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
    ptcl1['a'] = 1+0.25j
    ptcl1['radius'] = 0.3
    ptcl_cell['1'] = ptcl1

    if num_ptcl > 1:
        # Add another particle
        Z_ptcl = lambda t : 5 + 0.2*jnp.cos(t) + 1j*(0.2*jnp.sin(t)+0.)
        Zp_ptcl = lambda t : - 0.2*jnp.sin(t) + 1j*0.2*jnp.cos(t)
        Zpp_ptcl = lambda t : - 0.2*jnp.cos(t) - 1j*0.2*jnp.sin(t)
        ptcl2 = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
        ptcl2['a'] = 5+0j
        ptcl2['radius'] = 0.2

        ptcl_cell['2'] = ptcl2
    
else:
    ptcl_cell = {}

[vis(x['x'], x['nx'], True) for x in ptcl_cell.values()]

# Set up side walls 
L = side_wall(0., 2, N_side)
R = side_wall(0.+peri_len, 2, N_side)
L['x'] = L['x'] - 1j
R['x'] = R['x'] - 1j
vis(L['x'], L['nx'], True)
vis(R['x'], R['nx'], True)

# Set up proxy sources
Rp = 1.1 * peri_len
P = proxy(Rp, peri_len, N_prx)
vis(P['x'], P['nx'], True)

mu = 0.7

# ========== TEST 1: Porous media flow, stationary obstacles =========================
[E,A,B,C,Q] = ELSmatrix_wrapper(s, ptcl_cell, P, L, R, peri_len, mu)

tx = jnp.array([2+0.2j,4+0.1j])
tnx = jnp.array([1+0j,1+0j])
trg = {'x': tx, 'nx': tnx}

# a) Exact solution
# TODO: this may not work with stokeslets inside, since "exact" flow doesn't have the same singularities at point sources.
print('expt=t: running known Poisseuil flow BVP...\n')
h=.2
ue = lambda x: h*jnp.concatenate([jnp.imag(x)**2,jnp.zeros_like(jnp.imag(x))])
pe = lambda x: h*2*mu*jnp.real(x)
vrhs = ue(s['x'])
vrhs_ptcl_list = [ue(pt['x']) for pt in ptcl_cell.values()]
if vrhs_ptcl_list:
    vrhs_ptcl = jnp.concatenate(vrhs_ptcl_list) # TODO: should this be a vstack? check size.
else:
    vrhs_ptcl = jnp.array([])
jump = pe(jnp.real(R['x'][0]))-pe(jnp.real(L['x'][0]))
Tjump = -jump * jnp.concatenate([jnp.real(R['nx']), jnp.imag(R['nx'])])
erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((2*N_side,)), Tjump]) 

# Solve for density
[edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15) 
print(f'resid norm = {resid[0]:.3g}\n')
wall_dens = edens[:4*N_wall]
ptcl_dens = edens[4*N_wall:4*N_wall+2*num_ptcl*N_ptcl] # will just be a zero slice if num_ptcl = 0
prx_dens = edens[4*N_wall+2*num_ptcl*N_ptcl:]
wall_dens_norm = jnp.linalg.norm(wall_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
prx_dens_norm = jnp.linalg.norm(prx_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}\n')

# Evaluate at target
[ut, pt] = evalsol_wrapper(trg, s, ptcl_cell, P, peri_len, mu, edens)

err = jnp.linalg.norm(ut - ue(tx)) # short form check -- matrix norm for velocity error, abs val for pressure
perr = jnp.abs(pt - pe(tx))
# print(perr.shape)
print(f'u velocity err at zt = {err:.3g}\n')
print(f'p err at first trg: {perr[0]:.3g}, at second trg: {perr[1]:.3g}')

# b) No slip flow
print('expt=d: solving no-slip pressure-driven flow in pipe...\n')
vrhs = jnp.zeros((s['x'].size*2,))
vrhs_ptcl_list = [jnp.zeros((pt['x'].size*2,)) for pt in ptcl_cell.values()]
if vrhs_ptcl_list:
    vrhs_ptcl = jnp.concatenate(vrhs_ptcl_list) # TODO: should this be a vstack? check size.
else:
    vrhs_ptcl = jnp.array([])
jump = -1
Tjump = -jump * jnp.concatenate([jnp.real(R['nx']), jnp.imag(R['nx'])])
erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((2*N_side,)), Tjump]) 

# Solve for density
[edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15)
print(f'resid norm = {resid[0]:.3g}\n')
wall_dens = edens[:4*N_wall]
ptcl_dens = edens[4*N_wall:4*N_wall+2*num_ptcl*N_ptcl] # will just be a zero slice if num_ptcl = 0
prx_dens = edens[4*N_wall+2*num_ptcl*N_ptcl:]
wall_dens_norm = jnp.linalg.norm(wall_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
prx_dens_norm = jnp.linalg.norm(prx_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}\n')

# Evaluate at target
[ut, pt] = evalsol_wrapper(trg, s, ptcl_cell, P, peri_len, mu, edens)

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
    trg_jax = {'x': tx_jax, 'nx': tnx_jax}

    u_tot, _ = evalsol_wrapper(trg_jax, s, ptcl_cell, P, peri_len, mu, edens)

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
    plt.title("Flow visualization, no close evaluation")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

Xc_list = jnp.array([pt['a'] for pt in ptcl_cell.values()])
r_list = jnp.array([pt['radius'] for pt in ptcl_cell.values()])
plot_streamlines_total(edens, Xc_list, r_list, density=4)



# =================== TEST 2: Active particles, rigid body motion ======================
[E,bc_gamma_mat,B,intF,intT,C,Q] = ELSmatrix_rbm(s, ptcl_cell, P, L, R, peri_len, mu)

tx = jnp.array([2+0.2j,4+0.1j])
tnx = jnp.array([1+0j,1+0j])
trg = {'x': tx, 'nx': tnx}

B1 = 1.23; B2 = -0.73
vrhs = jnp.zeros((s['x'].size*2,))
vslip_ptcl = [get_vslip(B1,B2,pt['x']) for pt in ptcl_cell.values()] 
if vslip_ptcl:
    vrhs_ptcl = jnp.concatenate(vslip_ptcl) 
else:
    vrhs_ptcl = jnp.array([])
jump = 0
Tjump = -jump * jnp.concatenate([jnp.real(R['nx']), jnp.imag(R['nx'])])
erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((3*num_ptcl,)), jnp.zeros((2*N_side,)), Tjump]) 

# Solve for density
[edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15)
print(f'resid norm = {resid[0]:.3g}\n')
N_nodes_wall = len(s['x'])
N_nodes_ptcls = sum([len(pt['x']) for pt in ptcl_cell.values()])
wall_dens = edens[:2*N_nodes_wall]
ptcl_dens = edens[2*N_nodes_wall:2*(N_nodes_wall+N_nodes_ptcls)] 
UOmega_all = edens[2*(N_nodes_wall+N_nodes_ptcls):2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl]
prx_dens = edens[2*(N_nodes_wall+N_nodes_ptcls) + 3*num_ptcl:]
wall_dens_norm = jnp.linalg.norm(wall_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
prx_dens_norm = jnp.linalg.norm(prx_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}\n')
# TODO: print statement below just for 2 particle case for now..
print(f"Mobility problem for B1={B1},B2={B2}, is U1 = [{UOmega_all[0]:.3g},{UOmega_all[1]:.3g}], Omega1 = {UOmega_all[2]:.3g} U2 = [{UOmega_all[3]:.3g},{UOmega_all[4]:.3g}], Omega2 = {UOmega_all[5]:.3g}")

# Evaluate at target
edens_rbm = jnp.concatenate([
    edens[:2*(N_nodes_wall+N_nodes_ptcls)],
    edens[2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl:]
])
[ut, pt] = evalsol_wrapper(trg, s, ptcl_cell, P, peri_len, mu, edens_rbm)

print('u velocity at zt = {:.12g}, {:.12g}, p at first trg = {:.12g}, at second = {:.12g}'.format(ut[0], ut[1], pt[0], pt[1]))

Xc_list = jnp.array([pt['a'] for pt in ptcl_cell.values()])
r_list = jnp.array([pt['radius'] for pt in ptcl_cell.values()])
plot_streamlines_total(edens_rbm, Xc_list, r_list, density=4)

