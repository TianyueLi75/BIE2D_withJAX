"""
A test script for setting up, solving, and evaluating in the extended linear system (ELS) 
for periodized 2d Stokes equations. 
Use dictionaries (pytree compatible in JAX) and partially jitted functions 
for a much leaner code.
"""

from jax import config
config.update("jax_enable_x64", True)

import time
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
def get_vslip(B1, B2, ptcl_coords, ptcl_tang):
    """
    ptcl_coords: of a single particle, in complex notation. 
    returns: vslip of a single particle, in stacked x then y.
    """
    xc = jnp.mean(ptcl_coords)
    xmxc = ptcl_coords - xc
    theta = jnp.atan2(jnp.imag(xmxc), jnp.real(xmxc))
    u_theta = B1 * jnp.sin(theta) + B2 * jnp.sin(theta) * jnp.cos(theta)
    if len(ptcl_tang) == 0:
        print("WARNING: no particle tangent given when computing vslip, will assume circle.")
        vslip = jnp.concatenate([-u_theta * jnp.sin(theta), u_theta * jnp.cos(theta)])
    else:
        vslip = jnp.concatenate([u_theta * jnp.real(ptcl_tang), u_theta * jnp.imag(ptcl_tang)])
    return vslip

# start_setup = time.perf_counter()

# Set discretization parameters
Np_wall = 10 # number of panels
p_wall = 10 # GL grid order on each panel
N_wall = Np_wall * p_wall # total number of discr. points on EACH wall
N_ptcl = 100 # total number of discr. points on EACH particle (global quadr)
N_obs = 100 # total number of global discr. points on EACH obstacle
N_side = 80
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
# vis(s['x'], s['nx'], True)

# Add particle 
num_ptcl = 2 # number of particles on the interior, for self eval.
ptcl_cell = {}
if num_ptcl > 0:
    Z_ptcl = lambda t : 1 + 0.2*jnp.cos(t-jnp.pi/7.) + 1j*(0.3*jnp.sin(t)+0.25)
    Zp_ptcl = lambda t : - 0.2*jnp.sin(t-jnp.pi/7.) + 1j*0.3*jnp.cos(t)
    Zpp_ptcl = lambda t : - 0.2*jnp.cos(t-jnp.pi/7.) - 1j*0.3*jnp.sin(t)
    # R_t = lambda t : 0.2+0.1*jnp.cos(5*t)
    # R_dt = lambda t : -0.5*jnp.sin(5*t)
    # R_ddt = lambda t : -2.5*jnp.cos(5*t)
    # Z_ptcl = lambda t : 1 + R_t(t)*jnp.cos(t) + 1j*(R_t(t)*jnp.sin(t)+0.25) # new test: non-circular
    # Zp_ptcl = lambda t : - R_t(t)*jnp.sin(t) + R_dt(t)*jnp.cos(t) + 1j* (R_t(t)*jnp.cos(t) + R_dt(t)*jnp.sin(t))
    # Zpp_ptcl = lambda t : R_ddt(t)*jnp.cos(t) - R_dt(t)*jnp.sin(t) - R_dt(t)*jnp.sin(t) - R_t(t)*jnp.cos(t) + \
    #                     1j* (R_ddt(t)*jnp.sin(t) + R_dt(t)*jnp.cos(t) + R_dt(t)*jnp.cos(t) - R_t(t)*jnp.sin(t))
    ptcl1 = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
    ptcl1['a'] = 1+0.25j
    ptcl1['radius'] = 0.3 # approximate, use larger for vis...
    ptcl_cell['ptcl_1'] = ptcl1

    if num_ptcl > 1:
        # Add another particle
        # Z_ptcl = lambda t : 5 + 0.2*jnp.cos(t) + 1j*(0.2*jnp.sin(t)+0.)
        # Zp_ptcl = lambda t : - 0.2*jnp.sin(t) + 1j*0.2*jnp.cos(t)
        # Zpp_ptcl = lambda t : - 0.2*jnp.cos(t) - 1j*0.2*jnp.sin(t)
        Z_ptcl = lambda t : 5 + 0.2*jnp.cos(t) + 1j*(0.1*jnp.sin(t)+0.)
        Zp_ptcl = lambda t : - 0.2*jnp.sin(t) + 1j*0.1*jnp.cos(t)
        Zpp_ptcl = lambda t : - 0.2*jnp.cos(t) - 1j*0.1*jnp.sin(t)
        ptcl2 = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
        ptcl2['a'] = 5+0j
        ptcl2['radius'] = 0.2
        ptcl_cell['ptcl_2'] = ptcl2

num_obs = 1
obs_cell = {}
if num_obs > 0:
    # Z_obs = lambda t : 3.6 + 0.24*jnp.cos(t) + 1j*(0.24*jnp.sin(t)-0.4)
    # Zp_obs = lambda t : -0.24*jnp.sin(t) + 1j*0.24*jnp.cos(t)
    # Zpp_obs = lambda t : -0.24*jnp.cos(t) - 1j*0.24*jnp.sin(t)
    Z_obs = lambda t : 3.6 + 0.2*jnp.cos(t-jnp.pi/8.) + 1j*(0.24*jnp.sin(t)-0.4) # new test: rotated ellipse
    Zp_obs = lambda t : -0.2*jnp.sin(t-jnp.pi/8.) + 1j*0.24*jnp.cos(t)
    Zpp_obs = lambda t : -0.2*jnp.cos(t-jnp.pi/8.) - 1j*0.24*jnp.sin(t)
    obs1 = channel_wall_func(Z_obs,N_obs,Zp_obs,Zpp_obs)
    obs1['a'] = 3.6-0.4j
    obs1['radius'] = 0.24
    obs_cell['obs_1'] = obs1

# [vis(x['x'], x['nx'], True) for x in ptcl_cell.values()]
# [vis(x['x'], x['nx'], True) for x in obs_cell.values()]

# Set up side walls 
L = side_wall(0., 2, N_side)
R = side_wall(0.+peri_len, 2, N_side)
L['x'] = L['x'] - 1j
R['x'] = R['x'] - 1j
# vis(L['x'], L['nx'], True)
# vis(R['x'], R['nx'], True)

# Set up proxy sources
Rp = 1.1 * peri_len
P = proxy(Rp, peri_len, N_prx)
# vis(P['x'], P['nx'], True, True)

mu = 0.7

# ========== TEST 1: Porous media flow, stationary obstacles -- uses ptcl_cell as PASSIVE particles (arbitrary) =========================
"""
# end_setup_half = time.perf_counter()
# print(f"TIMING RESULT: geometry set up: {end_setup_half - start_setup:.3g}.\n ")
# [E,A,B,_,_,C,Q] = ELSmatrix_wrapper(s, ptcl_cell, P, L, R, peri_len, mu) # Dummy
# Q.block_until_ready()
# start_setup = time.perf_counter()
[E,A,B,_,_,C,Q] = ELSmatrix_wrapper(s, ptcl_cell, {}, P, L, R, peri_len, mu)
# Q.block_until_ready()
# end_setup = time.perf_counter()

tx = jnp.array([2+0.2j,4+0.1j])
tnx = jnp.array([1+0j,1+0j])
trg = {'x': tx, 'nx': tnx}

# a) Exact solution
print("========= PASSIVE PARTICLES ========= ")
print('expt=t: running known Poisseuil flow BVP...')
# # Dummy run
# h=.2
# ue = lambda x: h*jnp.concatenate([jnp.imag(x)**2,jnp.zeros_like(jnp.imag(x))])
# pe = lambda x: h*2*mu*jnp.real(x)
# vrhs = ue(s['x'])
# vrhs_ptcl_list = [ue(pt['x']) for pt in ptcl_cell.values()]
# if vrhs_ptcl_list:
#     vrhs_ptcl = jnp.concatenate(vrhs_ptcl_list) 
# else:
#     vrhs_ptcl = jnp.array([])
# jump = pe(jnp.real(R['x'][0]))-pe(jnp.real(L['x'][0]))
# Tjump = -jump * jnp.concatenate([jnp.real(R['nx']), jnp.imag(R['nx'])])
# erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((2*N_side,)), Tjump]) 
# [edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15) 

# start_solve = time.perf_counter()
h=.2
ue = lambda x: h*jnp.concatenate([jnp.imag(x)**2,jnp.zeros_like(jnp.imag(x))])
pe = lambda x: h*2*mu*jnp.real(x)
vrhs = ue(s['x'])
vrhs_ptcl_list = [ue(pt['x']) for pt in ptcl_cell.values()]
if vrhs_ptcl_list:
    vrhs_ptcl = jnp.concatenate(vrhs_ptcl_list) 
else:
    vrhs_ptcl = jnp.array([])
jump = pe(jnp.real(R['x'][0]))-pe(jnp.real(L['x'][0]))
Tjump = -jump * jnp.concatenate([jnp.real(R['nx']), jnp.imag(R['nx'])])
erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((2*N_side,)), Tjump]) 

# Solve for density
[edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15) 
# end_solve = time.perf_counter()
# print(f'resid norm = {resid[0]:.3g}')
wall_dens = edens[:4*N_wall]
ptcl_dens = edens[4*N_wall:4*N_wall+2*num_ptcl*N_ptcl] # will just be a zero slice if num_ptcl = 0
prx_dens = edens[4*N_wall+2*num_ptcl*N_ptcl:]
wall_dens_norm = jnp.linalg.norm(wall_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
prx_dens_norm = jnp.linalg.norm(prx_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}')

# [ut, pt] = evalsol_wrapper(trg, s, ptcl_cell, P, peri_len, mu, edens)
# ut.block_until_ready()

# start_eval = time.perf_counter()
# Evaluate at target
# [ut, pt] = evalsol_wrapper(trg, s, ptcl_cell, P, peri_len, mu, edens)
[ut, pt] = evalsol_all(trg, s, ptcl_cell, {}, P, peri_len, mu, edens) # use ptcl_cell as PASSIVE particles right now.
# ut.block_until_ready()
# end_eval = time.perf_counter()

err = jnp.linalg.norm(ut - ue(tx)) # short form check -- matrix norm for velocity error, abs val for pressure
perr = jnp.abs(pt - pe(tx))
# print(perr.shape)
print(f'u velocity err at zt = {err:.3g}, p err at first trg: {perr[0]:.3g}, at second trg: {perr[1]:.3g}')
# print(f"TIMING RESULTS: Set up time: {end_setup-start_setup:.3g}, solve time: {end_solve - start_solve:.3g}, eval time: {end_eval - start_eval:.3g}.\n")


# b) No slip flow
print('expt=d: solving no-slip pressure-driven flow in pipe...')
# start_solve2 = time.perf_counter()
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
# end_solve2 = time.perf_counter()
# print(f'resid norm = {resid[0]:.3g}')
wall_dens = edens[:4*N_wall]
ptcl_dens = edens[4*N_wall:4*N_wall+2*num_ptcl*N_ptcl] # will just be a zero slice if num_ptcl = 0
prx_dens = edens[4*N_wall+2*num_ptcl*N_ptcl:]
wall_dens_norm = jnp.linalg.norm(wall_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
prx_dens_norm = jnp.linalg.norm(prx_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}')

# Evaluate at target
# start_eval2 = time.perf_counter()
# [ut, pt] = evalsol_wrapper(trg, s, ptcl_cell, P, peri_len, mu, edens)
[ut, pt] = evalsol_all(trg, s, ptcl_cell, {}, P, peri_len, mu, edens) 
# ut.block_until_ready()
# end_eval2 = time.perf_counter()

print('u velocity at zt = {:.12g}, {:.12g}, p at first trg = {:.12g}, at second = {:.12g}'.format(ut[0], ut[1], pt[0], pt[1]))
# print(f"TIMING RESULTS: solve time: {end_solve2 - start_solve2:.3g}, eval time: {end_eval2 - start_eval2:.3g}.\n")
"""

def plot_streamlines_total(edens, passive_cells, active_cells, Xc_list, r_list, nxg=140, ng=70, ypad=0.5, density=1.3, buffer_factor=1.0):
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
    for cell in list(passive_cells.values()) + list(active_cells.values()):
        if 'x' in cell and cell['x'].size > 0:
            boundary = np.array(cell['x'])
            hole |= point_in_poly(X, Y, np.real(boundary), np.imag(boundary))

    # fall back to circular masking if shape data is absent
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

    # u_tot, _ = evalsol_wrapper(trg_jax, s, ptcl_cell, P, peri_len, mu, edens)
    u_tot, _ = evalsol_all(trg_jax, s, passive_cells, active_cells, P, peri_len, mu, edens) 

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

Xc_list = jnp.array([])
r_list = jnp.array([])
# Xc_list = jnp.array([pt['a'] for pt in ptcl_cell.values()])
# r_list = jnp.array([pt['radius'] for pt in ptcl_cell.values()])
# plot_streamlines_total(edens, ptcl_cell, {}, Xc_list, r_list, density=4)



# =================== TEST 2: Active particles, rigid body motion, with obstacles ======================
# """
print("========= ACTIVE PARTICLES ========= ")
# [E,bc_gamma_mat,B,intF,intT,C,Q] = ELSmatrix_wrapper(s, obs_cell, ptcl_cell, P, L, R, peri_len, mu)
# Q.block_until_ready()

# start_setup3 = time.perf_counter()
# [E,bc_gamma_mat,B,intF,intT,C,Q] = ELSmatrix_wrapper(s, obs_cell, ptcl_cell, P, L, R, peri_len, mu)
[E,bc_gamma_mat,B,intF,intT,C,Q] = ELSmatrix_wrapper(s, obs_cell, ptcl_cell, P, L, R, peri_len, mu)
# Q.block_until_ready()
# end_setup3 = time.perf_counter()

tx = jnp.array([2+0.2j,4+0.1j])
tnx = jnp.array([1+0j,1+0j])
trg = {'x': tx, 'nx': tnx}

# B1 = 1.23; B2 = -0.73
# vrhs = jnp.zeros((s['x'].size*2,))
# pt_tang = pt['nx'] * 1j
# vslip_ptcl = [get_vslip(B1,B2,pt['x'],pt_tang) for pt in ptcl_cell.values()] 
# if vslip_ptcl:
#     vrhs_ptcl = jnp.concatenate(vslip_ptcl) 
# else:
#     vrhs_ptcl = jnp.array([])
# jump = 0
# Tjump = -jump * jnp.concatenate([jnp.real(R['nx']), jnp.imag(R['nx'])])
# erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((3*num_ptcl,)), jnp.zeros((2*N_side,)), Tjump]) 
# # Solve for density
# [edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15)

N_nodes_wall = len(s['x'])
N_nodes_ptcls = sum([len(pt['x']) for pt in ptcl_cell.values()])
N_nodes_obs = sum([len(obs['x']) for obs in obs_cell.values()])
# start_solve3 = time.perf_counter()
B1 = 1.23; B2 = -0.73
vrhs = jnp.zeros((N_nodes_wall*2+N_nodes_obs*2,))
vslip_ptcl = [get_vslip(B1,B2,pt['x'],pt['nx']*1j) for pt in ptcl_cell.values()] 
if vslip_ptcl:
    vrhs_ptcl = jnp.concatenate(vslip_ptcl) 
else:
    vrhs_ptcl = jnp.array([])
jump = 0
Tjump = -jump * jnp.concatenate([jnp.real(R['nx']), jnp.imag(R['nx'])])
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
# edens_rbm = jnp.concatenate([
#     edens[:2*(N_nodes_wall+N_nodes_ptcls)],
#     edens[2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl:]
# ])
# [ut, pt] = evalsol_wrapper(trg, s, ptcl_cell, P, peri_len, mu, edens_rbm)
# ut.block_until_ready()

# start_eval3 = time.perf_counter()
edens_rbm = jnp.concatenate([
    edens[:2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls)],
    edens[2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls)+3*num_ptcl:]
])
[ut, pt] = evalsol_all(trg, s, obs_cell, ptcl_cell, P, peri_len, mu, edens_rbm)
# ut.block_until_ready()
# end_eval3 = time.perf_counter()

print('u velocity at zt = {:.12g}, {:.12g}, p at first trg = {:.12g}, at second = {:.12g}'.format(ut[0], ut[1], pt[0], pt[1]))
# print(f"TIMING RESULTS: setup time: {end_setup3 - start_setup3:.3g}; solve time: {end_solve3 - start_solve3:.3g}, eval time: {end_eval3 - start_eval3:.3g}.\n")

# Xc_list = jnp.append(Xc_list, jnp.array([obs['a'] for obs in obs_cell.values()]))
# r_list = jnp.append(r_list, jnp.array([obs['radius'] for obs in obs_cell.values()]))
plot_streamlines_total(edens_rbm, obs_cell, ptcl_cell, Xc_list, r_list, density=4)
# """

# ====== INVESTIGATE SPEEDING UP SOLVE BY USING SCHUR COMPLEMENT
"""
# blocks of matrix E in rbm context
Asc = jnp.vstack([bc_gamma_mat, intF[:,:2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl],intT[:,:2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl]])
Bsc = jnp.vstack([B, intF[:, 2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl:],intT[:, 2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl:]])
Csc = jnp.hstack([C, jnp.zeros((C.shape[0],3*num_ptcl))])
Qsc = Q
vsc = erhs[:2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl]
gsc = erhs[2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl:]
[AinvB, _, _, _] = jnp.linalg.lstsq(Asc,Bsc,rcond=1e-15)
[Ainvv, _, _, _] = jnp.linalg.lstsq(Asc,vsc,rcond=1e-15)
S = Qsc - Csc @ AinvB
xi_rhs = gsc - Csc @ Ainvv
[xi,xi_resid,_,_] = jnp.linalg.lstsq(S, xi_rhs, rcond=1e-15)
sig_rhs = vsc - Bsc @ xi
[sig,sig_resid,_,_] = jnp.linalg.lstsq(Asc, sig_rhs, rcond=1e-15)

start_solve4 = time.perf_counter()
Asc = jnp.vstack([bc_gamma_mat, intF[:,:2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl],intT[:,:2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl]])
Bsc = jnp.vstack([B, intF[:, 2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl:],intT[:, 2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl:]])
Csc = jnp.hstack([C, jnp.zeros((C.shape[0],3*num_ptcl))])
Qsc = Q
vsc = erhs[:2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl]
gsc = erhs[2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl:]
[AinvB, _, _, _] = jnp.linalg.lstsq(Asc,Bsc,rcond=1e-15)
[Ainvv, _, _, _] = jnp.linalg.lstsq(Asc,vsc,rcond=1e-15)
S = Qsc - Csc @ AinvB
xi_rhs = gsc - Csc @ Ainvv
[xi,xi_resid,_,_] = jnp.linalg.lstsq(S, xi_rhs, rcond=1e-15)
sig_rhs = vsc - Bsc @ xi
[sig,sig_resid,_,_] = jnp.linalg.lstsq(Asc, sig_rhs, rcond=1e-15)
end_solve4 = time.perf_counter()

wall_dens = sig[:2*N_nodes_wall]
ptcl_dens = sig[2*N_nodes_wall:2*(N_nodes_wall+N_nodes_ptcls)] 
UOmega_all = sig[2*(N_nodes_wall+N_nodes_ptcls):2*(N_nodes_wall+N_nodes_ptcls)+3*num_ptcl]
prx_dens = xi
wall_dens_norm = jnp.linalg.norm(wall_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
prx_dens_norm = jnp.linalg.norm(prx_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}')
# TODO: print statement below just for 2 particle case for now..
print(f"Mobility problem for B1={B1},B2={B2}, is U1 = [{UOmega_all[0]:.3g},{UOmega_all[1]:.3g}], Omega1 = {UOmega_all[2]:.3g} U2 = [{UOmega_all[3]:.3g},{UOmega_all[4]:.3g}], Omega2 = {UOmega_all[5]:.3g}")

# Evaluate at target
start_eval4 = time.perf_counter()
edens_rbm = jnp.concatenate([
    sig[:2*(N_nodes_wall+N_nodes_ptcls)],
    xi
])
[ut, pt] = evalsol_wrapper(trg, s, ptcl_cell, P, peri_len, mu, edens_rbm)
ut.block_until_ready()
end_eval4 = time.perf_counter()

print('u velocity at zt = {:.12g}, {:.12g}, p at first trg = {:.12g}, at second = {:.12g}'.format(ut[0], ut[1], pt[0], pt[1]))
print(f"TIMING RESULTS: solve time: {end_solve4 - start_solve4:.3g}, eval time: {end_eval4 - start_eval4:.3g}.\n")
"""

# Xc_list = jnp.array([pt['a'] for pt in ptcl_cell.values()])
# r_list = jnp.array([pt['radius'] for pt in ptcl_cell.values()])
# plot_streamlines_total(edens_rbm, Xc_list, r_list, density=4)

