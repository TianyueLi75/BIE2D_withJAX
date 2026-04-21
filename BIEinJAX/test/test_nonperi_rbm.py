
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
from nonperiodic.BIOsolve_pytree import *

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
N_obs = 80 # total number of global discr. points on EACH obstacle

mu = 0.7

# Set up container
R_container = 1.
Z_container = lambda t : R_container * jnp.cos(t) + 1j * R_container * jnp.sin(t)
Zp_container = lambda t : -R_container * jnp.sin(t) + 1j * R_container * jnp.cos(t)
Zpp_container = lambda t : -R_container * jnp.cos(t) - 1j * R_container * jnp.sin(t)
s = channel_wall_glpanels(Z_container,Np_wall,p_wall,Zp_container,Zpp_container)
# vis(s['x'],s['nx'], True)

# Add particle 
num_ptcl = 2 # number of particles on the interior, for self eval.
ptcl_cell = {}
if num_ptcl > 0:
    theta1 = 0.
    a1 = 0.12; b1 = 0.1 # radii
    c1 = 0; d1 = 0.15 # center
    Z_ptcl = lambda t : a1*jnp.cos(t)*jnp.cos(theta1) - b1*jnp.sin(t)*jnp.sin(theta1) + c1 \
                + 1j * (a1*jnp.cos(t)*jnp.sin(theta1) + b1*jnp.sin(t)*jnp.cos(theta1) + d1)
    Zp_ptcl = lambda t : -a1*jnp.sin(t)*jnp.cos(theta1) - b1*jnp.cos(t)*jnp.sin(theta1) \
                + 1j * (-a1*jnp.sin(t)*jnp.sin(theta1) + b1*jnp.cos(t)*jnp.cos(theta1))
    Zpp_ptcl = lambda t : -a1*jnp.cos(t)*jnp.cos(theta1) + b1*jnp.sin(t)*jnp.sin(theta1) \
                + 1j * (-a1*jnp.cos(t)*jnp.sin(theta1) - b1*jnp.sin(t)*jnp.cos(theta1))
    ptcl1 = channel_wall_func(Z_ptcl,N_ptcl,Zp_ptcl, Zpp_ptcl)
    ptcl1['a'] = c1+d1*1j
    ptcl1['theta0'] = theta1
    ptcl1['radius'] = a1 # approximate, use larger for vis...
    ptcl_cell['ptcl_1'] = ptcl1

    if num_ptcl > 1:
        theta2 = 0.
        a2 = 0.12; b2 = 0.1 # radii
        c2 = 0; d2 = -0.15 # center
        Z_ptcl2 = lambda t : a2*jnp.cos(t)*jnp.cos(theta2) - b2*jnp.sin(t)*jnp.sin(theta2) + c2 \
                    + 1j * (a2*jnp.cos(t)*jnp.sin(theta2) + b2*jnp.sin(t)*jnp.cos(theta2) + d2)
        Zp_ptcl2 = lambda t : -a2*jnp.sin(t)*jnp.cos(theta2) - b2*jnp.cos(t)*jnp.sin(theta2) \
                    + 1j * (-a2*jnp.sin(t)*jnp.sin(theta2) + b2*jnp.cos(t)*jnp.cos(theta2))
        Zpp_ptcl2 = lambda t : -a2*jnp.cos(t)*jnp.cos(theta2) + b2*jnp.sin(t)*jnp.sin(theta2) \
                    + 1j * (-a2*jnp.cos(t)*jnp.sin(theta2) - b2*jnp.sin(t)*jnp.cos(theta2))
        ptcl2 = channel_wall_func(Z_ptcl2,N_ptcl,Zp_ptcl2, Zpp_ptcl2)
        ptcl2['a'] = c2+d2*1j
        ptcl2['theta0'] = theta2
        ptcl2['radius'] = a2 # approximate, use larger for vis...
        ptcl_cell['ptcl_2'] = ptcl2

num_obs = 0
obs_cell = {}
if num_obs > 0:
    theta1 = 0.
    a1 = 0.1; b1 = 0.1 # radii
    c1 = 0.35; d1 = 0 # center
    Z_obs = lambda t : a1*jnp.cos(t)*jnp.cos(theta1) - b1*jnp.sin(t)*jnp.sin(theta1) + c1 \
                + 1j * (a1*jnp.cos(t)*jnp.sin(theta1) + b1*jnp.sin(t)*jnp.cos(theta1) + d1)
    Zp_obs = lambda t : -a1*jnp.sin(t)*jnp.cos(theta1) - b1*jnp.cos(t)*jnp.sin(theta1) \
                + 1j * (-a1*jnp.sin(t)*jnp.sin(theta1) + b1*jnp.cos(t)*jnp.cos(theta1))
    Zpp_obs = lambda t : -a1*jnp.cos(t)*jnp.cos(theta1) + b1*jnp.sin(t)*jnp.sin(theta1) \
                + 1j * (-a1*jnp.cos(t)*jnp.sin(theta1) - b1*jnp.sin(t)*jnp.cos(theta1))
    obs1 = channel_wall_func(Z_obs,N_obs,Zp_obs, Zpp_obs)
    obs1['a'] = c1+d1*1j
    obs1['theta0'] = theta1
    obs1['radius'] = a1 # approximate, use larger for vis...
    obs_cell['obs_1'] = obs1

# [vis(x['x'], x['nx'], True) for x in ptcl_cell.values()]
# [vis(x['x'], x['nx'], True) for x in obs_cell.values()]

def plot_streamlines_total(edens, passive_cells, active_cells, Xc_list, r_list, nxg=140, ng=140, ypad=0.5, density=1.3, buffer_factor=1.0):
    xg = np.linspace(-R_container - ypad, R_container + ypad, nxg)
    yg = np.linspace(-R_container - ypad, R_container + ypad, ng)
    X, Y = np.meshgrid(xg, yg)

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
    
    # Consider only points inside container
    boundary_container = np.array(s['x'])
    inside = point_in_poly(X,Y,np.real(boundary_container),np.imag(boundary_container))

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
    u_tot, _ = evalsol_all(trg_jax, s, passive_cells, active_cells, mu, edens) 

    M = tx_inside.size
    ux = u_tot[:M]
    uy = u_tot[M:2*M]

    Ux = np.full_like(X, np.nan, dtype=float)
    Uy = np.full_like(Y, np.nan, dtype=float)
    Ux[inside] = np.real(ux)
    Uy[inside] = np.real(uy)

    plt.figure(figsize=(5, 5))
    speed = np.sqrt(Ux**2 + Uy**2)
    plt.pcolormesh(X, Y, speed, shading="auto")
    plt.streamplot(X, Y, Ux, Uy, density=density, linewidth=0.8, arrowsize=0.8)

    plt.plot(np.real(s['x']), np.imag(s['x']), 'k', lw=2)
    [plt.plot(np.real(pt['x']), np.imag(pt['x']), 'k', lw=2) for pt in ptcl_cell.values()]
    [plt.plot(np.real(pt['x']), np.imag(pt['x']), 'k', lw=2) for pt in obs_cell.values()]

    plt.axis("equal")
    plt.xlim(-R_container - ypad, R_container + ypad)
    plt.ylim(-R_container-ypad, R_container+ypad)
    plt.colorbar(label="|u|")
    plt.title("Flow visualization")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

Xc_list = jnp.array([])
r_list = jnp.array([])

print("========= ACTIVE PARTICLES ========= ")
[E,bc_gamma_mat,intF,intT] = rbm_wrapper(s, obs_cell, ptcl_cell, mu)

tx = jnp.array([R_container*0.5 + R_container*0.5j,-R_container*0.3 -R_container*0.3j])
tnx = jnp.array([1+0j,1+0j])
trg = {'x': tx, 'nx': tnx}

N_nodes_wall = len(s['x'])
N_nodes_ptcls = sum([len(pt['x']) for pt in ptcl_cell.values()])
N_nodes_obs = sum([len(obs['x']) for obs in obs_cell.values()])
# start_solve3 = time.perf_counter()
B1 = 1.23; B2 = -0.73
# B1 = 1.0; B2 = 1.0
vrhs = jnp.zeros((N_nodes_wall*2+N_nodes_obs*2,))
vslip_ptcl = [get_vslip(B1,B2,pt['x'],pt['nx']*1j) for pt in ptcl_cell.values()] 
if vslip_ptcl:
    vrhs_ptcl = jnp.concatenate(vslip_ptcl) 
else:
    vrhs_ptcl = jnp.array([])
erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((3*num_ptcl,))]) 

# Solve for density
[edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15)
print(f'resid norm = {resid[0]:.3g}')
wall_dens = edens[:2*N_nodes_wall]
obs_dens = edens[2*N_nodes_wall:2*(N_nodes_wall+N_nodes_obs)] 
ptcl_dens = edens[2*(N_nodes_wall+N_nodes_obs):2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls)] 
UOmega_all = edens[2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls):2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls)+3*num_ptcl]
wall_dens_norm = jnp.linalg.norm(wall_dens)
obs_dens_norm = jnp.linalg.norm(obs_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of obstacle density = {obs_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}')
jax.debug.print("UOmega: {}",UOmega_all)

edens_rbm = edens[:2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls)]
[ut, pt] = evalsol_all(trg, s, obs_cell, ptcl_cell, mu, edens_rbm)

jax.debug.print("ut = {a}, pt = {b}", a=ut, b=pt)

plot_streamlines_total(edens_rbm, obs_cell, ptcl_cell, Xc_list, r_list, density=4)