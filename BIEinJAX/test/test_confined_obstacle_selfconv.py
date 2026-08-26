"""Self-convergence test for the confined obstacle-course geometry.

Initial static geometry: one slip swimmer + 4 elongated no-slip ellipse
obstacles inside a no-slip circular container (same setup as
test_confined_obstacle_course.py, before any time evolution).

We solve the slip-on-swimmer / no-slip-elsewhere mobility problem at a sequence
of increasing discretization node counts, evaluate the flow velocity at a fixed
cloud of target points spread uniformly through the domain BUT held moderately
far from every body (so we probe smooth-quadrature convergence, not near-eval),
and measure the max relative error of the velocity field against the finest
("self") solution. Also tracks the swimmer rigid-body velocity/spin error.
"""

from jax import config
config.update("jax_enable_x64", True)

import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '../periodic'))
from periodic.structure_pytree import *
from nonperiodic.BIOsolve_pytree import *


# ---------------------------------------------------------------------------
# Helpers (mirrors test_confined_obstacle_course.py)
# ---------------------------------------------------------------------------
@jit
def get_vslip(B1, B2, ptcl_coords, ptcl_tang):
    xc = jnp.mean(ptcl_coords)
    xmxc = ptcl_coords - xc
    theta = jnp.atan2(jnp.imag(xmxc), jnp.real(xmxc))
    u_theta = B1 * jnp.sin(theta) + B2 * jnp.sin(theta) * jnp.cos(theta)
    if len(ptcl_tang) == 0:
        vslip = jnp.concatenate([-u_theta * jnp.sin(theta), u_theta * jnp.cos(theta)])
    else:
        vslip = jnp.concatenate([u_theta * jnp.real(ptcl_tang), u_theta * jnp.imag(ptcl_tang)])
    return vslip

def rotated_ellipse(a, b, theta, cx, cy):
    Z = lambda t : a*jnp.cos(t)*jnp.cos(theta) - b*jnp.sin(t)*jnp.sin(theta) + cx \
                + 1j * (a*jnp.cos(t)*jnp.sin(theta) + b*jnp.sin(t)*jnp.cos(theta) + cy)
    Zp = lambda t : -a*jnp.sin(t)*jnp.cos(theta) - b*jnp.cos(t)*jnp.sin(theta) \
                + 1j * (-a*jnp.sin(t)*jnp.sin(theta) + b*jnp.cos(t)*jnp.cos(theta))
    Zpp = lambda t : -a*jnp.cos(t)*jnp.cos(theta) + b*jnp.sin(t)*jnp.sin(theta) \
                + 1j * (-a*jnp.cos(t)*jnp.sin(theta) - b*jnp.sin(t)*jnp.cos(theta))
    return Z, Zp, Zpp

def build_ellipse_body(a, b, theta, cx, cy, N):
    Z, Zp, Zpp = rotated_ellipse(a, b, theta, cx, cy)
    body = channel_wall_func(Z, N, Zp, Zpp)
    body['a'] = cx + cy*1j
    body['theta0'] = theta
    body['radius'] = max(a, b)
    return body

def solve_rbm(s, obs_cell, ptcl_cell, mu, B1, B2, static=None):
    """Slip on swimmer, no-slip on wall + obstacles, force/torque-free swimmer."""
    # `static` is the cached wall+obstacle block: the wall and the obstacles never
    # move, so it is built once outside the time loop and reused every step.
    [E, _, _, _, _] = rbm_wrapper(s, obs_cell, ptcl_cell, mu, static)
    N_nodes_wall = len(s['x'])
    N_nodes_ptcls = sum(len(pt['x']) for pt in ptcl_cell.values())
    N_nodes_obs = sum(len(o['x']) for o in obs_cell.values())
    n_ptcl = len(ptcl_cell)
    vrhs = jnp.zeros((N_nodes_wall*2 + N_nodes_obs*2,))
    vslip_ptcl = [get_vslip(B1, B2, pt['x'], pt['nx']*1j) for pt in ptcl_cell.values()]
    vrhs_ptcl = jnp.concatenate(vslip_ptcl) if vslip_ptcl else jnp.array([])
    erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((3*n_ptcl,))])
    [edens, resid, _, _] = jnp.linalg.lstsq(E, erhs, rcond=1e-15)
    base = 2*(N_nodes_wall + N_nodes_obs + N_nodes_ptcls)
    UOmega = edens[base:base + 3*n_ptcl]
    edens_rbm = edens[:base]
    U = np.array([float(UOmega[0]), float(UOmega[1])])
    Omega = float(UOmega[2])
    resid_val = float(resid[0]) if resid.size > 0 else float('nan')
    return U, Omega, edens_rbm, resid_val


# ---------------------------------------------------------------------------
# Fixed geometry -- the INITIAL obstacle-course layout (swimmer near center)
# ---------------------------------------------------------------------------
mu = 0.7
p_wall = 10                       # GL order per container panel (fixed)
R_container = 1.

Z_container   = lambda t :  R_container*jnp.cos(t) + 1j*R_container*jnp.sin(t)
Zp_container  = lambda t : -R_container*jnp.sin(t) + 1j*R_container*jnp.cos(t)
Zpp_container = lambda t : -R_container*jnp.cos(t) - 1j*R_container*jnp.sin(t)

# swimmer (active, slip) -- initial position, clear of everything
a_s, b_s, theta_s, cx_s, cy_s = 0.12, 0.10, 0.0, 0.0, 0.02
swimmer_size = max(a_s, b_s)

# obstacles (passive, no-slip): 4 elongated ellipses at distinct orientations
obs_specs = [
    (0.16, 0.055, 0.60,  -0.46, -0.34),
    (0.16, 0.055, -0.50,  0.43, -0.35),
    (0.16, 0.055, 1.25,  -0.48,  0.36),
    (0.16, 0.055, 2.35,   0.42,  0.40),
]

B1, B2 = 1.23, -0.73              # optimal squirmer slip


def build_geometry(N_body, Np_wall):
    """Rebuild the whole scene at a chosen resolution (N_body nodes on each
    body; Np_wall panels of order p_wall on the container)."""
    s = channel_wall_glpanels(Z_container, Np_wall, p_wall, Zp_container, Zpp_container)
    ptcl_cell = {'ptcl_1': build_ellipse_body(a_s, b_s, theta_s, cx_s, cy_s, N_body)}
    obs_cell = {}
    for i, spec in enumerate(obs_specs):
        obs_cell[f'obs_{i+1}'] = build_ellipse_body(*spec, N_body)
    return s, obs_cell, ptcl_cell


# ---------------------------------------------------------------------------
# Fixed target cloud: uniform grid inside the container, kept `margin` away
# from the wall and from every body's bounding circle (moderately-far regime).
# ---------------------------------------------------------------------------
def make_targets(margin=0.15, ng=26):
    g = np.linspace(-R_container, R_container, ng)
    X, Y = np.meshgrid(g, g)
    Z = (X + 1j*Y).ravel()
    keep = np.abs(Z) < (R_container - margin)                 # away from wall
    keep &= np.abs(Z - (cx_s + 1j*cy_s)) > (swimmer_size + margin)  # away from swimmer
    for (a, b, th, cx, cy) in obs_specs:
        keep &= np.abs(Z - (cx + 1j*cy)) > (max(a, b) + margin)     # away from obstacle
    return Z[keep]


def plot_targets(targets, fname='confined_obstacle_selfconv_targets.png'):
    s, obs_cell, ptcl_cell = build_geometry(100, 10)
    plt.figure(figsize=(5, 5))
    plt.plot(np.real(s['x']), np.imag(s['x']), 'k', lw=2)
    for obs in obs_cell.values():
        plt.fill(np.real(obs['x']), np.imag(obs['x']), color='0.6', ec='k')
    sw = ptcl_cell['ptcl_1']
    plt.fill(np.real(sw['x']), np.imag(sw['x']), color='tab:red', ec='k')
    plt.plot(np.real(targets), np.imag(targets), '.', color='tab:blue', ms=4,
             label=f'{targets.size} targets')
    plt.axis('equal'); plt.legend(loc='lower right', fontsize=8)
    plt.title('Self-convergence targets (moderately far from all bodies)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'saved {fname}')


# ---------------------------------------------------------------------------
# Convergence sweep
# ---------------------------------------------------------------------------
N_list = [40, 60, 90, 120, 160, 200]   # nodes per body; finest = self-reference
targets = make_targets(margin=0.15, ng=26)
print(f"number of far-field targets: {targets.size}")
plot_targets(targets)

trg = {'x': jnp.array(targets.astype(np.complex128)),
       'nx': jnp.ones(targets.size, dtype=jnp.complex128)}

vel = []          # complex velocity w = ux + i*uy at each target, per level
UOm = []          # [Ux, Uy, Omega] per level
ntot = []         # total boundary nodes per level

print("\n========= SELF-CONVERGENCE SWEEP =========")
for N in N_list:
    Np_wall = max(4, int(round(N / p_wall)))     # ~N wall nodes, scales with N
    s, obs_cell, ptcl_cell = build_geometry(N, Np_wall)
    N_wall = len(s['x'])
    n_bodies_nodes = N_wall + sum(len(o['x']) for o in obs_cell.values()) \
                     + sum(len(pt['x']) for pt in ptcl_cell.values())
    t0 = time.time()
    U, Omega, edens_rbm, resid = solve_rbm(s, obs_cell, ptcl_cell, mu, B1, B2)
    u, _ = evalsol_all(trg, s, obs_cell, ptcl_cell, mu, edens_rbm)
    u = np.asarray(u)
    M = targets.size
    w = np.real(u[:M]) + 1j*np.real(u[M:2*M])
    vel.append(w)
    UOm.append(np.array([U[0], U[1], Omega]))
    ntot.append(n_bodies_nodes)
    print(f"  N_body={N:4d}  N_wall={N_wall:4d}  N_tot={n_bodies_nodes:5d}  "
          f"U=[{U[0]:+.5f},{U[1]:+.5f}] Om={Omega:+.5f}  resid={resid:.1e}  "
          f"({time.time()-t0:.1f}s)")

# finest = reference
w_ref = vel[-1]
UOm_ref = UOm[-1]
vscale = np.max(np.abs(w_ref))           # sup-norm of the reference velocity field
uoscale = np.linalg.norm(UOm_ref)

err_vel = []
err_uo = []
for k in range(len(N_list) - 1):
    err_vel.append(np.max(np.abs(vel[k] - w_ref)) / vscale)     # relative sup-norm
    err_uo.append(np.linalg.norm(UOm[k] - UOm_ref) / uoscale)

print("\n--- max relative velocity error vs finest (N_body=%d) ---" % N_list[-1])
for k in range(len(N_list) - 1):
    print(f"  N_body={N_list[k]:4d}  N_tot={ntot[k]:5d}  "
          f"max_rel_vel_err={err_vel[k]:.3e}  rel_UOmega_err={err_uo[k]:.3e}")


# ---------------------------------------------------------------------------
# Plot convergence
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 4.5))
plt.semilogy(ntot[:-1], err_vel, 'o-', label='max rel. velocity error (far targets)')
plt.semilogy(ntot[:-1], err_uo, 's--', label='rel. swimmer $[U_x,U_y,\\Omega]$ error')
plt.xlabel('total boundary discretization nodes $N_{tot}$')
plt.ylabel('relative error vs finest solution')
plt.title('Confined obstacle course: self-convergence')
plt.grid(True, which='both', ls=':', alpha=0.6)
plt.legend(fontsize=9)
plt.savefig('confined_obstacle_selfconv.png', dpi=150, bbox_inches='tight')
print('\nsaved confined_obstacle_selfconv.png')
