
from jax import config
config.update("jax_enable_x64", True)

import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
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

def rotated_ellipse(a, b, theta, cx, cy):
    """Return (Z, Zp, Zpp) complex parametrizations of a rotated ellipse."""
    Z = lambda t : a*jnp.cos(t)*jnp.cos(theta) - b*jnp.sin(t)*jnp.sin(theta) + cx \
                + 1j * (a*jnp.cos(t)*jnp.sin(theta) + b*jnp.sin(t)*jnp.cos(theta) + cy)
    Zp = lambda t : -a*jnp.sin(t)*jnp.cos(theta) - b*jnp.cos(t)*jnp.sin(theta) \
                + 1j * (-a*jnp.sin(t)*jnp.sin(theta) + b*jnp.cos(t)*jnp.cos(theta))
    Zpp = lambda t : -a*jnp.cos(t)*jnp.cos(theta) + b*jnp.sin(t)*jnp.sin(theta) \
                + 1j * (-a*jnp.cos(t)*jnp.sin(theta) - b*jnp.sin(t)*jnp.cos(theta))
    return Z, Zp, Zpp

def build_ellipse_body(a, b, theta, cx, cy, N):
    """Discretize a rotated ellipse as a body dict (swimmer or obstacle)."""
    Z, Zp, Zpp = rotated_ellipse(a, b, theta, cx, cy)
    body = channel_wall_func(Z, N, Zp, Zpp)
    body['a'] = cx + cy*1j
    body['theta0'] = theta
    body['radius'] = max(a, b) # approximate bounding radius, use for broad-phase / vis
    return body

# ---------------------------------------------------------------------------
# Discretization parameters
# ---------------------------------------------------------------------------
Np_wall = 10 # number of panels on the container wall
p_wall = 10  # GL grid order on each panel
N_wall = Np_wall * p_wall # total number of discr. points on the wall
N_ptcl = 100 # total number of global discr. points on the swimmer
N_obs = 120  # total number of global discr. points on EACH obstacle (elongated -> more nodes)

mu = 0.7

# ---------------------------------------------------------------------------
# Container (outer confinement circle) -- no-slip
# ---------------------------------------------------------------------------
R_container = 1.
Z_container = lambda t : R_container * jnp.cos(t) + 1j * R_container * jnp.sin(t)
Zp_container = lambda t : -R_container * jnp.sin(t) + 1j * R_container * jnp.cos(t)
Zpp_container = lambda t : -R_container * jnp.cos(t) - 1j * R_container * jnp.sin(t)
s = channel_wall_glpanels(Z_container, Np_wall, p_wall, Zp_container, Zpp_container)

# ---------------------------------------------------------------------------
# Swimmer (one active particle) -- slip velocity, placed clear of everything
# ---------------------------------------------------------------------------
num_ptcl = 1 # single swimmer
a_s = 0.12; b_s = 0.1  # swimmer semi-axes
theta_s = 0.           # swimmer orientation (mutated by the time loop)
cx_s = 0.0; cy_s = 0.35  # swimmer start center (clear of obstacles and wall)
ptcl_cell = {}
if num_ptcl > 0:
    ptcl_cell['ptcl_1'] = build_ellipse_body(a_s, b_s, theta_s, cx_s, cy_s, N_ptcl)

swimmer_size = max(a_s, b_s) # characteristic swimmer radius (semi-major axis)

# ---------------------------------------------------------------------------
# Obstacles -- elongated ellipses of different orientations, no-slip.
# num_obs is a knob (default 4). Layout verified non-overlapping with
# every edge-to-edge / wall gap >= 1.05 * swimmer diameter.
# ---------------------------------------------------------------------------
num_obs = 4 # knob: number of interior obstacles (3 or 4 supported by obs_specs)
# (a, b, theta, cx, cy) -- elongated ellipses (aspect ~3) at distinct angles
obs_specs = [
    (0.16, 0.055, 0.60,  -0.46, -0.34),
    (0.16, 0.055, -0.50,  0.43, -0.35),
    (0.16, 0.055, 1.25,  -0.48,  0.36),
    (0.16, 0.055, 2.35,   0.42,  0.40),
]
obs_cell = {}
for i in range(num_obs):
    a_o, b_o, th_o, cx_o, cy_o = obs_specs[i]
    obs_cell[f'obs_{i+1}'] = build_ellipse_body(a_o, b_o, th_o, cx_o, cy_o, N_obs)


def plot_geometry(fname='confined_obstacle_course_geom.png'):
    """Draw and save the container + obstacles + swimmer geometry."""
    plt.figure(figsize=(5, 5))
    plt.plot(np.real(s['x']), np.imag(s['x']), 'k', lw=2, label='container (no-slip)')
    for obs in obs_cell.values():
        plt.fill(np.real(obs['x']), np.imag(obs['x']), color='0.6', ec='k')
    for pt in ptcl_cell.values():
        plt.fill(np.real(pt['x']), np.imag(pt['x']), color='tab:red', ec='k')
    # legend proxies
    plt.fill([], [], color='0.6', ec='k', label='obstacles (no-slip)')
    plt.fill([], [], color='tab:red', ec='k', label='swimmer (slip)')
    plt.axis('equal')
    plt.legend(loc='lower right', fontsize=8)
    plt.title('Confined obstacle course geometry')
    plt.xlabel('x'); plt.ylabel('y')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'saved {fname}')


def _point_in_poly(xq, yq, polyx, polyy):
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
                   (xq_flat < (xj - xi) * (yq_flat - yi) / (yj - yi + 1e-300) + xi)
        inside_flat ^= crossing
        j = i
    return inside_flat.reshape(xq.shape)


def compute_flow_grid(edens, passive_cells, active_cells, nxg=140, ng=140, ypad=0.5):
    """Evaluate the flow speed field on a grid inside the container, masking out
    every body. Returns (X, Y, Ux, Uy) with NaN outside the fluid domain."""
    xg = np.linspace(-R_container - ypad, R_container + ypad, nxg)
    yg = np.linspace(-R_container - ypad, R_container + ypad, ng)
    X, Y = np.meshgrid(xg, yg)

    boundary_container = np.array(s['x'])
    inside = _point_in_poly(X, Y, np.real(boundary_container), np.imag(boundary_container))
    hole = np.zeros_like(X, dtype=bool)
    for cell in list(passive_cells.values()) + list(active_cells.values()):
        if 'x' in cell and cell['x'].size > 0:
            boundary = np.array(cell['x'])
            hole |= _point_in_poly(X, Y, np.real(boundary), np.imag(boundary))
    inside = inside & (~hole)

    tx_inside = (X[inside] + 1j*Y[inside]).astype(np.complex128)
    trg_jax = {'x': jnp.array(tx_inside), 'nx': jnp.ones_like(jnp.array(tx_inside)) + 0j}
    u_tot, _ = evalsol_all(trg_jax, s, passive_cells, active_cells, mu, edens)

    M = tx_inside.size
    Ux = np.full_like(X, np.nan, dtype=float)
    Uy = np.full_like(Y, np.nan, dtype=float)
    Ux[inside] = np.real(u_tot[:M])
    Uy[inside] = np.real(u_tot[M:2*M])
    return X, Y, Ux, Uy


def plot_streamlines_total(edens, passive_cells, active_cells, nxg=140, ng=140,
                           ypad=0.5, density=1.3,
                           fname='confined_obstacle_course_streamlines.png',
                           show=False):
    X, Y, Ux, Uy = compute_flow_grid(edens, passive_cells, active_cells, nxg, ng, ypad)

    plt.figure(figsize=(5, 5))
    speed = np.sqrt(Ux**2 + Uy**2)
    plt.pcolormesh(X, Y, speed, shading="auto")
    plt.streamplot(X, Y, Ux, Uy, density=density, linewidth=0.8, arrowsize=0.8)

    plt.plot(np.real(s['x']), np.imag(s['x']), 'k', lw=2)
    [plt.plot(np.real(pt['x']), np.imag(pt['x']), 'k', lw=2) for pt in active_cells.values()]
    [plt.plot(np.real(pt['x']), np.imag(pt['x']), 'k', lw=2) for pt in passive_cells.values()]

    plt.axis("equal")
    plt.xlim(-R_container - ypad, R_container + ypad)
    plt.ylim(-R_container - ypad, R_container + ypad)
    plt.colorbar(label="|u|")
    plt.title("Confined obstacle course: flow with slip swimmer")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'saved {fname}')
    if show:
        plt.show()


def plot_trajectory(traj, fname='confined_obstacle_course_trajectory.png'):
    """Draw and save the swimmer's path through the obstacle course."""
    traj = np.array(traj)
    plt.figure(figsize=(5, 5))
    plt.plot(np.real(s['x']), np.imag(s['x']), 'k', lw=2)
    for obs in obs_cell.values():
        plt.fill(np.real(obs['x']), np.imag(obs['x']), color='0.6', ec='k')
    # final swimmer outline
    sw = ptcl_cell['ptcl_1']
    plt.fill(np.real(sw['x']), np.imag(sw['x']), color='tab:red', alpha=0.4, ec='k')
    plt.plot(np.real(traj), np.imag(traj), '-', color='tab:red', lw=1.5, label='swimmer path')
    plt.plot(np.real(traj[0]), np.imag(traj[0]), 'go', ms=7, label='start')
    plt.plot(np.real(traj[-1]), np.imag(traj[-1]), 'rs', ms=7, label='end')
    plt.axis('equal')
    plt.legend(loc='lower right', fontsize=8)
    plt.title('Swimmer trajectory through obstacle course')
    plt.xlabel('x'); plt.ylabel('y')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    print(f'saved {fname}')


def make_movie(frames, dt, fname_base='confined_obstacle_course_movie', fps=8, density=2.0):
    """Assemble per-step flow frames into a movie (mp4 via ffmpeg, else gif),
    mirroring the MATLAB VideoWriter loop. Each frame redraws speed + streamlines
    + boundaries + swimmer + trajectory-so-far (axis cleared like MATLAB clf)."""
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    X, Y = frames[0]['X'], frames[0]['Y']
    # common color scale across frames
    vmax = max((np.nanmax(np.sqrt(f['Ux']**2 + f['Uy']**2))
                for f in frames if np.isfinite(f['Ux']).any()), default=1.0)
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(i):
        ax.clear()
        fr = frames[i]
        speed = np.sqrt(fr['Ux']**2 + fr['Uy']**2)
        ax.pcolormesh(X, Y, speed, shading='auto', vmin=0, vmax=vmax)
        ax.streamplot(X, Y, fr['Ux'], fr['Uy'], density=density,
                      linewidth=0.7, arrowsize=0.7, color='w')
        ax.plot(np.real(s['x']), np.imag(s['x']), 'k', lw=2)
        for ob in obs_cell.values():
            ax.fill(np.real(ob['x']), np.imag(ob['x']), color='0.6', ec='k')
        sw = fr['swimmer']
        ax.fill(np.real(sw), np.imag(sw), color='tab:red', ec='k')
        tr = np.array(fr['traj'])
        ax.plot(np.real(tr), np.imag(tr), '-', color='tab:red', lw=1.2)
        ax.set_aspect('equal')
        ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max())
        ax.set_title(f"t = {i*dt:.2f}")
        ax.set_xlabel('x'); ax.set_ylabel('y')
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000.0/fps, blit=False)
    try:
        anim.save(fname_base + '.mp4', writer=FFMpegWriter(fps=fps))
        print(f'saved {fname_base}.mp4')
    except Exception as e:
        print(f'ffmpeg unavailable ({e.__class__.__name__}); saving GIF instead')
        anim.save(fname_base + '.gif', writer=PillowWriter(fps=fps))
        print(f'saved {fname_base}.gif')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mobility solve: slip on swimmer, no-slip on wall + all obstacles,
# force/torque-free swimmer. Returns swimmer velocity/rotation and density.
# ---------------------------------------------------------------------------
def solve_rbm(s, obs_cell, ptcl_cell, mu, B1, B2, static=None):
    # `static` is the cached wall+obstacle block: the wall and the obstacles never
    # move, so it is built once outside the time loop and reused every step.
    [E, _, _, _, _] = rbm_wrapper(s, obs_cell, ptcl_cell, mu, static)
    N_nodes_wall = len(s['x'])
    N_nodes_ptcls = sum(len(pt['x']) for pt in ptcl_cell.values())
    N_nodes_obs = sum(len(o['x']) for o in obs_cell.values())
    n_ptcl = len(ptcl_cell)
    # no-slip wall + obstacles -> zero velocity; slip on swimmer; force/torque-free rows
    vrhs = jnp.zeros((N_nodes_wall*2 + N_nodes_obs*2,))
    vslip_ptcl = [get_vslip(B1, B2, pt['x'], pt['nx']*1j) for pt in ptcl_cell.values()]
    vrhs_ptcl = jnp.concatenate(vslip_ptcl) if vslip_ptcl else jnp.array([])
    erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((3*n_ptcl,))])
    [edens, resid, _, _] = jnp.linalg.lstsq(E, erhs, rcond=1e-15)
    base = 2*(N_nodes_wall + N_nodes_obs + N_nodes_ptcls)
    UOmega = edens[base:base + 3*n_ptcl]           # single swimmer: [Ux, Uy, Omega]
    edens_rbm = edens[:base]
    U = np.array([float(UOmega[0]), float(UOmega[1])])
    Omega = float(UOmega[2])
    resid_val = float(resid[0]) if resid.size > 0 else float('nan')
    return U, Omega, edens_rbm, resid_val


# ---------------------------------------------------------------------------
# Generic collision geometry: min-distance + collision normal from the swimmer
# to a body. n_hat is a unit 2-vector pointing AWAY from the body (into free
# space, toward the swimmer), so dot(U, n_hat) < 0 means moving into the body.
# ---------------------------------------------------------------------------
def min_dist_to_wall(swimmer):
    """Swimmer vs the circular container (radius R_container, center 0)."""
    x = np.asarray(swimmer['x'])
    s_a = 0.0 + 0.0j
    gaps = R_container - np.abs(x - s_a)   # gap of each swimmer node to the wall
    i = int(np.argmin(gaps))
    d = x[i] - s_a
    n_hat = -np.array([d.real, d.imag]) / abs(d)   # toward domain center = away from wall
    return float(gaps[i]), n_hat

def min_dist_to_body(swimmer, body, broad=0.5):
    """Swimmer vs an obstacle: node-to-node min distance (with a broad-phase skip)."""
    if abs(swimmer['a'] - body['a']) > swimmer['radius'] + body['radius'] + broad:
        return float('inf'), np.array([1.0, 0.0])   # far away; ignore
    xs = np.asarray(swimmer['x'])
    xo = np.asarray(body['x'])
    D = np.abs(xs[:, None] - xo[None, :])
    k = np.unravel_index(int(np.argmin(D)), D.shape)
    diff = xs[k[0]] - xo[k[1]]
    n_hat = np.array([diff.real, diff.imag]) / abs(diff)   # from obstacle toward swimmer
    return float(D[k]), n_hat

def gather_min_dists(swimmer, wall, obs_cell):
    """Return list of (name, dist, n_hat) for the wall and every obstacle."""
    md = [('wall',) + min_dist_to_wall(swimmer)]
    for name, obs in obs_cell.items():
        md.append((name,) + min_dist_to_body(swimmer, obs))
    return md


# ---------------------------------------------------------------------------
# Sliding collision resolver: for every body within `buffer` that the swimmer
# is moving into, remove that normal velocity component (slide). Iterate so the
# corrected velocity does not drive into any other body. Cap iterations; the
# caller stops the simulation if the cap is reached (swimmer wedged/trapped).
# ---------------------------------------------------------------------------
def resolve_velocity(U, min_dist_list, buffer, max_iter, tol=1e-9):
    U_corr = np.array(U, dtype=float)
    for it in range(max_iter):
        active = [n_hat for (_, dist, n_hat) in min_dist_list
                  if dist < buffer and np.dot(U_corr, n_hat) < -tol]
        if not active:
            return U_corr, True, it
        for n_hat in active:
            U_corr = U_corr - np.dot(U_corr, n_hat) * n_hat   # remove into-body component
    return U_corr, False, max_iter


# Save the geometry figure up front
plot_geometry()

# ---------------------------------------------------------------------------
# Time evolution with generic sliding collision handling
# ---------------------------------------------------------------------------
B1 = 1.23; B2 = -0.73           # optimal swim slip velocity (same as other RBM tests)
dt = 0.05                       # time step
T = 2.0                         # total time
Nt = int(round(T / dt))         # number of steps
collision_buffer = 0.08         # trigger sliding when a body is within this gap
max_collision_iter = 20         # cap on the sliding projection iterations
MAKE_MOVIE = True               # render a per-step flow movie (mp4 if ffmpeg, else gif)
movie_grid = 90                 # grid resolution for the movie frames (coarser = faster)

print("========= CONFINED OBSTACLE COURSE (time evolution) ========= ")
cx, cy, th = cx_s, cy_s, theta_s
swimmer = ptcl_cell['ptcl_1']
traj = [complex(cx, cy)]
frames = []
stopped = False
# The container wall and every obstacle are fixed for the whole run -- only the swimmer
# moves -- so the wall+obstacle diagonal block of the operator is identical at every
# step.  Build it once here and hand it to solve_rbm instead of reassembling it Nt times.
static_block = rbm_static_block(s, obs_cell, mu)
for tstep in range(Nt):
    U, Omega, edens_rbm, resid = solve_rbm(s, obs_cell, ptcl_cell, mu, B1, B2, static_block)
    md = gather_min_dists(swimmer, s, obs_cell)
    min_gap = min(d for (_, d, _) in md)
    closest = min(md, key=lambda e: e[1])[0]
    U_corr, ok, iters = resolve_velocity(U, md, collision_buffer, max_collision_iter)
    slid = not np.allclose(U_corr, U)
    print(f"[step {tstep:02d}] pos=({cx:+.3f},{cy:+.3f}) th={th:+.3f} "
          f"min_gap={min_gap:.3f} (to {closest}) "
          f"U=[{U[0]:+.3f},{U[1]:+.3f}] -> U_corr=[{U_corr[0]:+.3f},{U_corr[1]:+.3f}]"
          f"{' [SLIDE]' if slid else ''} resid={resid:.1e}")
    # Capture a movie frame for the CURRENT (consistent) state before moving
    if MAKE_MOVIE:
        Xg, Yg, Uxg, Uyg = compute_flow_grid(edens_rbm, obs_cell, ptcl_cell,
                                             nxg=movie_grid, ng=movie_grid)
        frames.append({'X': Xg, 'Y': Yg, 'Ux': Uxg, 'Uy': Uyg,
                       'swimmer': np.array(swimmer['x']), 'traj': list(traj)})
    if not ok:
        print(f"  max collision iterations ({max_collision_iter}) reached without a "
              f"collision-free velocity -- stopping simulation.")
        stopped = True
        break
    # Forward-Euler update using the corrected (sliding) velocity
    cx += U_corr[0] * dt
    cy += U_corr[1] * dt
    th += Omega * dt
    swimmer = build_ellipse_body(a_s, b_s, th, cx, cy, N_ptcl)
    ptcl_cell['ptcl_1'] = swimmer
    traj.append(complex(cx, cy))

print(f"Simulation {'stopped early' if stopped else 'completed'} after {len(traj)-1} step(s).")

# ---------------------------------------------------------------------------
# Visualize: swimmer trajectory + streamlines of the final (consistent) state
# ---------------------------------------------------------------------------
plot_trajectory(traj)
U, Omega, edens_final, resid = solve_rbm(s, obs_cell, ptcl_cell, mu, B1, B2, static_block)
plot_streamlines_total(edens_final, obs_cell, ptcl_cell, density=4,
                       fname='confined_obstacle_course_streamlines_final.png')

# ---------------------------------------------------------------------------
# Assemble the frames into a movie (mp4 via ffmpeg, else gif) -- like the MATLAB
# VideoWriter loop in test/ptcl_rbm_constrained_time.m
# ---------------------------------------------------------------------------
if MAKE_MOVIE and frames:
    make_movie(frames, dt)
