# Time evolution of a squirmer in a singly-periodic 2D Stokes "pipe", with the
# generic sliding collision handling (upper + lower walls), mirroring
# singlyperiodic/perivelpipe_rbm_time.m. Each step: solve the periodic RBM
# mobility problem, resolve wall collisions by removing into-wall velocity
# components (iterated, with a max-iter cap that stops the sim), Euler-step the
# swimmer, and (optionally) capture a flow frame for a movie.
# The walls are fixed; the swimmer's x-center is wrapped by one period to stay
# in the unit cell (equivalent to, and simpler than, the MATLAB window shift).

from jax import config
config.update("jax_enable_x64", True)

import time
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(current_dir, '../periodic'))
from periodic.structure_jax import *
from periodic.periodic_ELS_jax import *


@jit
def get_vslip(B1, B2, ptcl_coords_2d, ptcl_tang):
    """Squirmer slip velocity. ptcl_coords_2d: (num_ptcl, N) complex nodes;
    ptcl_tang: 1d (num_ptcl*N) complex tangent."""
    xc = jnp.mean(ptcl_coords_2d, axis=1, keepdims=True)
    xmxc_tot = (ptcl_coords_2d - xc).reshape(-1)
    theta = jnp.atan2(jnp.imag(xmxc_tot), jnp.real(xmxc_tot))
    u_theta = B1 * jnp.sin(theta) + B2 * jnp.sin(theta) * jnp.cos(theta)
    if len(ptcl_tang) == 0:
        vslip = -u_theta * jnp.sin(theta) + 1j * u_theta * jnp.cos(theta)
    else:
        vslip = u_theta * jnp.real(ptcl_tang) + 1j * u_theta * jnp.imag(ptcl_tang)
    return vslip


# ---------------------------------------------------------------------------
# Discretization parameters
# ---------------------------------------------------------------------------
Np_wall = 10          # panels per wall
p_wall = 10           # GL order per panel
N_wall = Np_wall * p_wall
N_ptcl = 80           # global nodes on the swimmer
N_side = 40
N_prx = 2 * N_side
peri_len = 2 * jnp.pi
mu = 0.7
B1 = 1.23; B2 = -0.73
num_ptcl = 1

# ---------------------------------------------------------------------------
# Fixed geometry: upper + lower pipe walls (period 2*pi), side walls, proxies.
# Walls do not move (we wrap the swimmer instead), so keep these as globals.
# ---------------------------------------------------------------------------
Z_top = lambda t : peri_len/(2*jnp.pi)*(2*jnp.pi - t) + 1j*(1 + 0.3*jnp.sin(2*jnp.pi - t))
Zp_top = lambda t : -peri_len/(2*jnp.pi) - 1j*(0.3*jnp.cos(2*jnp.pi - t))
Zpp_top = lambda t : -1j*0.3*jnp.sin(2*jnp.pi - t)
Z_bot = lambda t : peri_len/(2*jnp.pi)*t + 1j*(-1 + 0.3*jnp.sin(t))
Zp_bot = lambda t : peri_len/(2*jnp.pi) + 1j*(0.3*jnp.cos(t))
Zpp_bot = lambda t : -1j*0.3*jnp.sin(t)
[sx, sxp, snx, scur, sw, swxp, sxlo, sxhi] = channel_wall_glpanels(Z_top, Np_wall, p_wall, Zp_top, Zpp_top)
[sx2, sxp2, snx2, scur2, sw2, swxp2, sxlo2, sxhi2] = channel_wall_glpanels(Z_bot, Np_wall, p_wall, Zp_bot, Zpp_bot)
sx = jnp.concatenate([sx, sx2]); sxp = jnp.concatenate([sxp, sxp2])
snx = jnp.concatenate([snx, snx2]); scur = jnp.concatenate([scur, scur2])
sw = jnp.concatenate([sw, sw2]); swxp = jnp.concatenate([swxp, swxp2])
sxlo = jnp.concatenate([sxlo, sxlo2]); sxhi = jnp.concatenate([sxhi, sxhi2])
N_nodes_wall = len(sx)

[lx, lnx] = side_wall(0., 2, N_side); lx = lx - 1j
[rx, rnx] = side_wall(0. + peri_len, 2, N_side); rx = rx - 1j
[px, pxp, pnx, pwt] = proxy(1.1 * peri_len, peri_len, N_prx)

# Numpy wall-graph helpers for the collision geometry (y = wall height at x).
# The "away" direction is the negated wall normal: it points from the wall into
# the domain (toward the swimmer), so dot(U, n_hat) < 0 means moving into a wall.
UXf = lambda x : x + 1j*(1 + 0.3*np.sin(x))
DXf = lambda x : x + 1j*(-1 + 0.3*np.sin(x))
away_upper = lambda x : 0.3*np.cos(x) - 1j    # = -UNX, points downward into the pipe
away_lower = lambda x : -0.3*np.cos(x) + 1j   # = -DNX, points upward into the pipe


# ---------------------------------------------------------------------------
# Swimmer builder (rotated ellipse) -- rebuilt each step like MATLAB setupquad
# ---------------------------------------------------------------------------
def build_swimmer(a, b, theta, cx, cy, N):
    Z = lambda t : a*jnp.cos(t)*jnp.cos(theta) - b*jnp.sin(t)*jnp.sin(theta) + cx \
                + 1j*(a*jnp.cos(t)*jnp.sin(theta) + b*jnp.sin(t)*jnp.cos(theta) + cy)
    Zp = lambda t : -a*jnp.sin(t)*jnp.cos(theta) - b*jnp.cos(t)*jnp.sin(theta) \
                + 1j*(-a*jnp.sin(t)*jnp.sin(theta) + b*jnp.cos(t)*jnp.cos(theta))
    Zpp = lambda t : -a*jnp.cos(t)*jnp.cos(theta) + b*jnp.sin(t)*jnp.sin(theta) \
                + 1j*(-a*jnp.cos(t)*jnp.sin(theta) - b*jnp.sin(t)*jnp.cos(theta))
    ptx, ptxp, ptnx, ptcur, ptw = channel_wall_func(Z, N, Zp, Zpp)
    ptwxp = 2*jnp.pi/N * ptxp
    ptt = jnp.linspace(0, 2*jnp.pi, N, endpoint=False)
    pta = jnp.array([cx + 1j*cy])
    return dict(x=ptx, xp=ptxp, nx=ptnx, cur=ptcur, w=ptw, wxp=ptwxp, t=ptt, a=pta)


# ---------------------------------------------------------------------------
# Periodic RBM mobility solve (no-slip walls, slip swimmer, force/torque-free).
# Returns swimmer velocity/spin and the full density vector `edens`.
# ---------------------------------------------------------------------------
def solve_rbm(pt):
    [E, _, _, _, _, _, _] = ELSmatrix_rbm(sx, snx, scur, sw, pt['x'], pt['nx'],
                                          pt['xp'], pt['cur'], pt['w'], num_ptcl,
                                          px, pwt, lx, lnx, rx, rnx, peri_len, mu)
    vrhs = jnp.zeros((N_nodes_wall*2,))
    ptx_2d = pt['x'].reshape((num_ptcl, N_ptcl))
    vrhs_ptcl_cpx = get_vslip(B1, B2, ptx_2d, 1j*pt['nx'])
    vrhs_ptcl = jnp.concatenate([jnp.real(vrhs_ptcl_cpx), jnp.imag(vrhs_ptcl_cpx)])
    Tjump = jnp.zeros((2*N_side,))                 # jump = 0 (no pressure drive)
    erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((3*num_ptcl,)),
                            jnp.zeros((2*N_side,)), Tjump])
    [edens, resid, _, _] = jnp.linalg.lstsq(E, erhs, rcond=1e-15)
    N_nodes_ptcl = len(pt['x'])
    base = 2*(N_nodes_wall + N_nodes_ptcl)
    UOmega = edens[base:base + 3*num_ptcl]
    U = np.array([float(UOmega[0]), float(UOmega[1])])
    Omega = float(UOmega[2])
    resid_val = float(resid[0]) if resid.size > 0 else float('nan')
    return U, Omega, edens, resid_val


# ---------------------------------------------------------------------------
# Generic collision geometry + sliding resolver (identical scheme to the
# confined obstacle course: gather min-dist + away-normal to each body, then
# remove into-body velocity components, iterate, cap iters -> stop the sim).
# ---------------------------------------------------------------------------
def min_dist_to_periwall(ptcl_x, Xfunc, nx_away):
    xr = np.real(np.asarray(ptcl_x))
    gaps = np.abs(np.imag(np.asarray(Xfunc(xr)) - np.asarray(ptcl_x)))  # vertical gap proxy
    i = int(np.argmin(gaps))
    nrm = complex(nx_away(xr[i]))
    n_hat = np.array([nrm.real, nrm.imag]) / abs(nrm)
    return float(gaps[i]), n_hat

def gather_min_dists(ptcl_x):
    return [('wall_upper',) + min_dist_to_periwall(ptcl_x, UXf, away_upper),
            ('wall_lower',) + min_dist_to_periwall(ptcl_x, DXf, away_lower)]

def resolve_velocity(U, min_dist_list, buffer, max_iter, tol=1e-9):
    U_corr = np.array(U, dtype=float)
    for it in range(max_iter):
        active = [n_hat for (_, dist, n_hat) in min_dist_list
                  if dist < buffer and np.dot(U_corr, n_hat) < -tol]
        if not active:
            return U_corr, True, it
        for n_hat in active:
            U_corr = U_corr - np.dot(U_corr, n_hat) * n_hat   # remove into-wall component
    return U_corr, False, max_iter


# ---------------------------------------------------------------------------
# Flow-field evaluation on a grid (for the movie), masked to the fluid domain.
# ---------------------------------------------------------------------------
def _point_in_poly(xq, yq, polyx, polyy):
    xq_flat = xq.ravel(); yq_flat = yq.ravel()
    inside_flat = np.zeros_like(xq_flat, dtype=bool)
    n = len(polyx)
    if n == 0:
        return inside_flat.reshape(xq.shape)
    j = n - 1
    for i in range(n):
        xi, yi = polyx[i], polyy[i]; xj, yj = polyx[j], polyy[j]
        crossing = ((yi > yq_flat) != (yj > yq_flat)) & \
                   (xq_flat < (xj - xi) * (yq_flat - yi) / (yj - yi + 1e-300) + xi)
        inside_flat ^= crossing; j = i
    return inside_flat.reshape(xq.shape)

def compute_flow_grid(pt, edens, nxg=120, ng=36, ypad=0.4, delta=0.02):
    xg = np.linspace(0.0, float(peri_len), nxg)
    yg = np.linspace(-1.0 - ypad, 1.0 + ypad, ng)
    X, Y = np.meshgrid(xg, yg)
    YT = np.imag(UXf(xg)); YB = np.imag(DXf(xg))
    inside = (Y >= YB[None, :] + delta) & (Y <= YT[None, :] - delta)
    boundary = np.array(pt['x'])
    inside &= ~_point_in_poly(X, Y, np.real(boundary), np.imag(boundary))
    tx_inside = (X[inside] + 1j*Y[inside]).astype(np.complex128)
    tx_jax = jnp.array(tx_inside); tnx_jax = jnp.ones_like(tx_jax) + 0j
    u_tot, _ = evalsol_rbm(tx_jax, tnx_jax, sx, sxlo, sxhi, snx, sxp, scur, sw,
                           pt['x'], pt['nx'], pt['t'], pt['a'], pt['xp'], pt['w'],
                           pt['wxp'], px, pwt, peri_len, mu, edens)
    M = tx_inside.size
    Ux = np.full_like(X, np.nan, dtype=float); Uy = np.full_like(Y, np.nan, dtype=float)
    Ux[inside] = np.real(np.asarray(u_tot)[:M]); Uy[inside] = np.real(np.asarray(u_tot)[M:2*M])
    return X, Y, Ux, Uy


def make_movie(frames, dt, fname_base='peri_rbm_time_movie', fps=8, density=2.0):
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    X, Y = frames[0]['X'], frames[0]['Y']
    vmax = max((np.nanmax(np.sqrt(f['Ux']**2 + f['Uy']**2))
                for f in frames if np.isfinite(f['Ux']).any()), default=1.0)
    vmax = vmax if vmax > 0 else 1.0
    tt = np.linspace(0.0, float(peri_len), 400)
    zt = np.imag(UXf(tt)); zb = np.imag(DXf(tt))
    fig, ax = plt.subplots(figsize=(10, 3))

    def draw(i):
        ax.clear()
        fr = frames[i]
        speed = np.sqrt(fr['Ux']**2 + fr['Uy']**2)
        ax.pcolormesh(X, Y, speed, shading='auto', vmin=0, vmax=vmax)
        ax.streamplot(X, Y, fr['Ux'], fr['Uy'], density=density,
                      linewidth=0.7, arrowsize=0.7, color='w')
        ax.plot(tt, zt, 'k', lw=2); ax.plot(tt, zb, 'k', lw=2)
        sw_x = fr['swimmer']
        ax.fill(np.real(sw_x), np.imag(sw_x), color='tab:red', ec='k')
        ax.set_xlim(0, float(peri_len)); ax.set_ylim(Y.min(), Y.max())
        ax.set_aspect('equal'); ax.set_title(f"t = {i*dt:.2f}")
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
# Time evolution
# ---------------------------------------------------------------------------
a_s = 0.25; b_s = 0.15                 # swimmer semi-axes
cx, cy, th = 1.0, 0.45, np.pi/7        # start clear of the walls, angled toward upper wall
dt = 0.1
T = 6.0
Nt = int(round(T / dt))
buffer = 0.03                          # trigger sliding when a wall is within this gap
max_collision_iter = 20                # cap on sliding-projection iters (stop sim if reached)
MAKE_MOVIE = True
movie_grid_x = 120; movie_grid_y = 36

print("========= PERIODIC PIPE SQUIRMER (time evolution) =========")
pt = build_swimmer(a_s, b_s, th, cx, cy, N_ptcl)
traj = [complex(cx, cy)]
frames = []
stopped = False
for tstep in range(Nt):
    U, Omega, edens, resid = solve_rbm(pt)
    md = gather_min_dists(pt['x'])
    min_gap = min(d for (_, d, _) in md)
    closest = min(md, key=lambda e: e[1])[0]
    U_corr, ok, iters = resolve_velocity(U, md, buffer, max_collision_iter)
    slid = not np.allclose(U_corr, U)
    print(f"[step {tstep:02d}] pos=({cx:+.3f},{cy:+.3f}) th={th:+.3f} "
          f"min_gap={min_gap:.3f} (to {closest}) "
          f"U=[{U[0]:+.3f},{U[1]:+.3f}] -> U_corr=[{U_corr[0]:+.3f},{U_corr[1]:+.3f}]"
          f"{' [SLIDE]' if slid else ''} resid={resid:.1e}")
    if MAKE_MOVIE:
        Xg, Yg, Uxg, Uyg = compute_flow_grid(pt, edens, nxg=movie_grid_x, ng=movie_grid_y)
        frames.append({'X': Xg, 'Y': Yg, 'Ux': Uxg, 'Uy': Uyg,
                       'swimmer': np.array(pt['x'])})
    if not ok:
        print(f"  max collision iterations ({max_collision_iter}) reached without a "
              f"collision-free velocity -- stopping simulation.")
        stopped = True
        break
    # Forward-Euler update with the corrected (sliding) velocity
    cx += U_corr[0] * dt
    cy += U_corr[1] * dt
    th += Omega * dt
    if cx >= float(peri_len):              # keep the swimmer in the unit cell (period wrap)
        cx -= float(peri_len)
        print(f"  wrapped x by one period -> cx={cx:+.3f}")
    elif cx < 0.0:
        cx += float(peri_len)
        print(f"  wrapped x by one period -> cx={cx:+.3f}")
    pt = build_swimmer(a_s, b_s, th, cx, cy, N_ptcl)
    traj.append(complex(cx, cy))

print(f"Simulation {'stopped early' if stopped else 'completed'} after {len(traj)-1} step(s).")

if MAKE_MOVIE and frames:
    make_movie(frames, dt)
