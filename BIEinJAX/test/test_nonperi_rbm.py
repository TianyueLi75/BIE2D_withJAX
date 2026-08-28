
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

# Solver for the extended (density + U/Omega) system:
#   1 = lstsq on the full E (historical default)
#   2 = jnp.linalg.solve on the full E (it is square)
#   3 = Schur complement, eliminating the static wall+obstacle block first
SOLVER_MODE = 1
COMPARE_MODES = False # run all three and report timings + agreement
# Check the matrix-free operator against the dense one, solve it with scipy GMRES on a
# LinearOperator, and report timings for both routes.  See solve_gmres_matvec.
TEST_MATVEC = True
# Check the near/far-masked construction of E: the far body-body couplings go through the
# smooth rule instead of close evaluation, and both the assembled E and its matvec must
# still agree with the all-close operator (accuracy-neutral), while all three solvers agree
# on the twist.  See nearfar_split_E in nonperiodic/BIOsolve_pytree.py.
TEST_NEARFAR_E = True

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
[E,bc_gamma_mat,intF,intT,blocks] = rbm_wrapper(s, obs_cell, ptcl_cell, mu)

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
def unpack_UOmega(x):
    off = 2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls)
    return x[off:off+3*num_ptcl]

if COMPARE_MODES:
    # Densities may legitimately differ between modes by the rank-one nullspace of
    # the wall DL operator, which generates no flow.  U/Omega and the evaluated
    # field are the quantities that must agree.
    ref = None
    for m in (1, 2, 3):
        solve_rbm_system(E, blocks, erhs, mode=m) # warm up jit / compilation
        t0 = time.perf_counter()
        [x_m, resid_m, _] = solve_rbm_system(E, blocks, erhs, mode=m)
        x_m.block_until_ready()
        t_m = time.perf_counter() - t0
        u_m, _ = evalsol_all(trg, s, obs_cell, ptcl_cell, mu,
                             x_m[:2*(N_nodes_wall+N_nodes_obs+N_nodes_ptcls)])
        if ref is None:
            ref, ref_u = x_m, u_m
            print(f'mode {m}: {t_m*1e3:8.2f} ms   resid = {resid_m:.3g}   (reference)')
        else:
            d_dens = jnp.max(jnp.abs(x_m - ref))
            d_uom = jnp.max(jnp.abs(unpack_UOmega(x_m) - unpack_UOmega(ref)))
            d_u = jnp.max(jnp.abs(u_m - ref_u))
            print(f'mode {m}: {t_m*1e3:8.2f} ms   resid = {resid_m:.3g}   '
                  f'max|d dens| = {d_dens:.3g}, max|d UOmega| = {d_uom:.3g}, '
                  f'max|d u| = {d_u:.3g}')

[edens, resid, static_fac] = solve_rbm_system(E, blocks, erhs, mode=SOLVER_MODE)
print(f'solver mode {SOLVER_MODE}: resid norm = {resid:.3g}')
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


# ---------------------------------------------------------------------------
#  Matrix-free (matvec) operator + GMRES, checked against the dense solve
#
#  rbm_matvec_flat applies exactly the operator rbm_wrapper assembles, without ever
#  forming it.  The first check below is the one that keeps the two implementations
#  honest: if a term is ever added to rbm / rbm_obs and not to rbm_matvec /
#  rbm_matvec_obs, matvec(v) stops matching E @ v and this fails.
# ---------------------------------------------------------------------------
def _matvec_report(tag, s_, obs_, ptcl_, mu_, rhs_, E_, blocks_, n_ptcl_, timing=False):
    layout = rbm_dof_layout(s_, obs_, ptcl_)
    assert layout['n_total'] == E_.shape[0], \
        f'{tag}: layout says {layout["n_total"]} dofs, E is {E_.shape[0]}'
    off_uom = layout['off_uom']

    # --- 1. operator consistency: matvec(v) must equal E @ v ---
    # Measured as a backward error, ||matvec(v) - E v||_inf / (||E||_inf ||v||_inf).
    # Normalising by ||E v||_inf instead would be misleading: E is numerically
    # singular, so for a solution vector carrying a large nullspace component E v is
    # a near-total cancellation and any sane matvec would look inaccurate.
    #
    # The tolerance is 1e-11, not machine precision, because the two sides no longer
    # do the same arithmetic: stoDLP_closeglobal's traction takes its density-driven
    # route for the narrow densities of a matvec and its dense-matrix route for the
    # eye(2N) of an assembly, and the two reorder the same sum differently.  They
    # agree to ~1e-13 in backward error, which is two orders below the ~1e-10 error
    # floor of the close-eval traction quadrature itself, so the gap is roundoff and
    # not a discrepancy.  A genuinely missing term -- what this check exists to catch
    # -- shows up at O(1), so the looser bound costs nothing.
    E_scale = float(jnp.linalg.norm(E_, ord=jnp.inf))
    rng = np.random.default_rng(0)
    worst = 0.
    for label, v in (('random vector', jnp.array(rng.standard_normal(layout['n_total']))),
                     ('dense solution', solve_rbm_system(E_, blocks_, rhs_, mode=2)[0])):
        ref = E_ @ v
        got = rbm_matvec_flat(v, s_, obs_, ptcl_, mu_)
        rel = float(jnp.max(jnp.abs(got - ref)) / (E_scale * jnp.max(jnp.abs(v))))
        worst = max(worst, rel)
        print(f'  {tag} matvec vs E@v [{label:15s}] backward err = {rel:.3e}')
    assert worst < 1e-11, f'{tag}: matvec does not reproduce E (backward err {worst:.3e})'

    # --- 2. GMRES on the matrix-free operator vs the dense solve ---
    # make_rbm_linear_operator warms up the jit, so nothing timed below compiles.
    op, matvec_fn, _ = make_rbm_linear_operator(s_, obs_, ptcl_, mu_)
    [x_dense, resid_dense, _] = solve_rbm_system(E_, blocks_, rhs_, mode=2)
    # Preconditioned by default (block-Jacobi over the self-interaction blocks); the
    # unpreconditioned run is kept alongside it so the iteration counts stay visible.
    pc_ = rbm_block_jacobi(s_, obs_, ptcl_, mu_)
    [x_g, resid_g, info_g, iters_g, _] = solve_gmres_matvec(s_, obs_, ptcl_, mu_, rhs_,
                                                            op=op, pc=pc_)
    [_, _, info_np, iters_np, _] = solve_gmres_matvec(s_, obs_, ptcl_, mu_, rhs_,
                                                      op=op, precond=False)
    d_uom = float(jnp.max(jnp.abs(x_g[off_uom:] - x_dense[off_uom:])))
    d_dens = float(jnp.max(jnp.abs(x_g[:off_uom] - x_dense[:off_uom])))
    u_g, _ = evalsol_all(trg, s_, obs_, ptcl_, mu_, x_g[:off_uom])
    u_d, _ = evalsol_all(trg, s_, obs_, ptcl_, mu_, x_dense[:off_uom])
    d_u = float(jnp.max(jnp.abs(u_g - u_d)))
    print(f'  {tag} gmres: info = {info_g}, {iters_g} iters, ||A x - b|| = {resid_g:.3e}'
          f'   (unpreconditioned: {iters_np} iters, info {info_np})')
    print(f'  {tag} vs dense: max|d UOmega| = {d_uom:.3e}, max|d u| = {d_u:.3e}')
    print(f'  {tag} max|d dens| = {d_dens:.3e}  <- the wall-DL nullspace, generates no'
          f' flow; not an error')
    assert info_g == 0, f'{tag}: gmres did not converge (info {info_g})'
    assert d_uom < 1e-8, f'{tag}: U/Omega disagree by {d_uom:.3e}'
    assert d_u < 1e-8, f'{tag}: evaluated velocity disagrees by {d_u:.3e}'
    print(f'  {tag} PASSED')
    if not timing:
        return

    # --- 3. Timing: dense assembly + solve versus matrix-free GMRES ---
    # Everything below has already been traced and compiled once, so these are
    # steady-state numbers, not compile times.
    def _time(f, reps=3):
        f()  # untimed: everything here is already compiled, this just settles caches
        t0 = time.perf_counter()
        for _ in range(reps):
            r = f()
            (r[0] if isinstance(r, tuple) else r).block_until_ready()
        return (time.perf_counter() - t0) / reps

    v_zero = jnp.zeros((layout['n_total'],))

    t_assemble = _time(lambda: rbm_wrapper(s_, obs_, ptcl_, mu_))
    t_direct = _time(lambda: solve_rbm_system(E_, blocks_, rhs_, mode=2)[0])
    t_lstsq = _time(lambda: solve_rbm_system(E_, blocks_, rhs_, mode=1)[0])
    t_matvec = _time(lambda: matvec_fn(v_zero), reps=10)
    t0 = time.perf_counter()
    solve_gmres_matvec(s_, obs_, ptcl_, mu_, rhs_, op=op, pc=pc_)
    t_gmres = time.perf_counter() - t0
    t0 = time.perf_counter()
    solve_gmres_matvec(s_, obs_, ptcl_, mu_, rhs_, op=op, precond=False)
    t_gmres_np = time.perf_counter() - t0
    def _time_pc(reps=3):
        rbm_block_jacobi(s_, obs_, ptcl_, mu_)  # untimed: settle the jit caches
        t0 = time.perf_counter()
        for _ in range(reps):
            rbm_block_jacobi(s_, obs_, ptcl_, mu_)
        return (time.perf_counter() - t0) / reps
    t_pc = _time_pc()

    n = layout['n_total']
    print(f'\n  --- timing, n = {n} unknowns (jit warmed up) ---')
    print(f'  dense: assemble E          {t_assemble*1e3:9.2f} ms')
    print(f'  dense: solve mode 2        {t_direct*1e3:9.2f} ms')
    print(f'  dense: solve mode 1        {t_lstsq*1e3:9.2f} ms')
    print(f'  dense TOTAL (assemble+2)   {(t_assemble+t_direct)*1e3:9.2f} ms')
    print(f'  matrix-free: one matvec    {t_matvec*1e3:9.2f} ms'
          f'   ({t_matvec/t_assemble*100:.0f}% of a full assembly)')
    print(f'  matrix-free: block-Jacobi  {t_pc*1e3:9.2f} ms  (self blocks only, no close eval)')
    print(f'  matrix-free: gmres total   {t_gmres*1e3:9.2f} ms  ({iters_g} iters, preconditioned)')
    print(f'  matrix-free: gmres, no pc  {t_gmres_np*1e3:9.2f} ms  ({iters_np} iters)')
    print(f'  ratio matrix-free / dense  {t_gmres/(t_assemble+t_direct):9.2f} x')
    print('  NOTE: a matvec is not 1/n of an assembly because the close-eval kernels do'
          '\n        density-independent O(M*N) setup (panel Cauchy quadrature, spectral'
          '\n        upsampling) on every apply, and GMRES redoes it every iteration.'
          '\n        The matrix-free path wins on memory well before it wins on time:'
          '\n        it never allocates the n x n operator.')


if TEST_MATVEC:
    print("\n========= MATRIX-FREE MATVEC + GMRES =========")
    _matvec_report('[main]', s, obs_cell, ptcl_cell, mu, erhs, E, blocks, num_ptcl, timing=True)

    # A second, self-contained case at reduced resolution so that rbm_matvec_obs is
    # exercised on every run, whatever num_obs is set to above.
    print("\n--- obstacle path (self-contained, independent of the knobs above) ---")
    def _small_body(a, b, cx, cy, N):
        Zb = lambda t : a*jnp.cos(t) + cx + 1j*(b*jnp.sin(t) + cy)
        Zbp = lambda t : -a*jnp.sin(t) + 1j*(b*jnp.cos(t))
        Zbpp = lambda t : -a*jnp.cos(t) + 1j*(-b*jnp.sin(t))
        bd = channel_wall_func(Zb, N, Zbp, Zbpp)
        bd['a'] = cx + 1j*cy; bd['theta0'] = 0.; bd['radius'] = max(a, b)
        return bd
    s_o = channel_wall_glpanels(Z_container, 6, 10, Zp_container, Zpp_container)
    obs_o = {'obs_1': _small_body(0.1, 0.1, 0.35, 0.0, 50)}
    ptcl_o = {'ptcl_1': _small_body(0.12, 0.1, 0.0, 0.15, 60)}
    lay_o = rbm_dof_layout(s_o, obs_o, ptcl_o)
    rhs_o = jnp.concatenate([jnp.zeros((lay_o['n_wall'] + lay_o['n_obs'],))]
                            + [get_vslip(B1, B2, pt['x'], pt['nx']*1j) for pt in ptcl_o.values()]
                            + [jnp.zeros((lay_o['n_uom'],))])
    [E_o, _, _, _, blocks_o] = rbm_wrapper(s_o, obs_o, ptcl_o, mu)
    _matvec_report('[obs]', s_o, obs_o, ptcl_o, mu, rhs_o, E_o, blocks_o, len(ptcl_o))

    print("\nAll jitted entry points traced and compiled: rbm_wrapper, rbm_matvec_wrapper,"
          "\nrbm_matvec_flat, evalsol_all, get_vslip.")


def _nearfar_E_report(tag, s_, obs_, ptcl_, mu_, rhs_):
    """Near/far-masked E: assembled and matrix-free, both vs the all-close operator.

    Masking sends the far body-body couplings through the smooth rule.  It must be
    accuracy-neutral: the masked E equals the all-close E to the smooth-vs-close floor
    (roundoff for well-separated bodies), the masked matvec equals masked E @ v to the
    close-eval traction floor, and direct / Schur / GMRES on the masked system all agree
    on the twist with the all-close direct solve.
    """
    split = nearfar_split_E(s_, obs_, ptcl_)
    n_masked = sum(1 for v in split.values() if v is not None)
    print(f"  {tag} body-body pairs sent to smooth: {n_masked}/{len(split)}")

    E_close, _, _, _, blk_close = rbm_wrapper(s_, obs_, ptcl_, mu_)
    E_mask, _, _, _, blk_mask = rbm_wrapper(s_, obs_, ptcl_, mu_, split=split)
    E_close = np.asarray(E_close); E_mask = np.asarray(E_mask)
    d_E = float(np.max(np.abs(E_mask - E_close)) / np.max(np.abs(E_close)))
    print(f"  {tag} ||E_mask - E_close|| / ||E_close|| = {d_E:.3e}  (far pairs smooth vs close)")
    assert d_E < 1e-9, f'{tag}: masked E disagrees with all-close E by {d_E:.3e}'

    # masked matvec must reproduce masked E @ v
    E_scale = float(np.linalg.norm(E_mask, ord=np.inf))
    rng = np.random.default_rng(1)
    v = jnp.array(rng.standard_normal(E_mask.shape[0]))
    got = rbm_matvec_flat(v, s_, obs_, ptcl_, mu_, split=split)
    be = float(jnp.max(jnp.abs(got - E_mask @ v)) / (E_scale * jnp.max(jnp.abs(v))))
    print(f"  {tag} matvec(split) vs E_mask @ v backward err = {be:.3e}")
    assert be < 1e-11, f'{tag}: masked matvec does not reproduce masked E ({be:.3e})'

    # the three solvers on the masked system agree with the all-close direct twist
    off = rbm_dof_layout(s_, obs_, ptcl_)['off_uom']
    tw_close = np.asarray(solve_rbm_system(E_close, blk_close, rhs_, mode=2)[0][off:])
    tw_direct = np.asarray(solve_rbm_system(E_mask, blk_mask, rhs_, mode=2)[0][off:])
    tw_schur = np.asarray(solve_rbm_system(E_mask, blk_mask, rhs_, mode=3)[0][off:])
    pc_ = rbm_block_jacobi(s_, obs_, ptcl_, mu_)
    x_g, _, info_g, iters_g, _ = solve_gmres_matvec(s_, obs_, ptcl_, mu_, rhs_,
                                                    split=split, pc=pc_)
    tw_gmres = np.asarray(x_g[off:])
    d_direct = float(np.max(np.abs(tw_direct - tw_close)))
    d_schur = float(np.max(np.abs(tw_schur - tw_close)))
    d_gmres = float(np.max(np.abs(tw_gmres - tw_close)))
    print(f"  {tag} masked twist vs all-close: direct {d_direct:.2e}, "
          f"schur {d_schur:.2e}, gmres {d_gmres:.2e} ({iters_g} its, info {info_g})")
    assert d_direct < 1e-8, f'{tag}: masked direct twist off by {d_direct:.2e}'
    assert d_schur < 1e-6, f'{tag}: masked schur twist off by {d_schur:.2e}'
    assert info_g == 0 and d_gmres < 1e-8, f'{tag}: masked gmres twist off by {d_gmres:.2e}'
    print(f"  {tag} PASSED")


if TEST_NEARFAR_E:
    print("\n========= NEAR/FAR-MASKED E =========")
    # Three well-separated obstacles so the split actually gates: every obstacle-obstacle
    # and swimmer-obstacle pair is far, so all go to the smooth branch.
    def _small_body(a, b, cx, cy, N):
        Zb = lambda t : a*jnp.cos(t) + cx + 1j*(b*jnp.sin(t) + cy)
        Zbp = lambda t : -a*jnp.sin(t) + 1j*(b*jnp.cos(t))
        Zbpp = lambda t : -a*jnp.cos(t) + 1j*(-b*jnp.sin(t))
        bd = channel_wall_func(Zb, N, Zbp, Zbpp)
        bd['a'] = cx + 1j*cy; bd['theta0'] = 0.; bd['radius'] = max(a, b)
        return bd
    s_m = channel_wall_glpanels(Z_container, 6, 10, Zp_container, Zpp_container)
    obs_m = {'obs_1': _small_body(0.1, 0.1, 0.55, 0.0, 40),
             'obs_2': _small_body(0.1, 0.1, -0.5, 0.4, 40),
             'obs_3': _small_body(0.1, 0.1, -0.2, -0.55, 40)}
    ptcl_m = {'ptcl_1': _small_body(0.12, 0.1, 0.0, 0.12, 60)}
    lay_m = rbm_dof_layout(s_m, obs_m, ptcl_m)
    rhs_m = jnp.concatenate([jnp.zeros((lay_m['n_wall'] + lay_m['n_obs'],))]
                            + [get_vslip(B1, B2, pt['x'], pt['nx']*1j) for pt in ptcl_m.values()]
                            + [jnp.zeros((lay_m['n_uom'],))])
    _nearfar_E_report('[nearfar]', s_m, obs_m, ptcl_m, mu, rhs_m)

plot_streamlines_total(edens_rbm, obs_cell, ptcl_cell, Xc_list, r_list, density=4)