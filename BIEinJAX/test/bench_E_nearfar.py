"""Timing benchmark: near/far-masked E vs all-close E, for the three solvers.

Times one forward mobility solve three ways -- the direct dense solve, the cached
fixed-block Schur reduction, and matrix-free GMRES on the operator apply -- for two
geometries:

    empty      one swimmer in a container, no obstacles.  There are no body-body pairs,
               so the *only* thing to mask is the swimmer<->wall coupling: with the
               swimmer adrift near the centre the wall is a large curve most of whose
               nodes are far from it, so both the swimmer->wall (global close eval) and
               wall->swimmer (panel close eval) blocks go smooth.  This used to be a null
               control (~1x); with the wall split it is the cleanest win.
    10-obs     one swimmer among ten well-separated obstacles (most obstacle-obstacle
               and swimmer-obstacle pairs are far apart, so masking sends them through
               the smooth rule instead of a full close-eval pass)

For each geometry every solver is run twice: `close` uses the historical all-close E,
`masked` supplies the near/far split from ``nearfar_split_E``.  The evaluated twist
[Ux, Uy, Omega] is asserted equal between the two -- masking is an accuracy-neutral
speedup, not an approximation -- and scored against an untimed all-close lstsq solve.

    pixi run python BIE2D_withJAX/BIEinJAX/test/bench_E_nearfar.py [--obstacles 10] [--reps 5]
"""

from jax import config
config.update("jax_enable_x64", True)

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "periodic"))
sys.path.append(os.path.join(ROOT, "nonperiodic"))
from periodic.structure_pytree import *          # noqa: E402,F401,F403
from nonperiodic.BIOsolve_pytree import *         # noqa: E402,F401,F403

MU = 0.7
R_CONTAINER = 1.0
B1, B2 = 1.23, -0.73


def _get_vslip(B1, B2, x, tang):
    xc = jnp.mean(x)
    xmxc = x - xc
    th = jnp.atan2(jnp.imag(xmxc), jnp.real(xmxc))
    ut = B1 * jnp.sin(th) + B2 * jnp.sin(th) * jnp.cos(th)
    return jnp.concatenate([ut * jnp.real(tang), ut * jnp.imag(tang)])


def _ellipse(a, b, cx, cy, N):
    Zb = lambda t: a * jnp.cos(t) + cx + 1j * (b * jnp.sin(t) + cy)
    Zbp = lambda t: -a * jnp.sin(t) + 1j * (b * jnp.cos(t))
    Zbpp = lambda t: -a * jnp.cos(t) + 1j * (-b * jnp.sin(t))
    bd = channel_wall_func(Zb, N, Zbp, Zbpp)
    bd["a"] = cx + 1j * cy
    bd["theta0"] = 0.0
    bd["radius"] = max(a, b)
    return bd


def _container(np_wall=10, p_wall=10):
    Zc = lambda t: R_CONTAINER * jnp.cos(t) + 1j * R_CONTAINER * jnp.sin(t)
    Zcp = lambda t: -R_CONTAINER * jnp.sin(t) + 1j * R_CONTAINER * jnp.cos(t)
    Zcpp = lambda t: -R_CONTAINER * jnp.cos(t) - 1j * R_CONTAINER * jnp.sin(t)
    return channel_wall_glpanels(Zc, np_wall, p_wall, Zcp, Zcpp)


def _geometry(num_obs, n_obs=40, n_ptcl=60):
    """(wall, obs_cell, ptcl_cell): one swimmer, num_obs obstacles on a ring."""
    s = _container()
    ptcl_cell = {"ptcl_1": _ellipse(0.12, 0.10, 0.0, 0.12, n_ptcl)}
    obs_cell = {}
    ring = 0.62
    for i in range(num_obs):
        ang = 2.0 * np.pi * i / max(num_obs, 1)
        cx, cy = ring * np.cos(ang), ring * np.sin(ang)
        obs_cell[f"obs_{i+1}"] = _ellipse(0.1, 0.1, float(cx), float(cy), n_obs)
    return s, obs_cell, ptcl_cell


def _rhs(s, obs_cell, ptcl_cell):
    lay = rbm_dof_layout(s, obs_cell, ptcl_cell)
    return jnp.concatenate(
        [jnp.zeros((lay["n_wall"] + lay["n_obs"],))]
        + [_get_vslip(B1, B2, pt["x"], pt["nx"] * 1j) for pt in ptcl_cell.values()]
        + [jnp.zeros((lay["n_uom"],))]
    ), lay


def _sync(result):
    """Block on a jax array (or the first element of a tuple) so timings are real."""
    obj = result[0] if isinstance(result, tuple) else result
    if hasattr(obj, "block_until_ready"):
        obj.block_until_ready()
    return result


def _time(fn, reps):
    """Warm once (settle caches), then median wall-ms over `reps` timed calls."""
    _sync(fn())
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _sync(fn())
        samples.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(samples))


def _reference_twist(s, obs_cell, ptcl_cell, rhs, off):
    """Untimed all-close rank-revealing (lstsq) twist -- the honest reference."""
    E, _, _, _, blocks = rbm_wrapper(s, obs_cell, ptcl_cell, MU)
    x, _, _ = solve_rbm_system(E, blocks, rhs, mode=1)
    return np.asarray(x[off:], dtype=float)


def _bench_variant(s, obs_cell, ptcl_cell, rhs, lay, split, reps):
    """Time the three solvers for one split (None = all-close, dict = masked)."""
    off = lay["off_uom"]

    # --- assemble (matrix form, shared by direct + Schur) ---
    def assemble():
        return rbm_wrapper(s, obs_cell, ptcl_cell, MU, split=split)
    t_assemble = _time(assemble, reps)
    E, _, _, _, blocks = rbm_wrapper(s, obs_cell, ptcl_cell, MU, split=split)
    jax.block_until_ready(E)

    # --- direct dense solve (mode 2) ---
    t_direct = _time(lambda: solve_rbm_system(E, blocks, rhs, mode=2)[0], reps)
    x_direct = solve_rbm_system(E, blocks, rhs, mode=2)[0]
    twist = np.asarray(x_direct[off:], dtype=float)

    # --- Schur: static block assembled + factored once, then the reduced solve ---
    def static_build():
        return rbm_static_block(s, obs_cell, MU, split=split)["Eww"]
    t_static = _time(static_build, reps)
    static_fac = factor_static_block(np.asarray(blocks["Eww"]), method="pinv")
    jax.block_until_ready(static_fac["op"])
    t_schur = _time(lambda: solve_schur(blocks, rhs, static_fac=static_fac)[0], reps)
    x_schur = solve_schur(blocks, rhs, static_fac=static_fac)[0]

    # --- matrix-free GMRES (block-Jacobi preconditioned) ---
    op, matvec_fn, _ = make_rbm_linear_operator(s, obs_cell, ptcl_cell, MU, split=split)
    pc = rbm_block_jacobi(s, obs_cell, ptcl_cell, MU)
    v0 = jnp.zeros((lay["n_total"],))
    t_matvec = _time(lambda: matvec_fn(v0), max(reps, 5))
    # gmres timing (warm once)
    solve_gmres_matvec(s, obs_cell, ptcl_cell, MU, rhs, op=op, pc=pc, split=split)
    t0 = time.perf_counter()
    x_g, resid_g, info_g, iters_g, _ = solve_gmres_matvec(
        s, obs_cell, ptcl_cell, MU, rhs, op=op, pc=pc, split=split
    )
    t_gmres = 1e3 * (time.perf_counter() - t0)

    return {
        "twist": twist,
        "twist_schur": np.asarray(x_schur[off:], dtype=float),
        "twist_gmres": np.asarray(x_g[off:], dtype=float),
        "t_assemble": t_assemble,
        "t_direct": t_direct,
        "t_static": t_static,
        "t_schur": t_schur,
        "t_matvec": t_matvec,
        "t_gmres": t_gmres,
        "iters": iters_g,
        "resid_gmres": float(resid_g),
        "info": info_g,
    }


def _print_geometry(title, s, obs_cell, ptcl_cell, reps):
    rhs, lay = _rhs(s, obs_cell, ptcl_cell)
    n = lay["n_total"]
    split = nearfar_split_E(s, obs_cell, ptcl_cell)
    n_pairs = len(split)
    n_masked = sum(1 for v in split.values() if v is not None)
    ref = _reference_twist(s, obs_cell, ptcl_cell, rhs, lay["off_uom"])

    close = _bench_variant(s, obs_cell, ptcl_cell, rhs, lay, None, reps)
    masked = _bench_variant(s, obs_cell, ptcl_cell, rhs, lay, split, reps)

    # correctness: masked must reproduce the close twist
    d_twist = float(np.max(np.abs(masked["twist"] - close["twist"])))
    assert d_twist < 1e-8, f"{title}: masked twist disagrees with close by {d_twist:.2e}"

    print(f"\n{title}   E = {n} x {n}, "
          f"{len(obs_cell)} obstacles, couplings sent to smooth: {n_masked}/{n_pairs} "
          f"(incl. swimmer<->wall)")
    header = (f"{'method':<20}{'close ms':>11}{'masked ms':>11}{'speedup':>9}"
              f"{'|dtwist|':>12}   notes")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    def row(label, ck, mk, twist_close, twist_masked, note=""):
        sp = f"{ck/mk:7.2f}x" if mk > 0 else " " * 8
        dt = float(np.max(np.abs(np.asarray(twist_masked) - ref)))
        print(f"{label:<20}{ck:11.2f}{mk:11.2f}{sp:>9}{dt:12.2e}   {note}")

    # direct = assemble + solve
    row("direct (asm+solve)", close["t_assemble"] + close["t_direct"],
        masked["t_assemble"] + masked["t_direct"], close["twist"], masked["twist"])
    # schur = static build + reduced solve (static factor cached, as in a time loop)
    row("schur (static+solve)", close["t_static"] + close["t_schur"],
        masked["t_static"] + masked["t_schur"], close["twist"], masked["twist_schur"],
        note="static block cached across poses")
    # gmres total
    row("gmres matvec", close["t_gmres"], masked["t_gmres"],
        close["twist_gmres"], masked["twist_gmres"],
        note=f"close {close['iters']} its / masked {masked['iters']} its")
    print("-" * len(header))
    print(f"  assemble E:         close {close['t_assemble']:8.2f} ms   "
          f"masked {masked['t_assemble']:8.2f} ms   "
          f"({close['t_assemble']/max(masked['t_assemble'],1e-9):.2f}x)")
    print(f"  static block build: close {close['t_static']:8.2f} ms   "
          f"masked {masked['t_static']:8.2f} ms   "
          f"({close['t_static']/max(masked['t_static'],1e-9):.2f}x)")
    print(f"  one matvec:         close {close['t_matvec']:8.2f} ms   "
          f"masked {masked['t_matvec']:8.2f} ms   "
          f"({close['t_matvec']/max(masked['t_matvec'],1e-9):.2f}x)")
    print(f"  masked twist == close twist to {d_twist:.2e}; "
          f"|dtwist| scored against untimed all-close lstsq.")
    print(f"  reference twist [Ux, Uy, Omega] = {ref}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--obstacles", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    print(f"near/far-masked E vs all-close E  |  mu = {MU}, reps = {args.reps}")

    s0, obs0, ptcl0 = _geometry(0)
    _print_geometry("EMPTY CONTAINER (control)", s0, obs0, ptcl0, args.reps)

    s1, obs1, ptcl1 = _geometry(args.obstacles)
    _print_geometry(f"{args.obstacles} OBSTACLES", s1, obs1, ptcl1, args.reps)

    print("\nDone.")


if __name__ == "__main__":
    main()
