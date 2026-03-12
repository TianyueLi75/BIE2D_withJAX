"""
stokes_layers.py
================
JAX-based implementation of 2D Stokes single-layer (SLP)
and double-layer (DLP) boundary integral operators.

Based on MATLAB codes by A. Barnett (2016):
    StoSLP.m, StoDLP.m
Translated and refactored into object-oriented JAX Python.

This file is generated with the help of ChatGPT

Author: Choco Li
Date: 2025-11-10
"""

from jax import config
config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
from jax import jit
from scipy.io import loadmat

import sys, os
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the folder containing the module you want to import
utils_path = os.path.join(current_dir, '..')
peri_path = os.path.join(current_dir, '../periodic')
# Add the folder to the system path
sys.path.append(utils_path)
sys.path.append(peri_path)

# jax.clear_caches()

from periodic.structure_jax import channel_wall_func, channel_wall_glpanels

# ======================================================================
# --- Laplace Single-Layer Potential -----------------------------------
# ======================================================================
@jit
def LapSLP(tx, sx, sp, sw, dens):
    results = LapSLPmat(tx, sx, sp, sw)

    # if no density -> return operator matrices
    if dens is None or dens.size == 0:
        return results

    # if density given -> apply matrices
    if len(results) == 2:
        A, An = results
        return A @ dens, An @ dens
    elif len(results) == 5:
        A, An, A11, A12, A22 = results
        u = A @ dens
        un = An @ dens
        uxx = A11 @ dens
        uxy = A12 @ dens
        uyy = A22 @ dens
        return u, un, uxx, uxy, uyy
    else:
        (A,) = results
        return A @ dens,

@jit
def LapSLPmat(tx, sx, sp, sw, *args):
    self_eval = False
    N = sx.size
    M = tx.size

    # displacement matrix
    d = tx[:, None] - sx[None, :]
    r1, r2 = jnp.real(d), jnp.imag(d)
    absd = jnp.abs(d)

    # --- base potential kernel matrix ---
    if self_eval:
        # crude diagonal limit for self-case
        # (Kress quadrature approximation skipped for simplicity)
        log_abs = jnp.log(absd + 1e-15)
        diag_term = -jnp.log(sp + 1e-15)
        A = -log_abs / (2 * jnp.pi)
        idx = jnp.arange(N)
        A = A.at[idx, idx].set(diag_term / (2 * jnp.pi))
        A *= sw
    else:
        A = (-1 / (2 * jnp.pi)) * jnp.log(absd + 1e-15) * sw

    results = [A]

    # --- target normal derivative kernel (dipole form) ---
    # TODO: whether cur can be computed here
    if len(args)>0 :
        tnx = args[0]
        An = (1 / (2 * jnp.pi)) * jnp.real((-tnx[:, None]) / d)
        if len(args) > 1:
            cur = args[1]
            idx = jnp.arange(N)
            An = An.at[idx, idx].set(-cur / (4 * jnp.pi))
        An *= sw
        results.append(An)

    # --- 2nd derivatives (optional) ---
    A11 = (1 / (2 * jnp.pi)) * (r1**2 - r2**2) / (absd**4 + 1e-15)
    A12 = (1 / (2 * jnp.pi)) * (2 * r1 * r2) / (absd**4 + 1e-15)
    A22 = (1 / (2 * jnp.pi)) * (r2**2 - r1**2) / (absd**4 + 1e-15)
    A11 *= sw
    A12 *= sw
    A22 *= sw
    results += [A11, A12, A22]

    return tuple(results)

@jit
def LapSLP_self(tx, sx, sp, sw, dens):
    results = LapSLPmat_self(tx, sx, sp, sw)

    # if no density -> return operator matrices
    if dens is None or dens.size == 0:
        return results

    # if density given -> apply matrices
    if len(results) == 2:
        A, An = results
        return A @ dens, An @ dens
    elif len(results) == 5:
        A, An, A11, A12, A22 = results
        u = A @ dens
        un = An @ dens
        uxx = A11 @ dens
        uxy = A12 @ dens
        uyy = A22 @ dens
        return u, un, uxx, uxy, uyy
    else:
        (A,) = results
        return A @ dens,

@jit
# Helper function with toeplitz for fft in LapSLP self.
def circulant(X):
    X = X.reshape(-1) # X = X(:)
    N = X.shape[0]
    col = jnp.concatenate((X[:1], X[:0:-1]))

    # # jax.debug.print("X= {}", X)
    # # jax.debug.print("col= {}", col)
    
    i = jnp.arange(N)[:, None]
    j = jnp.arange(N)[None, :]
    A = col[(i - j) % N]
    # # jax.debug.print("A= {}", A)

    return A

@jit
def LapSLPmat_self(tx, sx, sp, sw, *args):
    self_eval = True
    N = sx.size
    M = tx.size

    # displacement matrix
    d = tx[:, None] - sx[None, :]
    r1, r2 = jnp.real(d), jnp.imag(d)
    absd = jnp.abs(d)

    # # jax.debug.print("sp = {}", sp)

    # --- base potential kernel matrix ---
    if self_eval:

        # Find diagonal limit using Kress rule

        A = -jnp.log(absd) + circulant(0.5*jnp.log(4*jnp.sin(jnp.pi*(jnp.arange(N)/N))**2))
        # # jax.debug.print("A = -log = {}", A)

        idx = jnp.arange(N)
        A = A.at[idx, idx].set(-jnp.log(sp))

        # # jax.debug.print("A sub diag terms = {}", A)

        m = jnp.arange(N/2-1)+1
        # # jax.debug.print("m = {}", m)

        Rjn = jnp.fft.ifft(jnp.concatenate([
            jnp.array([0.]),
            1. / m,
            jnp.array([2./N]),
            1./m[::-1]
        ])) / 2.

        # # jax.debug.print("Rjn = {}", Rjn)

        A /= N
        A += circulant(Rjn)

        # # jax.debug.print("A added Rjn = {}", A)

        A *= sp

        # # jax.debug.print("A *sp = {}", A)

    else:
        A = (-1 / (2 * jnp.pi)) * jnp.log(absd + 1e-15) * sw

    A = jnp.real(A) # to avoid numerical instability causing complex log values when N large.

    results = [A]

    # --- target normal derivative kernel (dipole form) ---
    if len(args) > 0: # tnx and scur are included in input
        tnx = args[0]
        An = (1 / (2 * jnp.pi)) * jnp.real((-tnx[:, None]) / d)
        if len(args) > 1:
            cur = args[1]
            idx = jnp.arange(N)
            An = An.at[idx, idx].set(-cur / (4 * jnp.pi))
        An *= sw
        results.append(An)

    # --- 2nd derivatives (optional) ---
    A11 = (1 / (2 * jnp.pi)) * (r1**2 - r2**2) / (absd**4 + 1e-15)
    A12 = (1 / (2 * jnp.pi)) * (2 * r1 * r2) / (absd**4 + 1e-15)
    A22 = (1 / (2 * jnp.pi)) * (r2**2 - r1**2) / (absd**4 + 1e-15)
    A11 *= sw
    A12 *= sw
    A22 *= sw
    results += [A11, A12, A22]

    return tuple(results)


# ----------------------------------------------------------------------
# --- Stokes Single-Layer Potential (SLP) -------------------------------
# ----------------------------------------------------------------------

@jit
def StoSLP(tx, tnx, sx, snx, sxp, scur, sw, mu, dens):

    if dens is None:
        dens = jnp.array([])

    u, p, T = StoSLPmat(tx, tnx, sx, snx, sxp, scur, sw, mu)
    if dens.size > 0:
        u = u @ dens
        p = p @ dens
        T = T @ dens
    return u, p, T

@jit
def StoSLPmat(tx, tnx, sx, snx, sxp, scur, sw, mu):
    # jax.debug.print("\n in SLPmat far eval", ordered=True)
    self_eval = False
    N = sx.size
    M = tx.size

    r = tx[:, None] - sx[None, :]
    # # jax.debug.print("{r}",r=r)
    irr = 1.0 / (jnp.conj(r) * r)
    d1, d2 = jnp.real(r), jnp.imag(r)
    c = 1.0 / (4.0 * jnp.pi * mu)
    irr = jnp.real(irr)

    # Compute some geometric properties on the fly
    sp = jnp.abs(sxp)
    stang = sxp/sp

    # --- Velocity matrix ---
    if self_eval:
        [S,_,_,_] = LapSLP_self(sx, sx, sp, sw, jnp.array([]))
        A = jnp.kron(jnp.eye(2) / (2 * mu), S)
        t1, t2 = jnp.real(stang), jnp.imag(stang)
        A11 = d1**2 * irr
        A12 = d1 * d2 * irr
        A22 = d2**2 * irr
        idx = jnp.arange(N)
        A11 = A11.at[idx, idx].set(t1**2)
        A12 = A12.at[idx, idx].set(t1 * t2)
        A22 = A22.at[idx, idx].set(t2**2)
        A += c * jnp.block([[A11, A12], [A12, A22]]) * jnp.concatenate([sw, sw])
    else:
        logir = -jnp.log(jnp.abs(r))
        A12 = d1 * d2 * irr
        A = c * jnp.block([
            [logir + d1**2 * irr, A12],
            [A12, logir + d2**2 * irr]
        ]) * jnp.concatenate([sw, sw])

    # --- Pressure matrix ---
    P = jnp.block([d1 * irr, d2 * irr])
    P *= (1.0 / (2 * jnp.pi)) * jnp.concatenate([sw, sw])

    # --- Traction matrix ---
    # rdotn = d1[:, None] * jnp.real(tnx) + d2[:, None] * jnp.imag(tnx)
    rdotn = d1 * jnp.real(tnx)[:, None] + d2 * jnp.imag(tnx)[:, None] 
    rdotnir4 = rdotn * irr * irr
    A12 = (-1.0 / jnp.pi) * d1 * d2 * rdotnir4
    T = jnp.block([
        [(-1.0 / jnp.pi) * d1**2 * rdotnir4, A12],
        [A12, (-1.0 / jnp.pi) * d2**2 * rdotnir4]
    ])

    if self_eval:
        cdiag = -scur / (2 * jnp.pi)
        tx = 1j * snx
        t1, t2 = jnp.real(tx), jnp.imag(tx)
        idx = jnp.arange(N)
        T = T.at[idx, idx].set(cdiag * t1**2)
        T = T.at[idx + N, idx].set(cdiag * t1 * t2)
        T = T.at[idx, idx + N].set(cdiag * t1 * t2)
        T = T.at[idx + N, idx + N].set(cdiag * t2**2)
    T *= jnp.concatenate([sw, sw])

    return A, P, T


@jit
def StoSLP_self(tx, tnx, sx, snx, sxp, scur, sw, mu, dens):

    if dens is None:
        dens = jnp.array([])

    u, p, T = StoSLPmat_self(tx, tnx, sx, snx, sxp, scur, sw, mu)
    if dens.size > 0:
        u = u @ dens
        p = p @ dens
        T = T @ dens
    return u, p, T

@jit
def StoSLPmat_self(tx, tnx, sx, snx, sxp, scur, sw, mu):
    # jax.debug.print("\n in SLPmat selfeval", ordered=True)
    self_eval = True
    N = sx.size
    M = tx.size

    r = tx[:, None] - sx[None, :]
    irr = 1.0 / (jnp.conj(r) * r)
    irr = jnp.real(irr)
    d1, d2 = jnp.real(r), jnp.imag(r)
    c = 1.0 / (4.0 * jnp.pi * mu)

    # Compute some geometric properties on the fly
    sp = jnp.abs(sxp)
    stang = sxp/sp

    # --- Velocity matrix ---
    if self_eval:
        [S,_,_,_] = LapSLP_self(sx, sx, sp, sw, jnp.array([]))
        
        # print("======= DEBUG: Check LapSLP self ========= ")
        # # jax.debug.print("S = {}", S)

        A = jnp.kron(jnp.eye(2) / (2 * mu), S)
        t1, t2 = jnp.real(stang), jnp.imag(stang)
        A11 = d1**2 * irr
        A12 = d1 * d2 * irr
        A22 = d2**2 * irr
        idx = jnp.arange(N)
        A11 = A11.at[idx, idx].set(t1**2)
        A12 = A12.at[idx, idx].set(t1 * t2)
        A22 = A22.at[idx, idx].set(t2**2)

        # print("S shape: ",S.shape, ", A shape: ", A.shape,", A11 shape: ", A11.shape, ", sw shape: ", sw.shape)
        A += c * jnp.block([[A11, A12], [A12, A22]]) * jnp.concatenate([sw, sw])

    else:
        logir = -jnp.log(jnp.abs(r))
        A12 = d1 * d2 * irr
        A = c * jnp.block([
            [logir + d1**2 * irr, A12],
            [A12, logir + d2**2 * irr]
        ]) * jnp.concatenate([sw, sw])

    # TODO: Check P and T self, also might need to check for DL self...
    # TODO: turn into unit tests to compare with premade matlab files.
    
    # --- Pressure matrix ---
    P = jnp.block([d1 * irr, d2 * irr])
    P *= (1.0 / (2 * jnp.pi)) * jnp.concatenate([sw, sw])

    # --- Traction matrix ---
    # rdotn = d1[:, None] * jnp.real(tnx) + d2[:, None] * jnp.imag(tnx)
    rdotn = d1 * jnp.real(tnx)[:, None] + d2 * jnp.imag(tnx)[:, None] 
    rdotnir4 = rdotn * irr * irr
    A12 = (-1.0 / jnp.pi) * d1 * d2 * rdotnir4
    T = jnp.block([
        [(-1.0 / jnp.pi) * d1**2 * rdotnir4, A12],
        [A12, (-1.0 / jnp.pi) * d2**2 * rdotnir4]
    ])

    if self_eval:
        cdiag = -scur / (2 * jnp.pi)
        tx = 1j * snx
        t1, t2 = jnp.real(tx), jnp.imag(tx)
        idx = jnp.arange(N)
        T = T.at[idx, idx].set(cdiag * t1**2)
        T = T.at[idx + N, idx].set(cdiag * t1 * t2)
        T = T.at[idx, idx + N].set(cdiag * t1 * t2)
        T = T.at[idx + N, idx + N].set(cdiag * t2**2)
    T *= jnp.concatenate([sw, sw])

    return A, P, T


# ----------------------------------------------------------------------
# --- Stokes Double-Layer Potential (DLP) -------------------------------
# ----------------------------------------------------------------------
@jit
def StoDLP(tx, tnx, sx, snx, sxp, scur, sw, mu, dens):
    """Evaluate 2D Stokes double-layer velocity, pressure, and traction."""
    if dens is None:
        dens = jnp.array([])

    u, p, T = StoDLPmat(tx, tnx, sx, snx, sxp, scur, sw, mu)
    if dens.size > 0:
        u = u @ dens
        p = p @ dens
        T = T @ dens
    return u, p, T

@jit
def StoDLPmat(tx, tnx, sx, snx, sxp, scur, sw, mu):
    # jax.debug.print("\n in DLPmat Far eval", ordered=True)
    self_eval = False
    """Build dense matrices for double-layer velocity, pressure, and traction."""
    N = sx.size
    M = tx.size

    r = tx[:, None] - sx[None, :]
    irr = 1.0 / (jnp.conj(r) * r)
    d1, d2 = jnp.real(r), jnp.imag(r)
    irr = jnp.real(irr)

    rdotny = d1 * jnp.real(snx)[None, :] + d2 * jnp.imag(snx)[None, :]
    rdotnir4 = rdotny * (irr * irr)

    # --- Velocity matrix ---
    A12 = (1.0 / jnp.pi) * d1 * d2 * rdotnir4
    A = jnp.block([
        [(1.0 / jnp.pi) * d1**2 * rdotnir4, A12],
        [A12, (1.0 / jnp.pi) * d2**2 * rdotnir4]
    ])
    if self_eval:
        cdiag = -scur / (2 * jnp.pi)
        tx = 1j * snx
        t1, t2 = jnp.real(tx), jnp.imag(tx)
        idx = jnp.arange(N)
        A = A.at[idx, idx].set(cdiag * t1**2)
        A = A.at[idx + N, idx].set(cdiag * t1 * t2)
        A = A.at[idx, idx + N].set(cdiag * t1 * t2)
        A = A.at[idx + N, idx + N].set(cdiag * t2**2)
        
    A *= jnp.concatenate([sw, sw])

    # --- Pressure matrix ---
    P = jnp.block([
        [jnp.real(-snx)[None, :] * irr + 2 * rdotnir4 * d1,
         jnp.imag(-snx)[None, :] * irr + 2 * rdotnir4 * d2]
    ])
    P *= (mu / jnp.pi) * jnp.concatenate([sw, sw])

    # --- Traction matrix ---
    nx1 = jnp.real(tnx)[:, None]
    nx2 = jnp.imag(tnx)[:, None]
    ny1 = jnp.real(snx)[None, :]
    ny2 = jnp.imag(snx)[None, :]

    rdotnx = d1 * nx1 + d2 * nx2
    dx = rdotnx * irr
    dy = rdotny * irr
    dxdy = dx * dy

    R12 = d1 * d2 * irr
    R = jnp.block([
        [d1**2 * irr, R12],
        [R12, d2**2 * irr]
    ])
    nydotnx = nx1 * ny1 + nx2 * ny2

    T = R * jnp.kron(jnp.ones((2, 2)), nydotnx * irr - 8 * dxdy) \
        + jnp.kron(jnp.eye(2), dxdy)
    T += jnp.block([
        [nx1 * ny1 * irr, nx1 * ny2 * irr],
        [nx2 * ny1 * irr, nx2 * ny2 * irr]
    ])
    T += jnp.kron(jnp.ones((2, 2)), dx * irr) * jnp.block([
        [ny1 * d1, ny1 * d2],
        [ny2 * d1, ny2 * d2]
    ])
    T += jnp.kron(jnp.ones((2, 2)), dy * irr) * jnp.block([
        [d1 * nx1, d1 * nx2],
        [d2 * nx1, d2 * nx2]
    ])
    T *= (mu / jnp.pi) * jnp.concatenate([sw, sw])

    return A, P, T

@jit
def StoDLP_self(tx, tnx, sx, snx, sxp, scur, sw, mu, dens):
    """Evaluate 2D Stokes double-layer velocity, pressure, and traction."""
    if dens is None:
        dens = jnp.array([])

    u, p, T = StoDLPmat_self(tx, tnx, sx, snx, sxp, scur, sw, mu)
    if dens.size > 0:
        u = u @ dens
        p = p @ dens
        T = T @ dens
    return u, p, T

@jit
def StoDLPmat_self(tx, tnx, sx, snx, sxp, scur, sw, mu):
    # jax.debug.print("\n in DLPmat selfeval", ordered=True)
    self_eval = True
    """Build dense matrices for double-layer velocity, pressure, and traction."""
    N = sx.size
    M = tx.size

    r = tx[:, None] - sx[None, :]
    irr = 1.0 / (jnp.conj(r) * r)
    d1, d2 = jnp.real(r), jnp.imag(r)
    irr = jnp.real(irr)

    rdotny = d1 * jnp.real(snx)[None, :] + d2 * jnp.imag(snx)[None, :]
    rdotnir4 = rdotny * (irr * irr)

    # --- Velocity matrix ---
    A12 = (1.0 / jnp.pi) * d1 * d2 * rdotnir4
    A = jnp.block([
        [(1.0 / jnp.pi) * d1**2 * rdotnir4, A12],
        [A12, (1.0 / jnp.pi) * d2**2 * rdotnir4]
    ])
    if self_eval:
        cdiag = -scur / (2 * jnp.pi)
        tx = 1j * snx
        t1, t2 = jnp.real(tx), jnp.imag(tx)
        idx = jnp.arange(N)
        A = A.at[idx, idx].set(cdiag * t1**2)
        A = A.at[idx + N, idx].set(cdiag * t1 * t2)
        A = A.at[idx, idx + N].set(cdiag * t1 * t2)
        A = A.at[idx + N, idx + N].set(cdiag * t2**2)
    A *= jnp.concatenate([sw, sw])


    # jax.debug.print("after A", ordered=True)

    # --- Pressure matrix ---
    P = jnp.block([
        [jnp.real(-snx)[None, :] * irr + 2 * rdotnir4 * d1,
         jnp.imag(-snx)[None, :] * irr + 2 * rdotnir4 * d2]
    ])
    P *= (mu / jnp.pi) * jnp.concatenate([sw, sw])
    # jax.debug.print("after P", ordered=True)

    # --- Traction matrix ---
    nx1 = jnp.real(tnx)[:, None]
    nx2 = jnp.imag(tnx)[:, None]
    ny1 = jnp.real(snx)[None, :]
    ny2 = jnp.imag(snx)[None, :]

    rdotnx = d1 * nx1 + d2 * nx2
    dx = rdotnx * irr
    dy = rdotny * irr
    dxdy = dx * dy

    R12 = d1 * d2 * irr
    R = jnp.block([
        [d1**2 * irr, R12],
        [R12, d2**2 * irr]
    ])
    nydotnx = nx1 * ny1 + nx2 * ny2

    T = R * jnp.kron(jnp.ones((2, 2)), nydotnx * irr - 8 * dxdy) \
        + jnp.kron(jnp.eye(2), dxdy)
    T += jnp.block([
        [nx1 * ny1 * irr, nx1 * ny2 * irr],
        [nx2 * ny1 * irr, nx2 * ny2 * irr]
    ])
    T += jnp.kron(jnp.ones((2, 2)), dx * irr) * jnp.block([
        [ny1 * d1, ny1 * d2],
        [ny2 * d1, ny2 * d2]
    ])
    T += jnp.kron(jnp.ones((2, 2)), dy * irr) * jnp.block([
        [d1 * nx1, d1 * nx2],
        [d2 * nx1, d2 * nx2]
    ])
    T *= (mu / jnp.pi) * jnp.concatenate([sw, sw])
    # jax.debug.print("after T", ordered=True)

    return A, P, T

# ----------------------------------------------------------------------
# --- Close evaluation funtions ----------------------------------------------
# ----------------------------------------------------------------------
def cau_closeglobal(x, sx, swxp, vb, side='i'):
    """
    JAX implementation of Globally compensated barycentric Cauchy integral.
    
    Args:
        x:   M target points (complex128)
        sx:  N quadrature nodes (complex128)
        swxp: N complex weights (s.cw in MATLAB)
        vb:  N x Nc boundary values
        side: 'i' for interior, 'e' for exterior
        
    Returns:
        vc, vcp, vcpp (Value, 1st Deriv, 2nd Deriv)
    """
    N = sx.shape[0]
    # Ensure vb is 2D (N, Nc)
    vb = jnp.atleast_2d(vb)
    if vb.shape[0] != N: # Handle case where vb might be (Nc, N)
        vb = vb.T

    # 1. Distance Matrix and Cauchy Kernel
    # Use jnp.where to avoid division by zero during the kernel calculation
    diff = sx[:, jnp.newaxis] - x[jnp.newaxis, :]
    is_hit = jnp.isclose(diff, 0, atol=1e-15)
    
    # Replace zeros with 1.0 temporarily to avoid Inf/NaN in 'comp'
    # We will overwrite these 'hit' values later anyway.
    safe_diff = jnp.where(is_hit, 1.0, diff)
    comp = swxp[:, jnp.newaxis] / safe_diff
    # Zero out the 'fake' values we created for the sum
    comp = jnp.where(is_hit, 0.0, comp)

    # 2. Compute Values (vc)
    I0 = jnp.matmul(vb.T, comp).T
    J0 = jnp.sum(comp, axis=0)[:, jnp.newaxis]
    
    if side == 'e':
        J0 = J0 - 2j * jnp.pi
        
    vc = I0 / J0
    
    # Replace hits using JAX select (jit-safe)
    node_indices = jnp.argmax(is_hit, axis=0)
    vc = jnp.where(jnp.any(is_hit, axis=0)[:, jnp.newaxis], vb[node_indices], vc)

    # 3. First Derivative (vcp) logic
    # Schneider-Werner Cauchy matrix
    dist_nodes = sx[:, jnp.newaxis] - sx[jnp.newaxis, :]
    Y = 1.0 / jnp.where(jnp.eye(N, dtype=bool), 1.0, dist_nodes)
    Y = jnp.where(jnp.eye(N, dtype=bool), 0.0, Y)
    Y = Y * swxp[:, jnp.newaxis]
    
    diag_vals = -jnp.sum(Y, axis=0)
    Y = Y + jnp.diag(diag_vals)
    
    vbp = jnp.matmul(Y.T, vb)
    if side == 'e':
        vbp = vbp + 2j * jnp.pi * vb
    vbp = (-1.0 / swxp[:, jnp.newaxis]) * vbp
    
    vcp = jnp.matmul(vbp.T, comp).T / J0
    vcp = jnp.where(jnp.any(is_hit, axis=0)[:, jnp.newaxis], vbp[node_indices], vcp)

    # 4. Second Derivative (vcpp)
    vbpp = jnp.matmul(Y.T, vbp)
    if side == 'e':
        vbpp = vbpp + 2j * jnp.pi * vbp
    vbpp = (-1.0 / swxp[:, jnp.newaxis]) * vbpp
    
    vcpp = jnp.matmul(vbpp.T, comp).T / J0
    vcpp = jnp.where(jnp.any(is_hit, axis=0)[:, jnp.newaxis], vbpp[node_indices], vcpp)

    return vc, vcp, vcpp

def cau_closepanel(tx, sx, swxp, a, b, side='i'):
    """
    JAX implementation of complex DLP special quadrature (Helsing).
    Args:
        tx: Target points (M,)
        sx: Source nodes (p,)
        swxp: Complex speed weights (p,)
        a, b: Panel endpoints (complex)
        side: 'i' or 'e'
    """
    p = sx.shape[0]
    M = tx.shape[0]
    
    # Rescaling factors
    zsc = (b - a) / 2.0
    zmid = (b + a) / 2.0
    y = (sx - zmid) / zsc
    x = (tx - zmid) / zsc
    
    # Precompute constants c_k = (1 - (-1)^k) / k
    ks = jnp.arange(1, p + 1)
    c = (1.0 - (-1.0)**ks) / ks
    
    # Vandermonde matrix V (p, p)
    V = jnp.vander(y, p, increasing=True).T

    # Phase for branch cut
    gam = jnp.exp(1j * jnp.pi / 4.0)
    if side == 'e':
        gam = jnp.conj(gam)

    # Initialize P_1 for all targets
    p1 = jnp.log(gam) + jnp.log((1.0 - x) / (gam * (-1.0 - x)))
    
    # --- Upward Recurrence (Near targets) ---
    def upward_step(p_prev, ck):
        p_next = x * p_prev + ck
        return p_next, p_next

    _, p_up = jax.lax.scan(upward_step, p1, c[:-1])
    P_near = jnp.vstack([p1, p_up]) # Shape (p, M)

    # --- Downward Recurrence (Far targets) ---
    # 1. Compute P_p directly via quadrature for far targets
    wxp = swxp / zsc
    diff = y[:, jnp.newaxis] - x[jnp.newaxis, :]
    # Basic Cauchy sum for the last order p
    pp_far = jnp.sum((wxp[:, jnp.newaxis] * V[-1:, :].T) / diff, axis=0)

    def downward_step(p_next, ck):
        p_curr = (p_next - ck) / x
        return p_curr, p_curr

    # Scan backwards from p-1 down to 2
    # We flip c and use indices p-1 to 2
    _, p_down = jax.lax.scan(downward_step, pp_far, c[1:-1][::-1])
    # p_down contains P_2...P_{p-1}. We need to pad P_1 and P_p.
    P_far = jnp.vstack([p1, p_down[::-1], pp_far])

    # --- Combine Near and Far ---
    d = 1.1
    is_near = jnp.abs(x) <= d
    P = jnp.where(is_near, P_near, P_far)

    # Solve for weights: A = ((V.T \ P).T * (1i / (2*pi)))
    # Note: jnp.linalg.solve is used for V.T \ P
    A_coeffs = jnp.linalg.solve(V, P)
    A = A_coeffs.T * (1j / (2.0 * jnp.pi)) 

    # --- Derivatives ---
    # R (hypersingular)
    ones_p = jnp.ones((p, 1))
    inv_dist_pos = 1.0 / (1.0 - x)
    inv_dist_neg = 1.0 / (1.0 + x)
    
    # Helsing 2009 eqn (14)
    term1 = -(ones_p @ inv_dist_pos[jnp.newaxis, :] + 
              ((-1.0)**jnp.arange(p))[:, jnp.newaxis] @ inv_dist_neg[jnp.newaxis, :])
    
    # Shifted P for recurrence part of R
    P_shifted = jnp.vstack([jnp.zeros((1, M)), P[:-1, :]])
    R = term1 + jnp.arange(p)[:, jnp.newaxis] * P_shifted
    
    Az = jnp.linalg.solve(V, R).T * (1j / (2.0 * jnp.pi * zsc))

    # S (supersingular)
    term1_s = -(ones_p @ (inv_dist_pos**2)[jnp.newaxis, :] - 
                ((-1.0)**jnp.arange(p))[:, jnp.newaxis] @ (inv_dist_neg**2)[jnp.newaxis, :]) / 2.0
    R_shifted = jnp.vstack([jnp.zeros((1, M)), R[:-1, :]])
    S = term1_s + jnp.arange(p)[:, jnp.newaxis] * R_shifted / 2.0
    
    Azz = jnp.linalg.solve(V, S).T * (1j / (2.0 * jnp.pi * zsc**2))

    # Output mapping
    A1 = jnp.real(Az)
    A2 = -jnp.imag(Az)
    A3 = jnp.real(Azz)
    A4 = -jnp.imag(Azz)

    return A, A1, A2, A3, A4

cau_closeglobal = jit(cau_closeglobal, static_argnums=(4,))
cau_closepanel = jit(cau_closepanel, static_argnums=(5,))

def CSLPselfmatrix(sx, st, sxp, side='i'):
    N = sx.shape[0]
    ssp = jnp.abs(sxp)
    
    # Distance matrices
    d = sx[:, None] - sx[None, :]
    x_circ = jnp.exp(1j * st)
    dx_circ = x_circ[:, None] - x_circ[None, :]
    
    # S matrix with Kress splitting
    S = -jnp.log(d) + jnp.log(dx_circ)
    diag_indices = jnp.diag_indices(N)
    S = S.at[diag_indices].set(jnp.log(1j * x_circ / sxp))
    
    # Phase jump removal across the matrix (Flattened vectorized approach)
    S_f = S.ravel()
    p = jnp.imag(jnp.diff(S_f))
    jumps = 2j * jnp.pi * jnp.cumsum(jnp.round(p / (2 * jnp.pi)))
    S_f = S_f.at[1:].add(-jumps)
    S = S_f.reshape((N, N))

    # Construct Circulant Rjn
    m = jnp.arange(1, N // 2)
    if side == 'e':
        r_vals = jnp.concatenate([jnp.array([0.0]), 1.0/m, jnp.array([1.0/N]), 0.0*m])
    else:
        r_vals = jnp.concatenate([jnp.array([0.0]), 0.0*m, jnp.array([1.0/N]), 1.0/m[::-1]])
    
    Rjn = jnp.fft.ifft(r_vals)
    
    circ_mat = circulant(Rjn)
    
    return (S / N + circ_mat) * ssp[None, :]

def lapSLP_closeglobal(x, sx, sxp, st, sa, sws, tau, side='i'):
    """
    Returns potential and derivatives at targets x due to single layer potential with real-valued density tau sampled at sx. 
    Uses global barycentric Cauchy close-evaluation.
    u(x) = (1/2pi) int_Gamma log(1/|r|) tau(y) ds_y,  where r:=x-y,  x,y in R2

    Inputs:
        x = target (as points in complex plane)
        sx, st, sws = source nodes (as complex points) and geometric properties
        sa = a point interior to the curve, used if side='e'
        tau = density sampled at sx, real valued, cannot be empty. apply u(x) for each column of tau.
        side = 'i' or 'e': interior or exterior problem. For particles, side = 'e'.

    Outputs:
        u = column vectors of Laplace SL from each column of tau.
        ux, uy = x and y partial derivatives of u
        uxx,uyy,uxy = second derivatives of u
    """
    N = sx.shape[0]
    tau = jnp.atleast_2d(tau)
    if tau.shape[0] != N: # Handle case where vb might be (Nc, N)
        tau = tau.T

    # Compute geometric properties
    ssp = jnp.abs(sxp)
    snx = -1j * (sxp / ssp)
    sw = sws / ssp
    swxp = sw * sxp
    
    vb = CSLPselfmatrix(sx, st, sxp, side) @ tau
    if side=='e':
        sawlog = st / 1j + jnp.log(sa - sx)
        p = jnp.imag(jnp.diff(sawlog))
        jumps = 2j * jnp.pi * jnp.cumsum(jnp.round(p / (2 * jnp.pi)))
        sawlog = sawlog.at[1:].add(-jumps)

        totchgp = jnp.sum(sws[:, None] * tau, axis=0) / (2 * jnp.pi)
        vb = vb + sawlog[:, None] * totchgp
        
        cw = 1j * snx * sws
        vinf = jnp.sum(vb * (cw / (sx - sa))[:, None], axis=0) / (2j * jnp.pi)
        vb = vb - vinf[None, :]

    [v, vp, vpp] = cau_closeglobal(x, sx, swxp, vb, side)
    # Potentials and partials
    u = jnp.real(v)
    ux = jnp.real(vp)
    uy = -jnp.imag(vp)
    uxx = jnp.real(vpp)
    uxy = -jnp.imag(vpp)
    uyy = -jnp.real(vpp)
    
    if side == 'e':
        inv_dist = 1.0 / (sa - x)
        inv_dist_sq = 1.0 / (sa - x)**2
        
        u = u - jnp.log(jnp.abs(sa - x))[:, None] * totchgp + jnp.real(vinf)[None, :]
        ux = ux + jnp.real(inv_dist)[:, None] * totchgp
        uy = uy - jnp.imag(inv_dist)[:, None] * totchgp
        
        uxx = uxx + jnp.real(inv_dist_sq)[:, None] * totchgp
        uxy = uxy - jnp.imag(inv_dist_sq)[:, None] * totchgp
        uyy = uyy - jnp.real(inv_dist_sq)[:, None] * totchgp

    return u, ux, uy, uxx, uxy, uyy

# SHOULD be the same as function in structure_jax.py.
def perispecdiff(f):
    N = f.shape[0]
    if N % 2 == 0:
        # wavenumber vector for even N
        k = jnp.concatenate([jnp.arange(0, N//2), jnp.array([0]), jnp.arange(-N//2 + 1, 0)])
    else:
        k = jnp.concatenate([jnp.arange(0, (N-1)//2 + 1), jnp.arange(-(N-1)//2, 0)])
        
    if jnp.isrealobj(f):
        return jnp.real(jnp.fft.ifft(1j * k * jnp.fft.fft(f)))
    else:
        return jnp.fft.ifft(1j * k * jnp.fft.fft(f))

def lapDLP_closeglobal(x, sx, swxp, tau, side='i'):
    """
    JAX JIT-compatible Laplace DLP potential and derivatives.
    
    Args:
        x: Target points (complex M)
        sx, snx, sxp, swxp: Curve arrays (complex/float N)
        tau: Density (real N x n)
        side: 'i' (interior) or 'e' (exterior)
    """
    N = sx.shape[0]
    tau = jnp.atleast_2d(tau)
    if tau.shape[0] != N: # Handle case where vb might be (Nc, N)
        tau = tau.T
    
    # 1. Evaluate boundary limits of the complex DLP holom. function (vb)
    # Numerical derivative of density: tau'
    taup = jax.vmap(perispecdiff, in_axes=1, out_axes=1)(tau)
    
    # Construct Cauchy matrix Y_{ij} = w_j / (x_i - x_j)
    # Using broadcasting for N x N matrix
    diff_mat = sx[:, None] - sx[None, :]
    # Avoid division by zero on diagonal for now
    Y = 1.0 / (diff_mat + jnp.eye(N)) 
    # print("Y before cw: ", Y)
    Y = Y * swxp[:, None] # include complex weights over columns
    
    # Set diagonal to -sum_{j != i} Y_{ij}
    diag_vals = -jnp.sum(Y * (1.0 - jnp.eye(N)), axis=1)
    Y = Y * (1.0 - jnp.eye(N)) + jnp.diag(diag_vals)
    
    # vb = Y * tau / (-2i*pi) - tau' / (i*N)
    vb = (Y.T @ tau) * (1.0 / (-2j * jnp.pi))
    vb = vb - (1.0 / (1j * N)) * taup

    # Jump condition: v_minus = v_plus - tau
    if side == 'i':
        vb = vb - tau

    # 2. Compensated close-evaluation
    [v, vp, _] = cau_closeglobal(x, sx, swxp, vb, side)
    
    u = jnp.real(v)
    ux = jnp.real(vp)
    uy = -jnp.imag(vp)

    # jax.debug.print("row dim of u: {}", u.shape[0])
    
    return u, ux, uy

lapSLP_closeglobal = jit(lapSLP_closeglobal, static_argnums=(7,))
lapDLP_closeglobal = jit(lapDLP_closeglobal, static_argnums=(4,))

def perispecinterp(f, N):
    """
    JAX implementation of periodic spectral interpolation, 1D.
    N must be a static Python integer.
    """
    # fft defaults to last axis; we assume f is (n, ) or (n, cols)
    # If f has multiple columns, vmap over them
    if f.ndim > 1:
        return jax.vmap(_perispecinterp_core, in_axes=(1,None), out_axes=1)(f, N)
    else:
        return _perispecinterp_core(f, N)
    
def _perispecinterp_core(f, N):
    n = f.shape[0]
    if N == n:
        return f

    F = jnp.fft.fft(f)
    
    if N > n:
        # Upsample logic: split the Nyquist frequency
        # F[n//2] is the Nyquist frequency component
        left = F[:n//2]
        nyquist = F[n//2] / 2.0
        middle = jnp.zeros(N - n - 1, dtype=F.dtype)
        right = F[n//2 + 1:]
        
        # Construct padded spectrum: [low-freq, nyq/2, zeros, nyq/2, high-freq]
        F_padded = jnp.concatenate([
            left, 
            jnp.array([nyquist]), 
            middle, 
            jnp.array([nyquist]), 
            right
        ])
        g = jnp.fft.ifft(F_padded)
    else:
        # Downsample logic: average the Nyquist frequency
        left = F[:N//2]
        # Average the components that alias to the new Nyquist frequency
        nyquist = (F[N//2] + F[n - N//2]) / 2.0
        right = F[n - N//2 + 1:]
        
        F_cropped = jnp.concatenate([
            left, 
            jnp.array([nyquist]), 
            right
        ])
        g = jnp.fft.ifft(F_cropped)
        
    return g * (N / n)

def stoSLP_closeglobal(x, nx, sx, sxp, st, sa, sws, sigma_real, mu, side='e'):
    """
    JAX implementation of Stokes SLP velocity, pressure, and traction.
    
    Args:
        sigma_real: (2N, n) density matrix [real_part; imag_part]
        mu: viscosity (float)
    """
    N = sx.shape[0]
    
    # Convert real 2N-by-n density to complex N-by-n
    sigma_real = jnp.atleast_2d(sigma_real)
    if sigma_real.shape[0] != N:
        sigma_real = sigma_real.T
    sigma = sigma_real[:N, :] + 1j * sigma_real[N:, :]

    # Compute geometric properties
    ssp = jnp.abs(sxp)
    snx = -1j * (sxp / ssp)
    sw = sws / ssp
    
    # --- Velocity calculation (Step 1) ---
    # SLP with real part of sigma
    I1x1, I3x1, I3x2, s1xx, s1xy, s1yy = lapSLP_closeglobal(
        x, sx, sxp, st, sa, sws, jnp.real(sigma), side
    )
    # SLP with imag part of sigma
    I1x2, I4x1, I4x2, s2xx, s2xy, s2yy = lapSLP_closeglobal(
        x, sx, sxp, st, sa, sws, jnp.imag(sigma), side
    )
    I1 = (I1x1 + 1j * I1x2) / 2.0
    
    # SLP with tau = Re(y * conj(sigma))
    tau_dens = jnp.real(sx[:, None] * jnp.conj(sigma))
    _, I2x1, I2x2, stxx, stxy, styy = lapSLP_closeglobal(
        x, sx, sxp, st, sa, sws, tau_dens, side
    )
    I2 = (I2x1 + 1j * I2x2) / 2.0
    
    # Combining components for velocity u
    I3 = (jnp.real(x) / 2.0)[:, None] * (I3x1 + 1j * I3x2)
    I4 = (jnp.imag(x) / 2.0)[:, None] * (I4x1 + 1j * I4x2)
    
    u_cmplx = (1.0 / mu) * (I1 + I2 - I3 - I4)
    u = jnp.vstack([jnp.real(u_cmplx), jnp.imag(u_cmplx)])
    
    # --- Pressure calculation (Step 2) ---
    # Pressure uses a Laplace DLP with a resampled/rotated density
    # Here we assume a fixed beta=2 for simplicity in JIT (static Nf = 2*N)
    Nf = 2*N
    sf_x  = perispecinterp(sx, Nf)
    sf_xp = perispecinterp(sxp, Nf)
    sf_nx = perispecinterp(snx, Nf)
    sf_nx = sf_nx / jnp.abs(sf_nx)
    sf_swxp = 2*jnp.pi / Nf * sf_xp # TODO: hardcodes global uniform grid weight..

    sigf = perispecinterp(sigma, Nf)
    tau_f = sigf / sf_nx[:, None] # rotation by n_y for pressure --- this requires the upsampled grid.

    # Re-use the LapDLP logic from earlier
    p_val, _, _ = lapDLP_closeglobal(x, sf_x, sf_swxp, tau_f, side)
    p = jnp.real(p_val)
    
    # --- Traction calculation (Step 3) ---
    x1, x2 = jnp.real(x)[:, None], jnp.imag(x)[:, None]
    nx1, nx2 = jnp.real(nx)[:, None], jnp.imag(nx)[:, None]
    s1x, s2y = I3x1, I4x2
    
    T1 = (s1xx * nx1 + s1xy * nx2) * x1 + (s2xx * nx1 + s2xy * nx2) * x2 \
         - (stxx * nx1 + stxy * nx2) - (s1x + s2y) * nx1
    T2 = (s1xy * nx1 + s1yy * nx2) * x1 + (s2xy * nx1 + s2yy * nx2) * x2 \
         - (stxy * nx1 + styy * nx2) - (s1x + s2y) * nx2
    
    T = -jnp.vstack([T1, T2])
    
    return u, p, T

def stoDLP_closeglobal(x, sx, sxp, swxp, sigma_real, mu, side='e'):
    """
    JIT-compatible Stokes Double Layer Potential (DLP) close evaluation.
    
    Args:
        x: Target points (M,) complex
        sx: Source nodes (N,) complex
        snx: Source unit normals (N,) complex
        mu: Viscosity (float)
        sigma_real: Density (2N, Nc) real, stacked 
        side: 'i' or 'e' (passed as string or encoded integer)
    """
    N = sx.shape[0]

    # Compose complex density sigma from sigma_real = [real(sigma);imag(sigma)]
    sigma_real = jnp.atleast_2d(sigma_real)
    if sigma_real.shape[0] != N:
        sigma_real = sigma_real.T
    sigma = sigma_real[:N, :] + 1j * sigma_real[N:, :]

    # Compute geometric properties
    ssp = jnp.abs(sxp)
    snx = -1j * (sxp / ssp)

    # fix interpolation grid to 2x dense
    beta = 2
    Nf = beta * N
    
    # Interpolate geometry to fine grid
    sf_x = perispecinterp(sx, Nf)
    sf_xp = perispecinterp(sxp, Nf)
    sf_nx = perispecinterp(snx, Nf)
    sf_nx = sf_nx / jnp.abs(sf_nx)
    sf_swxp = 2*jnp.pi / Nf * sf_xp 

    sigf = perispecinterp(sigma, Nf)
    
    # Potential I1 calculation
    # tauf = sigf * (real(n)/n) -> undoes the rotation inherent in some Laplace kernels
    tauf_x1 = sigf * (jnp.real(sf_nx) / sf_nx)[:, jnp.newaxis]
    [I1x1,_,_] = lapDLP_closeglobal(x, sf_x, sf_swxp, tauf_x1, side)
    
    tauf_x2 = sigf * (jnp.imag(sf_nx) / sf_nx)[:, jnp.newaxis]
    [I1x2,_,_] = lapDLP_closeglobal(x, sf_x, sf_swxp, tauf_x2, side)
    
    I1 = I1x1 + 1j * I1x2

    # --- 2. Potential I2 ---
    # tau = real(s.x * conj(sigma))
    tau2 = jnp.real(sx[:, jnp.newaxis] * jnp.conj(sigma))
    [_, I2x1, I2x2] = lapDLP_closeglobal(x, sx, swxp, tau2, side)
    I2 = I2x1 + 1j * I2x2

    # --- 3. Potentials I3 and I4 ---
    [_, I3x1, I3x2] = lapDLP_closeglobal(x, sx, swxp, jnp.real(sigma), side)
    I3 = jnp.real(x[:, jnp.newaxis]) * (I3x1 + 1j * I3x2)
    
    [_, I4x1, I4x2] = lapDLP_closeglobal(x, sx, swxp, jnp.imag(sigma), side)
    I4 = jnp.imag(x[:, jnp.newaxis]) * (I4x1 + 1j * I4x2)

    # Combine for Velocity
    u_complex = I1 + I2 - I3 - I4
    
    # Back to real notation: stack [real; imag]
    u = jnp.concatenate([jnp.real(u_complex), jnp.imag(u_complex)], axis=0)
    
    # Pressure calculation
    p = -2 * mu * (I3x1 + I4x2)
    
    return u, p


def stoDLP_closepanel(tx, tnx, sx, sxp, sws, scur, sxlo, sxhi, sigma_real, mu, side='i'):
    """
    JIT-compatible version of StoDLP_closepanel.
    
    Args:
        tx: Target points (M,) complex
        sx: All source nodes (N,) complex
        snx: All source normals (N,) complex
        sws: Speed weights |s'(t)|*w (N,)
        sxlo, sxhi: Panel endpoints (num_panels,) complex
        mu: Viscosity
        sigma: Density (2N, Nc) real
        side: 'i' or 'e'
    """
    N = sx.shape[0]
    M = tx.shape[0]
    Nc = sigma_real.shape[1]
    num_panels = sxlo.shape[0]
    p = N // num_panels # Nodes per panel

    # Compute geometric properties
    ssp = jnp.abs(sxp)
    snx = -1j * (sxp / ssp)
    sw = sws / ssp
    swxp = sw * sxp
    
    # Reshape sources and density into [num_panels, p]
    sx_p = sx.reshape((num_panels, p))
    snx_p = snx.reshape((num_panels, p))
    sws_p = sws.reshape((num_panels, p))
    swxp_p = swxp.reshape((num_panels, p))
    sxp_p = sxp.reshape((num_panels, p))
    scur_p = scur.reshape((num_panels, p))
    
    sigma = sigma_real[:N, :] + 1j * sigma_real[N:, :]
    sigma_p = sigma.reshape((num_panels, p, -1))

    # Initial state for the loop (The "Carry")
    # Holds (accumulated_velocity, accumulated_pressure)
    init_carry = (
        jnp.zeros((M, Nc), dtype=jnp.complex128), 
        jnp.zeros((M, Nc), dtype=jnp.complex128)
    )

    def panel_contribution(carry, i):
        u_acc, p_acc = carry

        # Extract panel data
        skx = sx_p[i]
        sknx = snx_p[i]
        skws = sws_p[i]
        skwxp = swxp_p[i]
        skxp = sxp_p[i]
        skcur = scur_p[i]
        sigk = sigma_p[i]
        slo, shi = sxlo[i], sxhi[i]
        
        # 1. Distance check (Near/Far logic)
        panlen = jnp.sum(skws)
        dist_to_ends = jnp.abs(tx - slo) + jnp.abs(tx - shi)
        is_near = dist_to_ends < 2.0 * panlen
        
        # 2. Close Evaluation (The "Dspecialquad" logic)
        # These functions (Ak, L1, L2) must also be JIT-compatible
        [Ak, L1, L2, _, _] = cau_closepanel(tx, skx, skwxp, slo, shi, side)
        
        # I1: Undo n_y rotation and apply special quadrature
        tauf_x1 = sigk * (jnp.real(sknx) / sknx)[:, jnp.newaxis]
        tauf_x2 = sigk * (jnp.imag(sknx) / sknx)[:, jnp.newaxis]
        I1x1 = jnp.real(Ak @ tauf_x1)
        I1x2 = jnp.real(Ak @ tauf_x2)
        I1 = I1x1 + 1j * I1x2
        
        # I2: x * conj(sigma)
        tau2 = jnp.real(skx[:, jnp.newaxis] * jnp.conj(sigk))
        I2 = (L1 @ tau2) + 1j * (L2 @ tau2)
        
        # I3 & I4: Separating real/imag parts of density
        I3_base = (L1 @ jnp.real(sigk)) + 1j * (L2 @ jnp.real(sigk))
        I4_base = (L1 @ jnp.imag(sigk)) + 1j * (L2 @ jnp.imag(sigk))
        I3 = jnp.real(tx[:, jnp.newaxis]) * I3_base
        I4 = jnp.imag(tx[:, jnp.newaxis]) * I4_base
        
        u_close = I1 + I2 - I3 - I4
        p_close = -2 * mu * (L1 @ jnp.real(sigk) + L2 @ jnp.imag(sigk))

        # 3. Far Evaluation (Naive StoDLP)
        sigk_real = jnp.vstack([jnp.real(sigk),jnp.imag(sigk)])
        [u_far, p_far, _] = StoDLP(tx, tnx, skx, sknx, skxp, skcur, skws, mu, sigk_real)
        u_far_complex = u_far[:M,:] + 1j * u_far[M:,:]

        # 4. Blend based on proximity
        # where(condition, if_true, if_false)
        # # DEBUGGING near eval
        # is_near = jnp.ones_like(is_near)
        u_panel = jnp.where(is_near[:, jnp.newaxis], u_close, u_far_complex)
        p_panel = jnp.where(is_near[:, jnp.newaxis], p_close, p_far)

        u_new = u_acc + u_panel
        p_new = p_acc + p_panel
        
        return (u_new, p_new), None

    # Run the loop across all panels
    (final_u_complex, final_p), _ = jax.lax.scan(panel_contribution, init_carry, jnp.arange(num_panels))

    # Convert back to real stacked notation [u_real; u_imag]
    A = jnp.concatenate([jnp.real(final_u_complex), jnp.imag(final_u_complex)], axis=0)
    
    return A, final_p

stoSLP_closeglobal = jit(stoSLP_closeglobal, static_argnums=(9,))
stoDLP_closeglobal = jit(stoDLP_closeglobal, static_argnums=(6,))
stoDLP_closepanel = jit(stoDLP_closepanel, static_argnums=(10,))

# ----------------------------------------------------------------------
# --- Example / self-test ----------------------------------------------
# ----------------------------------------------------------------------
def unit_test_far():
    N = 40
    Z_top = lambda t : t + 1j*(1 + 0.3*jnp.sin(t))
    Zp_top = lambda t : 1 + 1j*(0.3*jnp.cos(t))
    Z_bot = lambda t : t + 1j*(-1 + 0.3*jnp.sin(t))
    [sx,sxp,snx,scur,sw] = channel_wall_func(Z_top,N,True, Zp_top)
    [sx2,sxp2,snx2,scur2,sw2] = channel_wall_func(Z_bot,N,False, Zp_top)
    # Combine top and bottom walls into one Wall object.
    sx = jnp.concatenate([sx,sx2])
    sxp = jnp.concatenate([sxp,sxp2])
    snx = jnp.concatenate([snx,snx2])
    scur = jnp.concatenate([scur,scur2])
    sw = jnp.concatenate([sw,sw2])

    mu = 0.7
    tx = jnp.array([2+0.2j])
    tnx = jnp.array([1+0j])

    matlabMats = {}

    print("Unit Testing Stokes kernel functions using MATLAB Matrices. \n Note that MATLAB returns NAN for the following self eval matrices: SL p, DL p, DL T.")

    print("\n Testing Sotkes SLP far ...")
    u, p, T = StoSLP(tx, tnx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
    # load u,p,T matrices to compare
    matlabMats = loadmat("../ExpectedMatrices/SL.mat")
    usl_mat = jnp.array(matlabMats['u'])
    psl_mat = jnp.array(matlabMats['p'])
    Tsl_mat = jnp.array(matlabMats['T'])
    # Compare with JAX
    print("SL Diff from MATLAB: u: {:.3g}, p: {:.3g}, T: {:.3g}.".format(jnp.linalg.norm(usl_mat-u), jnp.linalg.norm(psl_mat-p), jnp.linalg.norm(Tsl_mat-T)))

    print("\n Testing Stokes SLP self...")
    u, p, T = StoSLP_self(sx, snx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
    # load u,p,T matrices to compare
    matlabMats = loadmat("../ExpectedMatrices/SLself.mat")
    usl_mat = jnp.array(matlabMats['u'])
    psl_mat = jnp.array(matlabMats['p'])
    Tsl_mat = jnp.array(matlabMats['T'])
    # Compare with JAX
    print("SL Diff from MATLAB: u: {:.3g}, p: {:.3g}, T: {:.3g}.".format(jnp.linalg.norm(usl_mat-u), jnp.linalg.norm(psl_mat-p), jnp.linalg.norm(Tsl_mat-T)))

    print("\n Testing Sotkes DLP far ...")
    u, p, T = StoDLP(tx, tnx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
    # load u,p,T matrices to compare
    matlabMats = loadmat("../ExpectedMatrices/DL.mat")
    udl_mat = jnp.array(matlabMats['u'])
    pdl_mat = jnp.array(matlabMats['p'])
    Tdl_mat = jnp.array(matlabMats['T'])
    # Compare with JAX
    print("DL Diff from MATLAB: u: {:.3g}, p: {:.3g}, T: {:.3g}.".format(jnp.linalg.norm(udl_mat-u), jnp.linalg.norm(pdl_mat-p), jnp.linalg.norm(Tdl_mat-T)))

    print("\n Testing Stokes DLP self...")
    u, p, T = StoDLP_self(sx, snx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
    # load u,p,T matrices to compare
    matlabMats = loadmat("../ExpectedMatrices/DLself.mat")
    udl_mat = jnp.array(matlabMats['u'])
    pdl_mat = jnp.array(matlabMats['p'])
    Tdl_mat = jnp.array(matlabMats['T'])
    # Compare with JAX
    print("DL Diff from MATLAB: u: {:.3g}, p: {:.3g}, T: {:.3g}.".format(jnp.linalg.norm(udl_mat-u), jnp.linalg.norm(pdl_mat-p), jnp.linalg.norm(Tdl_mat-T)))

def unit_test_close():
    Np_wall = 10 # number of panels
    p_wall = 10 # GL grid order on each panel
    peri_len = 2*jnp.pi
    Z_top = lambda t : peri_len / (2*jnp.pi) * (2*jnp.pi - t) + 1j*(1 + 0.3*jnp.sin(2*jnp.pi - t)) # NEW: wrong in rescaling t using peri_len, should rescale x instead. t always [0,2pi]
    Zp_top = lambda t : -peri_len / (2*jnp.pi) - 1j*(0.3*jnp.cos(2*jnp.pi-t))
    Zpp_top = lambda t : -1j*0.3*jnp.sin(2*jnp.pi-t)
    Z_bot = lambda t : peri_len / (2*jnp.pi) * t + 1j*(-1 + 0.3*jnp.sin(t))
    Zp_bot = lambda t : peri_len / (2*jnp.pi) + 1j*(0.3*jnp.cos(t))
    Zpp_bot = lambda t : -1j*0.3*jnp.sin(t)
    [sx,sxp,snx,scur,sws,swxp,sxlo,sxhi] = channel_wall_glpanels(Z_top,Np_wall,p_wall,Zp_top,Zpp_top)
    [sx2,sxp2,snx2,scur2,sws2,swxp2,sxlo2,sxhi2] = channel_wall_glpanels(Z_bot,Np_wall,p_wall,Zp_bot,Zpp_bot)
    # Combine top and bottom walls into one Wall object.
    sx = jnp.concatenate([sx,sx2])
    sxp = jnp.concatenate([sxp,sxp2])
    snx = jnp.concatenate([snx,snx2])
    scur = jnp.concatenate([scur,scur2])
    sws = jnp.concatenate([sws,sws2])
    swxp = jnp.concatenate([swxp,swxp2])
    sxlo = jnp.concatenate([sxlo,sxlo2])
    sxhi = jnp.concatenate([sxhi,sxhi2])

    N_ptcl = 8 # total number of discr. points on EACH particle (global quadr)
    Z_ptcl = lambda t : 1 + 0.3*jnp.cos(t) + 1j*(0.3*jnp.sin(t)+0.25)
    Zp_ptcl = lambda t : - 0.3*jnp.sin(t) + 1j*0.3*jnp.cos(t)
    Zpp_ptcl = lambda t : - 0.3*jnp.cos(t) - 1j*0.3*jnp.sin(t)
    [ptx,ptxp,ptnx,ptcur,ptw] = channel_wall_func(Z_ptcl,N_ptcl,False, Zp_ptcl, Zpp_ptcl)
    ptwxp = ptw / jnp.abs(ptxp) * ptxp
    pta = 1+0.25j # a point interior to ptcl, use its center
    ptt = jnp.linspace(0, 2 * jnp.pi, N_ptcl, endpoint=False) # use fact that global quadr always have this t.

    matlabMats = {}

    print("Unit testing near-evaluation using MATLAB outputs")

    tx = jnp.array([2+1.2j,1+0.8j])
    tnx = jnp.array([1+1j,1+1j])
    
    print("\n TEST 1: Cauchy integral computations (cau_closeglobal and cau_closepanel (DSpecialQuad in MATLAB).")
    test_panel = 1
    sx_test = sx[test_panel*p_wall : (test_panel+1)*p_wall]
    sxlo_test = sxlo[test_panel]
    sxhi_test = sxhi[test_panel]
    swxp_test = swxp[test_panel*p_wall : (test_panel+1)*p_wall]
    [v,dv1,dv2,ddv1,ddv2] = cau_closepanel(tx, sx_test, swxp_test, sxlo_test, sxhi_test, 'i')
    # Load matlab matrix
    matlabMats = loadmat("../ExpectedMatrices/cau_panel.mat")
    vwall_mat = jnp.array(matlabMats['v'])
    dv1wall_mat = jnp.array(matlabMats['dv1'])
    dv2wall_mat = jnp.array(matlabMats['dv2'])
    ddv1wall_mat = jnp.array(matlabMats['ddv1'])
    ddv2wall_mat = jnp.array(matlabMats['ddv2'])
    # Compare with JAX
    print("Cau Diff from MATLAB: v: {:.3g}, dv1: {:.3g}, dv2: {:.3g}, ddv1: {:.3g}, ddv2: {:.3g}".format(jnp.linalg.norm(vwall_mat-v), jnp.linalg.norm(dv1wall_mat-dv1), jnp.linalg.norm(dv2wall_mat-dv2), jnp.linalg.norm(ddv1wall_mat-ddv1), jnp.linalg.norm(ddv2wall_mat-ddv2)))

    tx = jnp.array([2+0.2j,1+0.8j])
    tnx = jnp.array([1+1j,1+1j])
    # cau_closeglobal is less flexible than MATLAB version, so need always to input some form of density. 
    [v1,vp1,vpp1] = cau_closeglobal(tx,ptx,ptwxp,jnp.eye(ptx.shape[0]),'e')
    # load matlab matrix
    matlabMats = loadmat("../ExpectedMatrices/cau_global.mat")
    vptcl_mat = jnp.array(matlabMats['v'])
    vpptcl_mat = jnp.array(matlabMats['vp'])
    vppptcl_mat = jnp.array(matlabMats['vpp'])
    # Compare with JAX
    print("Cau Diff from MATLAB: v: {:.3g}, vp: {:.3g}, vpp: {:.3g}".format(jnp.linalg.norm(vptcl_mat-v1), jnp.linalg.norm(vpptcl_mat-vp1), jnp.linalg.norm(vppptcl_mat-vpp1)))

    print("\n TEST 2: Laplace SL and DL close evals, global only. ")
    [v,dv1,dv2,ddv1,ddv12,ddv2] = lapSLP_closeglobal(tx,ptx,ptxp,ptt,pta,ptw,jnp.eye(ptx.shape[0]),'e')
    # load matlab matrix
    matlabMats = loadmat("../ExpectedMatrices/lapsl_global.mat")
    vptcl_mat = jnp.array(matlabMats['v'])
    dv1ptcl_mat = jnp.array(matlabMats['dv1'])
    dv2ptcl_mat = jnp.array(matlabMats['dv2'])
    ddv1ptcl_mat = jnp.array(matlabMats['ddv1'])
    ddv12ptcl_mat = jnp.array(matlabMats['ddv12'])
    ddv2ptcl_mat = jnp.array(matlabMats['ddv2'])
    # Compare with JAX
    print("Lap SL Diff from MATLAB: v: {:.3g}, dv1: {:.3g}, dv2: {:.3g}, ddv1: {:.3g}, ddv12: {:.3g}, ddv2: {:.3g}".format(jnp.linalg.norm(vptcl_mat-v), jnp.linalg.norm(dv1ptcl_mat-dv1), jnp.linalg.norm(dv2ptcl_mat-dv2), jnp.linalg.norm(ddv1ptcl_mat-ddv1), jnp.linalg.norm(ddv12ptcl_mat-ddv12), jnp.linalg.norm(ddv2ptcl_mat-ddv2)))

    [v,dv1,dv2] = lapDLP_closeglobal(tx,ptx,ptwxp,jnp.eye(ptx.shape[0]),'e')
    # load matlab matrix
    matlabMats = loadmat("../ExpectedMatrices/lapdl_global.mat")
    vptcl_mat = jnp.array(matlabMats['v'])
    dv1ptcl_mat = jnp.array(matlabMats['dv1'])
    dv2ptcl_mat = jnp.array(matlabMats['dv2'])
    # Compare with JAX
    print("Lap DL Diff from MATLAB: v: {:.3g}, dv1: {:.3g}, dv2: {:.3g}".format(jnp.linalg.norm(vptcl_mat-v), jnp.linalg.norm(dv1ptcl_mat-dv1), jnp.linalg.norm(dv2ptcl_mat-dv2)))

    print("\n TEST 3: Stokes SL and DL close eval, global only. ")
    mu = 0.7
    
    sigma_real = jnp.eye(2*ptx.shape[0])
    [u,p,T] = stoSLP_closeglobal(tx,tnx,ptx,ptxp,ptt,pta,ptw,sigma_real,mu,'e')
    # load matlab matrix
    matlabMats = loadmat("../ExpectedMatrices/stosl_global.mat")
    uptcl_mat = jnp.array(matlabMats['u'])
    pptcl_mat = jnp.array(matlabMats['p'])
    Tptcl_mat = jnp.array(matlabMats['T'])
    # Compare with JAX
    print("Stokes SL Diff from MATLAB: u: {:.3g}, p: {:.3g}, T: {:.3g}".format(jnp.linalg.norm(uptcl_mat-u), jnp.linalg.norm(pptcl_mat-p), jnp.linalg.norm(Tptcl_mat-T)))

    [u,p] = stoDLP_closeglobal(tx,ptx,ptxp,ptwxp,sigma_real,mu,'e')
    # load matlab matrix
    matlabMats = loadmat("../ExpectedMatrices/stodl_global.mat")
    uptcl_mat = jnp.array(matlabMats['u'])
    pptcl_mat = jnp.array(matlabMats['p'])
    # Compare with JAX
    print("Stokes DL Diff from MATLAB: u: {:.3g}, p: {:.3g}".format(jnp.linalg.norm(uptcl_mat-u), jnp.linalg.norm(pptcl_mat-p)))

    print("\n TEST 4: Stokes DLP panel based close eval. ")
    tx = jnp.array([2+1.2j,1+0.8j])
    sigma_real = jnp.eye(2*sx.shape[0])
    [u,p] = stoDLP_closepanel(tx,tnx,sx,sxp,sws,scur,sxlo,sxhi,sigma_real,mu,'i')
    # load matlab matrix 
    matlabMats = loadmat("../ExpectedMatrices/stodl_panel.mat")
    u_mat = jnp.array(matlabMats['A'])
    p_mat = jnp.array(matlabMats['P'])
    # Compare with JAX
    print("Stokes DL Diff from MATLAB: u: {:.3g}, p: {:.3g}".format(jnp.linalg.norm(u_mat-u), jnp.linalg.norm(p_mat-p)))



if __name__ == "__main__":
    unit_test_far()
    unit_test_close()
    
