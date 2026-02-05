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


from periodic.structure_jax import channel_wall_func

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
# --- Example / self-test ----------------------------------------------
# ----------------------------------------------------------------------
if __name__ == "__main__":
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
    matlabMats = loadmat("ExpectedMatrices/SL.mat")
    usl_mat = jnp.array(matlabMats['u'])
    psl_mat = jnp.array(matlabMats['p'])
    Tsl_mat = jnp.array(matlabMats['T'])
    # Compare with JAX
    print("SL Diff from MATLAB: u: {:.3g}, p: {:.3g}, T: {:.3g}.", jnp.linalg.norm(usl_mat-u), jnp.linalg.norm(psl_mat-p), jnp.linalg.norm(Tsl_mat-T))

    print("\n Testing Stokes SLP self...")
    u, p, T = StoSLP_self(sx, snx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
    # load u,p,T matrices to compare
    matlabMats = loadmat("ExpectedMatrices/SLself.mat")
    usl_mat = jnp.array(matlabMats['u'])
    psl_mat = jnp.array(matlabMats['p'])
    Tsl_mat = jnp.array(matlabMats['T'])
    # Compare with JAX
    print("SL Diff from MATLAB: u: {:.3g}, p: {:.3g}, T: {:.3g}.", jnp.linalg.norm(usl_mat-u), jnp.linalg.norm(psl_mat-p), jnp.linalg.norm(Tsl_mat-T))

    print("Testing Sotkes DLP far ...")
    u, p, T = StoDLP(tx, tnx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
    # load u,p,T matrices to compare
    matlabMats = loadmat("ExpectedMatrices/DL.mat")
    udl_mat = jnp.array(matlabMats['u'])
    pdl_mat = jnp.array(matlabMats['p'])
    Tdl_mat = jnp.array(matlabMats['T'])
    # Compare with JAX
    print("DL Diff from MATLAB: u: {:.3g}, p: {:.3g}, T: {:.3g}.", jnp.linalg.norm(udl_mat-u), jnp.linalg.norm(pdl_mat-p), jnp.linalg.norm(Tdl_mat-T))

    print("\n Testing Stokes DLP self...")
    u, p, T = StoDLP_self(sx, snx, sx, snx, sxp, scur, sw, mu, jnp.array([]))
    # load u,p,T matrices to compare
    matlabMats = loadmat("ExpectedMatrices/DLself.mat")
    udl_mat = jnp.array(matlabMats['u'])
    pdl_mat = jnp.array(matlabMats['p'])
    Tdl_mat = jnp.array(matlabMats['T'])
    # Compare with JAX
    print("DL Diff from MATLAB: u: {:.3g}, p: {:.3g}, T: {:.3g}.", jnp.linalg.norm(udl_mat-u), jnp.linalg.norm(pdl_mat-p), jnp.linalg.norm(Tdl_mat-T))

    
