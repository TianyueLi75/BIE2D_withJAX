"""
periodic_utils.py
================
JAX-based implementation of 2D singly-periodic structures support

Based on MATLAB codes by A. Barnett (2016):
    perivelpipe.m
Translated and refactored into object-oriented JAX Python.

This file is generated with the help of ChatGPT

Author: Choco Li
Date: 2025-11-10
"""

from jax.numpy.fft import fft, ifft
# from jax import jit
from jax import lax
import jax.numpy as jnp
import matplotlib.pyplot as plt


# -------------------------
# Periodic spectral diff (1st derivative)
# -------------------------
def perispecdiff(f: jnp.ndarray) -> jnp.ndarray:
    """
    Spectrally differentiate periodic samples f(t_j) on [0,2pi).
    """
    N = f.size
    k = jnp.fft.fftfreq(N, 2*jnp.pi / N)
    # k = k.at[N // 2 + 1 :].set(k[N // 2 + 1 :] - N)  # enforce symmetry
    k = 2*jnp.pi * k
    fhat = jnp.fft.fft(f)
    df = jnp.fft.ifft(1j* k * fhat)
    return df

def gauss(N: int):
    """
    Compute Gauss–Legendre nodes on [-1, 1].
    Fully JAX-compatible (jit/grad safe).
    """
    N = N - 1
    N1 = N + 1
    N2 = N + 2

    xu = jnp.linspace(-1.0, 1.0, N1)
    # Initial guess for roots
    y = jnp.cos((2 * jnp.arange(N + 1) + 1) * jnp.pi / (2 * N + 2)) \
        + (0.27 / N1) * jnp.sin(jnp.pi * xu * N / N2)
    eps = jnp.finfo(float).eps

    def legendre_eval(y):
        """Evaluate P_{N+1}(y) and its derivative."""
        L0 = jnp.ones_like(y)
        L1 = y

        def body(carry, k):
            Lkm2, Lkm1 = carry
            Lk = ((2 * k - 1) * y * Lkm1 - (k - 1) * Lkm2) / k
            return (Lkm1, Lk), Lk

        (_, _), L_rest = lax.scan(body, (L0, L1), jnp.arange(2, N1 + 1))
        L_all = jnp.column_stack([L0, L1, L_rest.T])
        Lp = N2 * (L_all[:, N1 - 1] - y * L_all[:, N2 - 1]) / (1 - y**2)
        return L_all[:, N2 - 1], Lp

    def cond_fun(carry):
        y, y_prev = carry
        return jnp.max(jnp.abs(y - y_prev)) > eps

    def body_fun(carry):
        y, _ = carry
        L, Lp = legendre_eval(y)
        y_new = y - L / Lp
        return (y_new, y)

    init = (y, y + 2.0)
    y_final, _ = lax.while_loop(cond_fun, body_fun, init)
    x = y_final[::-1]  # reverse order for ascending nodes

    return x

# -------------------------
# Initialize a channel wall (combining init and setup steps)
# when function and number of discretization points are given
# -------------------------
def channel_wall_func(Z,N,normal_orient,*args):
    t = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    x = Z(t)
    if args is not None:
        Zp = args[0]
        xp = Zp(t)
    else:
        xp = perispecdiff(x)
    if len(args) > 1:
        Zpp = args[1]
        xpp = Zpp(t)
    else:
        xpp = perispecdiff(xp)
    sp = jnp.abs(xp)
    nx = -1j*(xp / sp)
    if normal_orient:
        nx = -nx
        xp = -xp
    cur = -jnp.real(jnp.conj(xpp) * nx) / (sp**2)
    sw = (2 * jnp.pi / N) * sp
    return x,xp,nx,cur,sw
    
# -------------------------
# Initialize a channel wall (combining init and setup steps)
# when discretization nodes are given
# -------------------------
def channel_wall_x(x_, normal_orient):
    x = x_.flatten()
    xp = perispecdiff(x)
    xpp = perispecdiff(xp)
    sp = jnp.abs(xp)
    nx = -1j*(xp / sp)
    if normal_orient:
        nx = -1*nx
    cur = -jnp.real(jnp.conj(xpp) * nx) / (sp**2)
    sw = (2 * jnp.pi / x.size) * sp
    return x,xp,nx,cur,sw

def side_wall(xloc, height, gl_order):
    gl_nodes = gauss(gl_order)
    t = (1+gl_nodes) * jnp.pi
    Zx = lambda t : xloc + 1j * height / 2. / jnp.pi * t
    x = Zx(t)
    nx = jnp.ones_like(x) + 1j*jnp.zeros_like(x)
    return x,nx
        
def proxy(R, perilen, N):
    Zx = lambda t : 0.5*perilen + R*jnp.cos(t) + 1j*R*jnp.sin(t)
    Zxp = lambda t: -R*jnp.sin(t) + 1j*R*jnp.cos(t)
    t = jnp.linspace(0, 2 * jnp.pi, N, endpoint=False)
    x = Zx(t)
    xp = Zxp(t)
    sp = jnp.abs(xp)
    nx = -1j * (xp / sp)
    wt = (2 * jnp.pi / x.size) * sp
    # wt = jnp.ones_like(x)
    return x, xp, nx, wt

def vis(x_, nx_, hold: bool  = False):
    # Extract components for plotting
    x, y = jnp.real(x_), jnp.imag(x_)
    u, v = jnp.real(nx_), jnp.imag(nx_)

    # Create the quiver plot
    if not hold: 
        plt.figure(figsize=(7, 6))
    plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, color='b')
    plt.scatter(x, y, color='r', zorder=3)

    # # Add labels for each point
    # for xi, yi, w, cur, t in zip(x, y, self.w, self.cur, self.t):
    #     label = f"w={w:.2f}, cur={cur:.2f}, t={t:.1f}"
    #     plt.text(xi, yi, label, fontsize=8, ha='left', va='bottom')

    if not hold:
        plt.axis('equal')
        plt.xlabel('Real part')
        plt.ylabel('Imag part')
        plt.title('2D Vector Field Visualization')
        plt.grid(True)
        # plt.tight_layout()
        plt.show()

# ------------------------------
# Example usage / self-test for setup quad.
# ------------------------------
if __name__ == "__main__":
    Z_top = lambda t : t + 1j*(3 + 0.3*jnp.sin(t))
    Zp_top = lambda t : 1 + 1j*(0.3*jnp.cos(t))
    Z_bot = lambda t : t + 1j*(0.3*jnp.sin(t))
    
    # Z = lambda s: (1 + 0.3 * jnp.cos(5 * s)) * jnp.exp(1j * s)
    # Zp = lambda s: (-1.5 * jnp.sin(5 * s) + 1j * (1 + 0.3 * jnp.cos(5 * s))) * jnp.exp(1j * s)

    plt.figure(figsize=(7, 6))

    N_wall = 10
    N_side = 4
    N_prx = 5

    print("Channel, top")
    [x,xp,nx,cur] = channel_wall_func(Z_top,N_wall,True, Zp_top)
    vis(x,nx,True)

    print("Channel, bottom")
    [x,xp,nx,cur] = channel_wall_func(Z_bot,N_wall,False, Zp_top)
    vis(x,nx,True)

    print("Side, left")
    [x,nx] = side_wall(0., 3,  N_side)
    vis(x,nx,True)

    print("Side, right")
    [x,nx] = side_wall(2*jnp.pi, 3, N_side)
    vis(x,nx,True)

    print("Particle, circle")
    Z = lambda t : 0.5 + 0.3*jnp.cos(t) + 1j*(0.3*jnp.sin(t)+1)
    Zp = lambda t : - 0.1*jnp.sin(t) + 1j*0.1*jnp.cos(t)
    [x,xp,nx,cur] = channel_wall_func(Z,N_wall,False, Zp)
    vis(x,nx,True)

    print("Proxy")
    R = 2.2*jnp.pi
    [x,xp,nx,pwt] = proxy(R, 2*jnp.pi, N_prx)
    vis(x,nx,True)

    plt.axis('equal')
    plt.xlabel('Real part')
    plt.ylabel('Imag part')
    plt.title('2D Vector Field Visualization')
    plt.grid(True)
    # plt.tight_layout()
    plt.show()