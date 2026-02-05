# A test script to solve 2D Stokes periodic channel flow based on 
# BIE2D: https://github.com/ahbarnett/BIE2D/tree/master
# Choco Li, Dec 2025

from jax import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.io import savemat
import sys, os
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the folder containing the module you want to import
utils_path = os.path.join(current_dir, '..')
peri_path = os.path.join(current_dir, '../periodic')
# Add the folder to the system path
sys.path.append(utils_path)
sys.path.append(peri_path)
from periodic.structure_jax import *
from periodic.periodic_ELS_jax import *

# TODO: avoid doing this if JITing.
# jax.clear_caches()


# _manual() functions for checking kernel functions.
def StoSLP_manual(tx, sx, sxp, mu):
    N = sx.size
    M = tx.size

    sp = jnp.abs(sxp)
    sw = (2 * jnp.pi / N) * sp

    S = jnp.zeros((M*2,N*2))

    coeff = 1.0/4.0/jnp.pi/mu

    for i in range(N):
        wt = sw[i]
        coeff_wt = coeff * wt
        for j in range(M):
            xtrg = jnp.real(tx[j])
            ytrg = jnp.imag(tx[j])
            xsrc = jnp.real(sx[i])
            ysrc = jnp.imag(sx[i])
            r = jnp.array([xtrg-xsrc,ytrg-ysrc])
            rnorm = jnp.linalg.norm(r)
            S = S.at[j,i].set(-coeff_wt * jnp.log(rnorm) + coeff_wt * r[0]*r[0]/rnorm/rnorm)
            S = S.at[j+M,i].set(S[j+M,i] + coeff_wt * r[0]*r[1]/rnorm/rnorm)
            S = S.at[j,i+N].set(S[j,i+N] + coeff_wt * r[0]*r[1]/rnorm/rnorm)
            S = S.at[j+M,i+N].set(-coeff_wt * jnp.log(rnorm) + coeff_wt * r[1]*r[1]/rnorm/rnorm)
            
    return S

def StoDLP_manual(tx, sx, snx, sxp, mu):
    N = sx.size
    M = tx.size

    sp = jnp.abs(sxp)
    sw = (2 * jnp.pi / N) * sp

    S = jnp.zeros((M*2,N*2))


    for i in range(N):
        wt = sw[i]
        for j in range(M):
            xtrg = jnp.real(tx[j])
            ytrg = jnp.imag(tx[j])
            xsrc = jnp.real(sx[i])
            ysrc = jnp.imag(sx[i])
            xnsrc = jnp.real(snx[i])
            ynsrc = jnp.imag(snx[i])
            r = jnp.array([xtrg-xsrc,ytrg-ysrc])
            rnorm = jnp.linalg.norm(r)
            rnorm4 = rnorm ** 4
            rdotn = xnsrc*r[0] + ynsrc*r[1]
            coeff_wt = rdotn * wt / jnp.pi / mu / rnorm4
            S = S.at[j,i].set(coeff_wt * r[0]*r[0])
            S = S.at[j+M,i].set(coeff_wt * r[0] * r[1])
            S = S.at[j,i+N].set(coeff_wt * r[1] * r[0])
            S = S.at[j+M,i+N].set(coeff_wt * r[1] * r[1])
            
    return S

def StoDLP3_manual(peri_len, lx, sx, snx, sxp, mu):
    # Sum of p=-1,0,1 copies of sx on lx.
    A1 = StoDLP_manual(lx, sx, snx, sxp,mu)
    sx1 = sx - peri_len
    A2 = StoDLP_manual(lx, sx1, snx, sxp,mu)
    sx1 = sx + peri_len
    A3 = StoDLP_manual(lx, sx1, snx, sxp,mu)
    return A1+A2+A3

# Set discretization parameters
# N_wall = 4
# N_side = 4

N_wall = 80
N_side = 40

N_prx = 2*N_side
N_ptcl = N_wall
peri_len = 2*jnp.pi

# Set up channel walls
# NOTE: currently I hard coded the side walls to go from y=0 to y=height,
#       so top and bottom walls need to satisfy this at x=0 and x=peri_len.
Z_top = lambda t : t + 1j*(1 + 0.3*jnp.sin(2*jnp.pi/peri_len*t))
Zp_top = lambda t : 1 + 2*jnp.pi/peri_len*1j*(0.3*jnp.cos(2*jnp.pi/peri_len*t))
Z_bot = lambda t : t + 1j*(-1 + 0.3*jnp.sin(2*jnp.pi/peri_len*t))
[sx,sxp,snx,scur,sw] = channel_wall_func(Z_top,N_wall,True, Zp_top)
[sx2,sxp2,snx2,scur2,sw2] = channel_wall_func(Z_bot,N_wall,False, Zp_top)
# Combine top and bottom walls into one Wall object.
sx = jnp.concatenate([sx,sx2])
sxp = jnp.concatenate([sxp,sxp2])
snx = jnp.concatenate([snx,snx2])
scur = jnp.concatenate([scur,scur2])
sw = jnp.concatenate([sw,sw2])
vis(sx, snx, True)

# Add particle 
num_ptcl = 0 # number of particles on the interior, for self eval.
if num_ptcl:
    Z_ptcl = lambda t : 1 + 0.3*jnp.cos(t) + 1j*(0.3*jnp.sin(t)+0.25)
    Zp_ptcl = lambda t : - 0.3*jnp.sin(t) + 1j*0.3*jnp.cos(t)
    Zpp_ptcl = lambda t : - 0.3*jnp.cos(t) - 1j*0.3*jnp.sin(t)
    [ptx,ptxp,ptnx,ptcur,ptw] = channel_wall_func(Z_ptcl,N_ptcl,True, Zp_ptcl, Zpp_ptcl)
    # Add another particle
    Z_ptcl = lambda t : 5 + 0.2*jnp.cos(t) + 1j*(0.2*jnp.sin(t)+0.)
    Zp_ptcl = lambda t : - 0.2*jnp.sin(t) + 1j*0.2*jnp.cos(t)
    Zpp_ptcl = lambda t : - 0.2*jnp.cos(t) - 1j*0.2*jnp.sin(t)
    [ptx2,ptxp2,ptnx2,ptcur2,ptw2] = channel_wall_func(Z_ptcl,N_ptcl,True, Zp_ptcl, Zpp_ptcl)
    # Combine particle info, ASSUMES same discr on each ptcl!
    # TODO later: can allow for differences if use vstack and pads. Will need to change ptcl all to all in that case.
    ptx = jnp.concatenate([ptx,ptx2])
    ptxp = jnp.concatenate([ptxp,ptxp2])
    ptnx = jnp.concatenate([ptnx,ptnx2])
    ptcur = jnp.concatenate([ptcur,ptcur2])
    ptw = jnp.concatenate([ptw,ptw2])
    vis(ptx, ptnx, True)
else:
    ptx = jnp.array([])
    ptxp = jnp.array([])
    ptnx = jnp.array([])
    ptcur = jnp.array([])
    ptw = jnp.array([])

# Add Stokeslets
num_stokeslet = 2
if num_stokeslet:
    x_stokeslet = jnp.array([1+0.2j,4+0.25j])
    xp_stokeslet = jnp.ones((num_stokeslet,)) # Not needed in SLPmat so anything is fine.
    w_stokeslet = jnp.ones((num_stokeslet,))
    # Just for visualziation
    nx_stokeslet = jnp.array([1+0j, 1+0j])
    vis(x_stokeslet, nx_stokeslet, True)
else:
    x_stokeslet = jnp.array([])
    xp_stokeslet = jnp.array([])
    w_stokeslet = jnp.array([])

# Set up side walls 
[lx,lnx] = side_wall(0., 2, N_side)
[rx,rnx] = side_wall(0.+peri_len, 2, N_side)
lx = lx - 1j
rx = rx - 1j
vis(lx, lnx, True)
vis(rx, rnx, True)

# Set up proxy sources
R = 1.1 * peri_len
[px,pxp,pnx,pwt] = proxy(R, peri_len, N_prx)
vis(px, pnx, True)

# # To verify points and normals look right -- all normals on structures should point out of fluid domain.
# plt.axis('equal')
# plt.xlabel('Real part')
# plt.ylabel('Imag part')
# plt.title('2D Vector Field Visualization')
# plt.grid(True)
# # plt.tight_layout()
# plt.show()

mu = 0.7

# Make ELS matrix
# TODO later: if we want a combination of these cases, we may need to make a new ELSmatrix function.
if num_ptcl:
    [E,A,B,C,Q] = ELSmatrix_ptcl(sx, snx, sxp, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, num_ptcl, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu)
elif num_stokeslet: 
    # [E,A,B,C,Q] = ELSmatrix_stokeslet(sx, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu)
    [E,A,B,C,Q] = ELSmatrix_stokeslet_f0(sx, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, lx, lnx, rx, rnx, peri_len, mu)
else:
    [E,A,B,C,Q] = ELSmatrix(sx, snx, sxp, scur, sw, px, pxp, lx, lnx, rx, rnx, peri_len, mu)

tx = jnp.array([2+0.2j,4+0.1j])
tnx = jnp.array([1+0j,1+0j])
# tx = jnp.array([2+0.2j])
# tnx = jnp.array([1+0j])

# # a) Exact solution
# # TODO: this may not work with stokeslets inside, since "exact" flow doesn't have the same singularities at point sources.
# print('expt=t: running known Poisseuil flow BVP...\n')
# h=.2
# ue = lambda x: h*jnp.concatenate([jnp.imag(x)**2,jnp.zeros_like(jnp.imag(x))])
# pe = lambda x: h*2*mu*jnp.real(x)
# vrhs = ue(sx)
# vrhs_ptcl = ue(ptx) # will just be an empty array if num_ptcl is 0
# jump = pe(jnp.real(rx[0]))-pe(jnp.real(lx[0]))
# Tjump = -jump * jnp.concatenate([jnp.real(rnx), jnp.imag(rnx)])
# # erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((2*N_side,)), Tjump])
# erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((2*N_side,)), Tjump, jnp.array([0]), jnp.array([0])]) 

# # Solve for density
# [edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15) 
# print(f'resid norm = {resid[0]:.3g}\n')
# wall_dens = edens[:4*N_wall]
# ptcl_dens = edens[4*N_wall:4*N_wall+2*num_ptcl*N_ptcl] # will just be a zero slice if num_ptcl = 0
# f_stokeslet = edens[4*N_wall+2*num_ptcl*N_ptcl:4*N_wall+2*num_ptcl*N_ptcl+2*num_stokeslet] # will be zero slice if num_stokeslet=0
# prx_dens = edens[4*N_wall+2*num_ptcl*N_ptcl+2*num_stokeslet:]
# wall_dens_norm = jnp.linalg.norm(wall_dens)
# ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
# f_dens_norm = jnp.linalg.norm(f_stokeslet)
# prx_dens_norm = jnp.linalg.norm(prx_dens)
# print(f'norm of wall density = {wall_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of stokeslet force = {f_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}\n')

# # Evaluate at target
# if num_ptcl:
#     [ut, pt] = evalsol_ptcl(tx, tnx, sx, snx, sxp, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, px, pxp, pwt, peri_len, mu, edens)
# elif num_stokeslet:
#     [ut, pt] = evalsol_stokeslet(tx, tnx, sx, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, peri_len, mu, edens)
# else:
#     [ut, pt] = evalsol2(tx, tnx, sx, snx, sxp, scur, sw, px, pxp, peri_len, mu, edens)

# err = jnp.linalg.norm(ut - ue(tx)) # short form check -- matrix norm for velocity error, abs val for pressure
# perr = jnp.abs(pt - pe(tx))
# # print(perr.shape)
# print(f'u velocity err at zt = {err:.3g}\n')
# print(f'p err at first trg: {perr[0]:.3g}, at second trg: {perr[1]:.3g}')

# b) No slip flow
print('expt=d: solving no-slip pressure-driven flow in pipe...\n')
vrhs = jnp.zeros((sx.size*2,))
vrhs_ptcl = jnp.zeros((ptx.size*2,))
jump = -1
Tjump = -jump * jnp.concatenate([jnp.real(rnx), jnp.imag(rnx)])
# erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((2*N_side,)), Tjump])
erhs = jnp.concatenate([vrhs, vrhs_ptcl, jnp.zeros((2*N_side,)), Tjump, jnp.array([0]), jnp.array([0])]) # Add x- and y- force 0 constraint at the end.

# Solve for density
[edens, resid, _, _] = jnp.linalg.lstsq(E,erhs,rcond=1e-15)
print(f'resid norm = {resid[0]:.3g}\n')
wall_dens = edens[:4*N_wall]
ptcl_dens = edens[4*N_wall:4*N_wall+2*num_ptcl*N_ptcl] # will just be a zero slice if num_ptcl = 0
f_stokeslet = edens[4*N_wall+2*num_ptcl*N_ptcl:4*N_wall+2*num_ptcl*N_ptcl+2*num_stokeslet] # will be zero slice if num_stokeslet=0
prx_dens = edens[4*N_wall+2*num_ptcl*N_ptcl+2*num_stokeslet:]
wall_dens_norm = jnp.linalg.norm(wall_dens)
ptcl_dens_norm = jnp.linalg.norm(ptcl_dens)
f_dens_norm = jnp.linalg.norm(f_stokeslet)
prx_dens_norm = jnp.linalg.norm(prx_dens)
print(f'norm of wall density = {wall_dens_norm:.3g}, norm of ptcl density = {ptcl_dens_norm:.3g}, norm of stokeslet force = {f_dens_norm:.3g}, norm of proxy density = {prx_dens_norm:.3g}\n')

# Evaluate at target
if num_ptcl:
    [ut, pt] = evalsol_ptcl(tx, tnx, sx, snx, sxp, scur, sw, ptx, ptnx, ptxp, ptcur, ptw, px, pxp, pwt, peri_len, mu, edens)
elif num_stokeslet:
    [ut, pt] = evalsol_stokeslet(tx, tnx, sx, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, peri_len, mu, edens)
else:
    [ut, pt] = evalsol2(tx, tnx, sx, snx, sxp, scur, sw, px, pxp, peri_len, mu, edens)

print('u velocity at zt = {:.12g}, {:.12g}, p at first trg = {:.12g}, at second = {:.12g}'.format(ut[0], ut[1], pt[0], pt[1]))

# ==========================================================
# Cell 6: For plotting total field with holes around stokeslets
# ==========================================================
def plot_streamlines_total_with_multi_holes(edens, x0_list, f_list,
                                            title="Multiple internal stokeslets (total field)",
                                            nxg=140, ng=70, ypad=0.12,
                                            density=1.3, buffer_factor=3.0,
                                            hole_radius=0.06):
    xg = np.linspace(0.0, peri_len, nxg)
    yg = np.linspace(-1.0-ypad, 1.0+ypad, ng)
    X, Y = np.meshgrid(xg, yg)

    dx = (xg.max() - xg.min()) / (nxg - 1)
    dy = (yg.max() - yg.min()) / (ng - 1)
    delta = buffer_factor * min(dx, dy)

    Xj = jnp.array(xg)
    YT = np.array(jnp.imag(Z_top(Xj)))
    YB = np.array(jnp.imag(Z_bot(Xj)))

    inside = (Y >= (YB[None, :] + delta)) & (Y <= (YT[None, :] - delta))

    # holes around each source
    hole = np.zeros_like(X, dtype=bool)
    for z in np.array(x0_list):
        X0, Y0 = float(np.real(z)), float(np.imag(z))
        hole |= (X - X0)**2 + (Y - Y0)**2 <= hole_radius**2
    inside = inside & (~hole)

    tx_inside = (X[inside] + 1j*Y[inside]).astype(np.complex128)
    tx_jax = jnp.array(tx_inside)
    tnx_jax = jnp.ones_like(tx_jax) + 0j

    u_tot, _ = evalsol_stokeslet(tx_jax, tnx_jax, sx, snx, sxp, scur, sw, x_stokeslet, xp_stokeslet, w_stokeslet, px, pxp, pwt, peri_len, mu, edens)

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

    th = np.linspace(0, 2*np.pi, 200)
    for i, z in enumerate(np.array(x0_list)):
        X0, Y0 = float(np.real(z)), float(np.imag(z))
        plt.scatter([X0], [Y0], s=45, marker='x')
        plt.text(X0, Y0, f"  s{i}", va='center')
        plt.plot(X0 + hole_radius*np.cos(th), Y0 + hole_radius*np.sin(th), 'k--', lw=1)

    plt.axis("equal")
    plt.xlim(0, peri_len)
    plt.ylim(-1.0-ypad, 1.0+ypad)
    plt.colorbar(label="|u|")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

plot_streamlines_total_with_multi_holes(edens, x_stokeslet, f_stokeslet,
                                        title=f"Multiple stokeslets: K={num_stokeslet}, pos=uniform, force=random_dir", density=4)