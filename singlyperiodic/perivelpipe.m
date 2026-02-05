function perivelpipe(expt)
% singly-periodic in 2D velocity-BC Stokes BVP, ie "pipe" geom w/ press drop
% Rewrite of 2014 codes using BIE2D, as in DPLS codes. Dense E fill for now,
% DLP only on U,D (since wrapping S quad would be harder).  12/17/17
disp('2D 1-periodic pressure-driven pipe flow w/ vel BCs on walls')

% Issues:
% * we don't have close eval for open (periodized?) segments yet.
% * rewrite using periodized Stokes FMM, which should have a direct option too.

v=1;  % verbosity=0,1,2,3
if nargin==0, expt='t'; end  % expt='t' test known soln, 'd' driven no-slip demo

N=40;
m=40;

% set up upper and lower walls
uc.e1 = 2*pi;   % unitcell, e1=lattice vector as a complex number
% N = 80;     % pts per top and bottom wall (enough for 1e-15 in this case)
% N = 4;
U.Z = @(t) t + 1i*(1+0.3*sin(t)); U.Zp = @(t) 1 + 0.3i*cos(t); U.Zpp = @(t) -0.3i*sin(t);
% ----------------------
% U.Z = @(t) t + 1i; U.Zp = @(t) 1+0*t; U.Zpp = @(t) 0*t;
%  ---------------------------
U = setupquad(U,N);   % t=0 is left end, t=2pi right end
U.nx = -U.nx; U.cur = -U.cur; U.xp = -U.xp; U.tang=-U.tang; U.cw=-U.cw; % correct for sense of U, opp from periodicdirpipe
% reorder pts, makes no diff...
%U.x=U.x(end:-1:1); U.nx=U.nx(end:-1:1); U.xp=U.xp(end:-1:1); U.xpp=U.xpp(end:-1:1); U.tang=U.tang(end:-1:1); U.cw=U.cw(end:-1:1); U.w=U.w(end:-1:1); U.sp=U.sp(end:-1:1); U.cur=U.cur(end:-1:1); 

D.Z = @(t) t + 1i*(-1+0.3*sin(t)); D.Zp = @(t) 1 + 0.3i*cos(t);
D.Zpp = @(t) - 0.3i*sin(t);
% ----------------------
% D.Z = @(t) t - 1i; D.Zp = @(t) 1+0*t; D.Zpp = @(t) 0*t;
%  ---------------------------
D = setupquad(D,N);   % same dir (left to right); pipe wall normals outwards
% inside = @(z) imag(z-D.Z(real(z)))>0 & imag(z-U.Z(real(z)))<0; % ok U,D graphs
s = mergesegquads([U,D]);

% Choco Jan 2026: For unit test on JAX, save SLP and DLP matrices
mu = 0.7;
ttemp.x = 2+0.2j; ttemp.nx = 1+0j;
[u,p,T] = StoSLP(ttemp, s, mu);
save('SL.mat','u','p','T');
[u,p,T] = StoDLP(ttemp, s, mu);
save('DL.mat','u','p','T');
[u,p,T] = StoSLP(s, s, mu);
save('SLself.mat','u','p','T');
[u,p,T] = StoDLP(s, s, mu);
save('DLself.mat','u','p','T');

% Choco Dec 2025: add particle
Nptcl = N;
ptcl.Z = @(t) 1 + 0.3*cos(t) + 1j*(0.3*sin(t)+0.25);
ptcl.Zp = @(t) - 0.3*sin(t) + 1j*0.3*cos(t);
ptcl.Zpp = @(t) - 0.3*cos(t) - 1j*0.3*sin(t);
ptcl = setupquad(ptcl, Nptcl);
ptcl.nx = -ptcl.nx; ptcl.cur = -ptcl.cur; ptcl.xp = -ptcl.xp; ptcl.tang=-ptcl.tang; ptcl.cw=-ptcl.cw; % correct for sense of ptcl, opp from periodicdirpipe
% inside = @(z) imag(z-D.Z(real(z)))>0 & imag(z-U.Z(real(z)))<0 & ~inpolygon(real(z),imag(z),real(ptcl.x),imag(ptcl.x));

ptcl2.Z = @(t) 5 + 0.2*cos(t) + 1j*(0.2*sin(t)+0);
ptcl2.Zp = @(t) - 0.2*sin(t) + 1j*0.2*cos(t);
ptcl2.Zpp = @(t) - 0.2*cos(t) - 1j*0.2*sin(t);
ptcl2 = setupquad(ptcl2, Nptcl);
ptcl2.nx = -ptcl2.nx; ptcl2.cur = -ptcl2.cur; ptcl2.xp = -ptcl2.xp; ptcl2.tang=-ptcl2.tang; ptcl2.cw=-ptcl2.cw; % correct for sense of ptcl, opp from periodicdirpipe

ptcl_cell = {ptcl,ptcl2};

ptcl_tot = mergesegquads([ptcl,ptcl2]);
inside = @(z) imag(z-D.Z(real(z)))>0 & imag(z-U.Z(real(z)))<0 & ~inpolygon(real(z),imag(z),real(ptcl_tot.x),imag(ptcl_tot.x));

zt.x = [2+0.2i; 4+0.1i];    % point to test u soln at (expt='t' only)

% set up left and right walls
uc.nei = 1; % how many nei copies either side (use eg 1e3 to test A w/o AP)
uc.trlist = uc.e1*(-uc.nei:uc.nei);  % list of translations for direct images
% m = 80;    % pts per side wall
% m = 4;
[x w] = gauss(m); x = (1+x)/2; w = w'/2; % quadr on [0,1]
H = U.Z(0)-D.Z(0); L.x = D.Z(0) + H*x; L.nx = 0*L.x+1; L.w = H*w; % left side
R = L; R.x = L.x+uc.e1; % right side
uc.L = L; uc.R = R;

% set up aux periodizing basis
proxyrep = @StoSLP;      % sets proxy pt type via a kernel function call
Rp = 1.1*2*pi; 
M = 2*m;    % # proxy pts (2 force comps per pt, so 2M dofs)
p.x = pi + Rp*exp(2i*pi*(0:M-1)'/M); p = setupquad(p);     % proxy pts
if v>1, figure; showsegment({U,D,ptcl_tot},uc.trlist); showsegment({L,R}); plot(p.x,'r+'); plot(zt.x,'go'); 
    % plot(ptcl.x);  
end

mu = 0.7;                                           % fluid viscosity
if expt=='t' % Exact soln: either periodic or plus fixed pressure drop / period:
  % ue = @(x) [1+0*x; -2+0*x]; pe = @(x) 0*x; % the exact soln: uniform rigid flow, constant pressure everywhere (no drop)
  h=.2; ue = @(x) h*[imag(x).^2;0*x]; pe = @(x) h*2*mu*real(x); % horiz Poisseuil flow (has pres drop)
  % disp('expt=t: running known Poisseuil flow BVP...')
  vrhs = ue(s.x);        % bdry vel data: NB ordering Ux,Dx,Uy,Dy !
  vrhs_ptcl = ue(ptcl_tot.x);
  jump = pe(uc.e1)-pe(0); % known pressure growth across one period (a number)
elseif expt=='d'
  vrhs = zeros(2*numel(s.x),1);     % no-slip BCs, ie homog vel data on U,D
  vrhs_ptcl = zeros(2*numel(ptcl_tot.x),1);
  jump = -1;              % given pressure driving, for flow +x (a number)
  disp('expt=d: solving no-slip pressure-driven flow in pipe...')
end

tic
% [E,A,B,C,Q] = ELSmatrix(s,p,proxyrep,mu,uc);                % fill
% [E,A,B,C,Q] = ELSmatrix_sldl(s,ptcl,p,proxyrep,mu,uc); 
[E,A,B,C,Q] = ELSmatrix_multi(s,ptcl_cell,p,proxyrep,mu,uc); 

% data = load("Emat.mat");
% A_jax = data.A;
% B_jax = data.B;
% C_jax = data.C;
% Q_jax = data.Q;
% norm(A_jax-A)
% norm(B_jax-B)
% norm(C_jax-C)
% norm(Q_jax-Q)

% %{
Tjump = -jump * [real(R.nx);imag(R.nx)]; % traction driving growth (vector func)
% erhs = [vrhs; zeros(2*m,1);Tjump];       % expanded lin sys RHS
erhs = [vrhs; vrhs_ptcl; zeros(2*m,1);Tjump]; 

warning('off','MATLAB:nearlySingularMatrix')  % backward-stable ill-cond is ok!
warning('off','MATLAB:rankDeficientMatrix')
lso.RECT = true;  % linsolve opts, forces QR even when square
co = linsolve(E,erhs,lso);                           % direct bkw stable solve
toc
fprintf('resid norm = %.3g\n',norm(E*co - erhs))
sig = co(1:2*numel(s.x)); 
sig_ptcl = co(1+2*numel(s.x):2*numel(s.x)+2*numel(ptcl_tot.x));
psi = co(2*numel(s.x)+2*numel(ptcl_tot.x)+1:end);
fprintf('density norm = %.3g, ptcl denstiy norm = %.3g, proxy norm = %.3g\n',norm(sig), norm(sig_ptcl), norm(psi))
[ut pt] = evalsol_sldl(s,ptcl_tot,p,proxyrep,mu,uc,zt.x,co);
% [ut pt] = evalsol(s,p,proxyrep,mu,uc,zt.x,co);


format longE
% ut
if expt=='t'                         % check vs known soln at the point zt
  fprintf('u velocity err at zt = %.3g \n', norm(ut-ue(zt.x)))
  fprintf('perr at zt: ')
  pt-pe(zt.x)
else, 
    % fprintf('u velocity at zt = [%.15g, %.15g]; p at zt = %.15g \n', ut(1),ut(2),pt);
    fprintf('u velocity at zt: ')
    ut
    fprintf('p at zt: ')
    pt
end



if v   % plots
      nx = 80; gx = 2*pi*((1:nx)-0.5)/nx; ny = nx; gy = gx - pi; % plotting grid
      % gy = 6.0/5.0 * (((1:ny)-0.5)/ny - 0.5); % Choco Jan2026: to avoid
      % near eval errors and visualize actual interior to sin pile.
      [xx yy] = meshgrid(gx,gy); t.x = xx(:)+1i*yy(:); Mt = numel(t.x);
      di = reshape(inside(t.x),size(xx));  % boolean if inside domain
      if expt=='t', ueg = ue(t.x);      % evaluate & show known soln on grid...
            ue1 = reshape(ueg(1:Mt),size(xx)); ue2 = reshape(ueg(Mt+(1:Mt)),size(xx));
            peg = reshape(pe(t.x),size(xx));
            figure; imagesc(gx,gy, peg); colormap(jet(256)); colorbar;
            hold on; quiver(gx,gy, ue1,ue2, 10);
      else, figure; end
      showsegment({U,D,ptcl_tot},uc.trlist); showsegment({L,R}); plot(p.x,'r+'); plot(zt.x,'go'); 
      text(-.5,0,'L');text(2*pi+.5,0,'R');text(4,-1,'D');text(pi,0.5,'U');
      title('geom'); if expt=='t',title('geom and (u,p) known soln'); end
      % eval and plot soln...
      ug = nan(size([t.x;t.x])); pg = nan(size(t.x)); ii = inside(t.x);

      % [ug([ii;ii]) pg(ii)] = evalsol(s,p,proxyrep,mu,uc,t.x(ii),co);
      [ug([ii;ii]) pg(ii)] = evalsol_sldl(s,ptcl_tot,p,proxyrep,mu,uc,t.x(ii),co);

      u1 = reshape(ug(1:Mt),size(xx)); u2 = reshape(ug(Mt+(1:Mt)),size(xx));
      pp = reshape(pg,size(xx)); pp = pp - pp(ceil(ny/2),1); % zero p mid left edge
      figure; 
      % imagesc(gx,gy, pp); colormap(jet(256));
      % caxis(sort([0 jump])); colorbar;
      % hold on; 
      magvals = sqrt(u1.^2 + u2.^2);
      imagesc(gx,gy,log10(magvals)); colormap(jet(256)); colorbar; hold on; % Choco jan2026: plot magnitudes to see if no-slip was respected.
      quiver(gx,gy, u1,u2); title('soln (u,p), w/o close-eval scheme')
      showsegment({U,D,ptcl_tot}); showsegment({L,R}); plot(zt.x,'go'); 
      % plot(ptcl.x);
      if expt=='t'     % show error vs known...
            i=ceil(ny/2); j=ceil(nx/4); % index interior pt to get pres const
            pp = pp - pp(i,j) + peg(i,j);     % shift const in pres to match known
            eg2 = sum([u1(:)-ue1(:),u2(:)-ue2(:)].^2,2); % squared ptwise vector L2 errs
            figure; subplot(1,2,1); imagesc(gx,gy,log10(reshape(eg2,size(xx)))/2);
            axis xy equal tight; caxis([-16 0]); colorbar; hold on; plot(U.x,'k.-');
            plot(D.x,'k.-'); title('peri vel Stokes BVP: log_{10} u err')
            subplot(1,2,2); imagesc(gx,gy,log10(abs(pp-reshape(peg,size(xx)))));
            axis xy equal tight; caxis([-16 0]); colorbar; hold on;
            plot(U.x,'k.-'); plot(D.x,'k.-'); title('log_{10} p err (up to const)');
      end
end
% %}

% keyboard % don't forget to use dbquit to finish otherwise trapped in debug mode
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [u p] = evalsol(s,pr,proxyrep,mu,U,z,co,close) % eval soln rep u,p
% z = list of targets as C values. pr = proxy struct, s = source curve struct
% co = full coeff vec, U = unit cell struct, mu = viscosity. DLP only.
% Note: not for self-eval since srcsum2 used, 6/30/16. Taken from fig_stoconvK1.
% TODO: implement close eval.
if nargin<8, close=0; end              % default is plain native quadr
if close, error('close not implemented for open segments!'); end
z = struct('x',z);                     % make targets z a segment struct
N = numel(s.x);
sig = co(1:2*N); psi = co(2*N+1:end);  % split into sig (density) & psi (proxy)
if close
  % D = @(t,s,mu,dens) StoDLP_closeglobal(t,s,mu,dens,'e'); % exterior
  D = @(t,s,mu,dens) StoDLP_closeglobal(t,s,mu,dens,'e');
% else, D = @StoDLP; end                 % NB now native & close have same 4 args
else, D = @StoDLP; end
if nargout==1                          % don't want pressure output
  u = proxyrep(z,pr,mu,psi);           % init sol w/ proxies (always far)
  u = u + srcsum2(D,U.trlist,[],z,s,mu,sig);
else  
  [u p] = proxyrep(z,pr,mu,psi);       % init sol w/ proxies (always far)
  [uD pD] = srcsum2(D,U.trlist,[],z,s,mu,sig);
  u = u + uD;
  p = p + pD;
end

% Choco Jan2026: DL+SL formulation, with same density
function [u p] = evalsol_sldl(s,ptcl,pr,proxyrep,mu,U,z,co) % eval soln rep u,p
z = struct('x',z);                     % make targets z a segment struct
N = numel(s.x);
sig = co(1:2*N);
sig_ptcl = co(1+2*N:2*N+2*numel(ptcl.x));
psi = co(2*N+2*numel(ptcl.x)+1:end);  
if nargout==1                          % don't want pressure output
  u = proxyrep(z,pr,mu,psi);           % init sol w/ proxies (always far)
  u = u + srcsum(@StoDLP,U.trlist,[],z,s,mu,sig);
  u = u + srcsum(@StoSLP,U.trlist,[],z,ptcl,mu,sig_ptcl) + srcsum(@StoDLP,U.trlist,[],z,ptcl,mu,sig_ptcl);
else  
  [u p] = proxyrep(z,pr,mu,psi);       % init sol w/ proxies (always far)
  [uD pD] = srcsum(@StoDLP,U.trlist,[],z,s,mu,sig);
  [uDptcl pDptcl] = srcsum(@StoDLP,U.trlist,[],z,ptcl,mu,sig_ptcl);
  [uSptcl pSptcl] = srcsum(@StoSLP,U.trlist,[],z,ptcl,mu,sig_ptcl);
  u = u + uD + uDptcl + uSptcl;
  p = p + pD + pDptcl + pSptcl;
end


function [E A B C Q] = ELSmatrix(s,p,proxyrep,mu,uc)
% builds matrix blocks for Stokes extended linear system, D rep only
N = numel(s.x);

A = -eye(2*N)/2 + srcsum(@StoDLP,uc.trlist,[],s,s,mu);  % notes: DLP gives int JR term; srcsum self is ok

B = proxyrep(s,p,mu);     % map from proxy density to vel on curve
d = uc.e1*uc.nei;         % src transl to use

[CLD,~,TLD] = srcsum(@StoDLP,d,[],uc.L,s,mu);
[CRD,~,TRD] = srcsum(@StoDLP,-d,[],uc.R,s,mu);
C = [CRD-CLD; TRD-TLD];

[QL,~,QLt] = proxyrep(uc.L,p,mu); [QR,~,QRt] = proxyrep(uc.R,p,mu); % vel, tract
Q = [QR-QL; QRt-QLt];

E = [A B; C Q];

% Choco Jan2026: dl+sl formulation on particle
function [E A B C Q] = ELSmatrix_sldl(s,ptcl,p,proxyrep,mu,uc)
% builds matrix blocks for Stokes extended linear system, D rep only
N = numel(s.x);

% Separate formulation for wall vs ptcl
A11 = -eye(2*N)/2 + srcsum(@StoDLP,uc.trlist,[],s,s,mu);
A12 = srcsum(@StoDLP,uc.trlist,[],s,ptcl,mu) + srcsum(@StoSLP,uc.trlist,[],s,ptcl,mu);
A21 = srcsum(@StoDLP,uc.trlist,[],ptcl,s,mu);
A22 = -eye(2*numel(ptcl.x))/2 + srcsum(@StoDLP,uc.trlist,[],ptcl,ptcl,mu) + srcsum(@StoSLP,uc.trlist,[],ptcl,ptcl,mu);
A = [A11 A12; A21 A22];

B1 = proxyrep(s,p,mu);     % map from proxy density to vel on curve
B2 = proxyrep(ptcl,p,mu);
B = [B1;B2];
d = uc.e1*uc.nei;         % src transl to use

[CLD,~,TLD] = srcsum(@StoDLP,d,[],uc.L,s,mu); 
[CRD,~,TRD] = srcsum(@StoDLP,-d,[],uc.R,s,mu);
C1 = [CRD-CLD; TRD-TLD];
[CLD,~,TLD] = srcsum(@StoDLP,d,[],uc.L,ptcl,mu); 
[CLS,~,TLS] = srcsum(@StoSLP,d,[],uc.L,ptcl,mu);
[CRD,~,TRD] = srcsum(@StoDLP,-d,[],uc.R,ptcl,mu);
[CRS,~,TRS] = srcsum(@StoSLP,-d,[],uc.R,ptcl,mu);
C2 = [CRD-CLD + CRS-CLS; TRD-TLD + TRS-TLS];
C = [C1 C2];

[QL,~,QLt] = proxyrep(uc.L,p,mu); [QR,~,QRt] = proxyrep(uc.R,p,mu); % vel, tract
Q = [QR-QL; QRt-QLt];

E = [A B; C Q];

% Choc Jan 2026: dl+sl formulation; multi particle support
function [E A B C Q] = ELSmatrix_multi(s,ptcl_cell,p,proxyrep,mu,uc)
% builds matrix blocks for Stokes extended linear system, D rep only
N = numel(s.x);
% Collect info from all cells for far evals
ptcl_tot = ptcl_cell{1};
if numel(ptcl_cell)>1
    for cellind=2:numel(ptcl_cell)
        ptcl_tot = mergesegquads([ptcl_tot, ptcl_cell{cellind}]);
    end
end
% Separate formulation for wall vs ptcl
A11 = -eye(2*N)/2 + srcsum(@StoDLP,uc.trlist,[],s,s,mu); % Wall to wall
A12 = srcsum(@StoDLP,uc.trlist,[],s,ptcl_tot,mu) + srcsum(@StoSLP,uc.trlist,[],s,ptcl_tot,mu); % all particle to wall
A21 = srcsum(@StoDLP,uc.trlist,[],ptcl_tot,s,mu); % wall to ptcl
% A22 = -eye(2*numel(ptcl.x))/2 + srcsum(@StoDLP,uc.trlist,[],ptcl,ptcl,mu) + srcsum(@StoSLP,uc.trlist,[],ptcl,ptcl,mu);
A22 = -eye(2*numel(ptcl_tot.x))/2 + srcsum(@StoDLP,uc.trlist,[],ptcl_tot,ptcl_tot,mu) + srcsum_ptcl_wrapper(@StoSLP,uc.trlist,ptcl_cell,mu);
A = [A11 A12; A21 A22];

B1 = proxyrep(s,p,mu);     % map from proxy density to vel on curve
B2 = proxyrep(ptcl_tot,p,mu);
B = [B1;B2];
d = uc.e1*uc.nei;         % src transl to use

[CLD,~,TLD] = srcsum(@StoDLP,d,[],uc.L,s,mu); 
[CRD,~,TRD] = srcsum(@StoDLP,-d,[],uc.R,s,mu);
C1 = [CRD-CLD; TRD-TLD];
[CLD,~,TLD] = srcsum(@StoDLP,d,[],uc.L,ptcl_tot,mu); 
[CLS,~,TLS] = srcsum(@StoSLP,d,[],uc.L,ptcl_tot,mu);
[CRD,~,TRD] = srcsum(@StoDLP,-d,[],uc.R,ptcl_tot,mu);
[CRS,~,TRS] = srcsum(@StoSLP,-d,[],uc.R,ptcl_tot,mu);
C2 = [CRD-CLD + CRS-CLS; TRD-TLD + TRS-TLS];
C = [C1 C2];

[QL,~,QLt] = proxyrep(uc.L,p,mu); [QR,~,QRt] = proxyrep(uc.R,p,mu); % vel, tract
Q = [QR-QL; QRt-QLt];

E = [A B; C Q];

% for multiple particles contained in ptcl_cell, create large matrix with
% block structure [A11 A12 ...; A21 A22 ...] 
% Necessary since SLP self eval uses fft so requires input as one enclosed
% particle.
% Note: currently only one output since used only in self-to-self for all
% particles
% TODO: this currently does not give correct, converging flow in no-slip
% case.
function U = srcsum_ptcl_wrapper(kernel,trlist,ptcl_cell,mu)
    num_ptcl = numel(ptcl_cell);
    ptcl_tot = ptcl_cell{1};
    ptcl_idx = zeros(1,num_ptcl+1); % row of starting indices for each particle
    ptcl_idx(1) = 1; 
    for i=2:num_ptcl
        ptcl_tot = mergesegquads([ptcl_tot,ptcl_cell{i}]);
        ptcl_idx(i) = ptcl_idx(i-1) + numel(ptcl_cell{i-1}.x);
    end
    ptcl_tot_size = numel(ptcl_tot.x);
    ptcl_idx(num_ptcl+1) = ptcl_tot_size+1; % ghost entry for end of array
    U = zeros(ptcl_tot_size*2);

    for i=1:num_ptcl
        ptcl_i = ptcl_cell{i};
        srcind = [ptcl_idx(i):(ptcl_idx(i+1)-1),(ptcl_tot_size+ptcl_idx(i)) : (ptcl_tot_size+ptcl_idx(i+1)-1)];
        for j=1:num_ptcl
            trgind = [ptcl_idx(j):(ptcl_idx(j+1)-1),(ptcl_tot_size+ptcl_idx(j)) : (ptcl_tot_size+ptcl_idx(j+1)-1)];
            if i==j
                % ptcl_i self eval
                u = srcsum(kernel,trlist,[],ptcl_i,ptcl_i,mu);
            else
                ptcl_j = ptcl_cell{j};
                % particle i (with nbrs) to ptcl j
                u = srcsum(kernel,trlist,[],ptcl_j,ptcl_i,mu);
            end
            U(trgind,srcind) = U(trgind, srcind) + u;
        end
    end
