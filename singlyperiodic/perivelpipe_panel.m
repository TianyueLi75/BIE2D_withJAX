% Panel based discretization on walls.

U = []; D = []; % clear structs
% U.Z = @(t) 2*pi-t + 1j;
% U.Zp = @(t) -1 + 0.*t;
% U.Zpp = @(t) 0.*t;
U.Z = @(t) (2*pi-t) + 1i*(1+0.3*sin(2*pi-t)); U.Zp = @(t) -1 - 0.3i*cos(2*pi-t); U.Zpp = @(t) -0.3i*sin(2*pi-t);
% D.Z = @(t) t - 1j;
% D.Zp = @(t) 1 + 0.*t;
% D.Zpp = @(t) 0.*t;
D.Z = @(t) t + 1i*(-1+0.3*sin(t)); D.Zp = @(t) 1 + 0.3i*cos(t); D.Zpp = @(t) - 0.3i*sin(t);

% Panel based quadrature using GL grids, use axigeom functions
Num_panels = 40;
p = 10; % order on panel
N_perwall = p * Num_panels; qtype = 'p'; qntype = 'G'; 
U.p = p; D.p = p;
[U,~] = quadr(U, N_perwall, qtype, qntype); 
U.trlist = [-2*pi,2*pi];
U.tpan = [U.tlo;U.thi(end)];
U.cw = U.wxp; 
% U.nx = -U.nx; U.cur = -U.cur; U.xp = -U.xp; U.tang=-U.tang; U.cw=-U.cw; % correct for sense of U, opp from periodicdirpipe
[D,~] = quadr(D, N_perwall, qtype, qntype); 
D.trlist = [-2*pi,2*pi];
D.tpan = [D.tlo;D.thi(end)];
D.cw = D.wxp;
s = mergesegquads([U,D]);
s.cw = [U.cw; D.cw];
s.np = U.np + D.np;
s.tlo = [U.tlo; D.tlo];
s.thi = [U.thi; D.thi];
s.xlo = [U.xlo; D.xlo];
s.xhi = [U.xhi; D.xhi];
s.ws = [U.ws; D.ws];
s.wxp = [U.wxp; D.wxp];
s.tpan = [U.tpan; D.tpan];
% inside = @(z) imag(z-D.Z(real(z)))>0 & imag(z-U.Z(real(z)))<0; % Note: this also assumes input z is same domain and ratio as input t..
% Instead, form UX, DX functions of x..
UX = @(x) x + 1i*(1+0.3*sin(x));
DX = @(x) x + 1i*(-1+0.3*sin(x));
inside = @(z) imag(z-DX(real(z)))>0 & imag(z-UX(real(z)))<0; 

% % Global quadrature -- set up upper and lower walls
% N = 40;
% % U.Z = @(t) t + 1i*(1+0.3*sin(t)); U.Zp = @(t) 1 + 0.3i*cos(t); U.Zpp = @(t) -0.3i*sin(t);
% U = setupquad(U,N);   % t=0 is left end, t=2pi right end
% U.nx = -U.nx; U.cur = -U.cur; U.xp = -U.xp; U.tang=-U.tang; U.cw=-U.cw; % correct for sense of U, opp from periodicdirpipe
% % D.Z = @(t) t + 1i*(-1+0.3*sin(t)); D.Zp = @(t) 1 + 0.3i*cos(t); D.Zpp = @(t) - 0.3i*sin(t);
% D = setupquad(D,N);   % same dir (left to right); pipe wall normals outwards
% s = mergesegquads([U,D]);

figure; 
quiver(real(s.x),imag(s.x),real(s.nx),imag(s.nx)); 

% set up left and right walls
uc.e1 = 2*pi;
uc.nei = 1; % how many nei copies either side (use eg 1e3 to test A w/o AP)
uc.trlist = uc.e1*(-uc.nei:uc.nei);  % list of translations for direct images
m = 120;    % pts per side wall
% [x, w] = gauss(m); 
[x, w] = gauss_here(m); % GL grid..
x = (1+x)/2; w = w'/2; % quadr on [0,1]
H = U.x(end)-D.x(1); L.x = D.Z(0) + H*x; L.nx = 0*L.x+1; L.w = H*w; % left side
R = L; R.x = L.x+uc.e1; % right side
uc.L = L; uc.R = R;

% set up aux periodizing basis
proxyrep = @StoSLP;      % sets proxy pt type via a kernel function call
Rp = 1.1*2*pi; 
M = 2*m;    % # proxy pts (2 force comps per pt, so 2M dofs)
P.x = pi + Rp*exp(2i*pi*(0:M-1)'/M); P = setupquad(P);     % proxy pts

expt = 'd';

mu = 1.0;  % fluid viscosity
if expt=='t' % Exact soln: either periodic or plus fixed pressure drop / period:
  % ue = @(x) [1+0*x; -2+0*x]; pe = @(x) 0*x; % the exact soln: uniform rigid flow, constant pressure everywhere (no drop)
  h=.2; ue = @(x) h*[imag(x).^2;0*x]; pe = @(x) h*2*mu*real(x); % horiz Poisseuil flow (has pres drop)
  % disp('expt=t: running known Poisseuil flow BVP...')
  vrhs = ue(s.x);        % bdry vel data: NB ordering Ux,Dx,Uy,Dy !
  jump = pe(uc.e1)-pe(0); % known pressure growth across one period (a number)
elseif expt=='d'
  vrhs = zeros(2*numel(s.x),1);     % no-slip BCs, ie homog vel data on U,D
  jump = -1;              % given pressure driving, for flow +x (a number)
  disp('expt=d: solving no-slip pressure-driven flow in pipe...')
end
    
% Use existing density (already solved)
[E,A,B,C,Q] = ELSmatrix(s,P,proxyrep,mu,uc);                % fill

Tjump = -jump * [real(R.nx);imag(R.nx)]; % traction driving growth (vector func)
erhs = [vrhs; zeros(2*m,1);Tjump];       % expanded lin sys RHS
dens = E \ erhs;
fprintf('resid norm = %.3g\n',norm(E*dens - erhs))
sig = dens(1:2*numel(s.x)); 
psi = dens(1+2*numel(s.x):end);
fprintf('density norm = %.3g, proxy norm = %.3g\n',norm(sig)/numel(sig), norm(psi)/numel(psi));

close = 1;
if close
    % Set near targets
    zt.x = [ 1 + 0.9i];
else
    % Set far targets
    zt.x = [2+0.2i; 4+0.1i]; 
end
[ut, pt] = evalsol(s,P,proxyrep,mu,uc,zt.x,dens,close);
% ut = evalsol(s,P,proxyrep,mu,uc,zt.x,dens,1); % near eval doesn't implement p yet.

format longE
% ut
if expt=='t'                         % check vs known soln at the point zt
    fprintf('u velocity err at zt = %.3g \n', norm(ut-ue(zt.x)))
    % fprintf('perr at zt: ')
    % pt-pe(zt.x)
else
    % fprintf('u velocity at zt = [%.15g, %.15g]; p at zt = %.15g \n', ut(1),ut(2),pt);
    fprintf('u velocity at zt: ')
    ut
    % fprintf('p at zt: ')
    % pt
end

nx = 160; gx = 2*pi*((1:nx)-0.5)/nx; ny = nx; gy = gx - pi; % plotting grid
[xx yy] = meshgrid(gx,gy); t.x = xx(:)+1i*yy(:); Mt = numel(t.x);
di = reshape(inside(t.x),size(xx));  % boolean if inside domain
if expt=='t'
    ueg = ue(t.x);      % evaluate & show known soln on grid...
    ue1 = reshape(ueg(1:Mt),size(xx)); ue2 = reshape(ueg(Mt+(1:Mt)),size(xx));
    peg = reshape(pe(t.x),size(xx));
    % figure; imagesc(gx,gy, peg); colormap(jet(256)); colorbar;
    % hold on; quiver(gx,gy, ue1,ue2, 10);
else
    % figure; 
end
% showsegment({U,D},uc.trlist); showsegment({L,R}); plot(P.x,'r+'); plot(zt.x,'go'); 
% text(-.5,0,'L');text(2*pi+.5,0,'R');text(4,-1,'D');text(pi,0.5,'U');
% title('geom'); 
% if expt=='t'
%     title('geom and (u,p) known soln'); 
% end

% eval and plot soln...
ug = nan(size([t.x;t.x])); pg = nan(size(t.x)); ii = inside(t.x);

[ug([ii;ii]) pg(ii)] = evalsol(s,P,proxyrep,mu,uc,t.x(ii),dens,close); 

u1 = reshape(ug(1:Mt),size(xx)); u2 = reshape(ug(Mt+(1:Mt)),size(xx));
pp = reshape(pg,size(xx)); pp = pp - pp(ceil(ny/2),1); % zero p mid left edge
figure; 
% imagesc(gx,gy, pp); colormap(jet(256));
% caxis(sort([0 jump])); 
% colorbar;
% hold on; 
magvals = sqrt(u1.^2 + u2.^2);
imagesc(gx,gy,log10(magvals)); colormap(jet(256)); colorbar; hold on; % Choco jan2026: plot magnitudes to see if no-slip was respected.
% quiver(gx,gy, u1,u2); title('soln (u,p), with close-eval scheme')
showsegment({U,D}); showsegment({L,R}); plot(zt.x,'go'); 
% plot(ptcl.x);
if expt=='t'     % show error vs known...
    i=ceil(ny/2); j=ceil(nx/4); % index interior pt to get pres const
    % pp = pp - pp(i,j) + peg(i,j);     % shift const in pres to match known
    eg2 = sum([u1(:)-ue1(:),u2(:)-ue2(:)].^2,2); % squared ptwise vector L2 errs
    figure; 
    % subplot(1,2,1); 
    imagesc(gx,gy,log10(reshape(eg2,size(xx)))/2);
    axis xy equal tight; caxis([-16 0]); colorbar; hold on; plot(U.x,'k.-');
    plot(D.x,'k.-'); title('peri vel Stokes BVP: log_{10} u err')
    % subplot(1,2,2); imagesc(gx,gy,log10(abs(pp-reshape(peg,size(xx)))));
    % axis xy equal tight; caxis([-16 0]); colorbar; hold on;
    % plot(U.x,'k.-'); plot(D.x,'k.-'); title('log_{10} p err (up to const)');
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
end

function [u, p] = evalsol(s,pr,proxyrep,mu,U,z,co,close) % eval soln rep u,p
    % z = list of targets as C values. pr = proxy struct, s = source curve struct
    % co = full coeff vec, U = unit cell struct, mu = viscosity. DLP only.
    % Note: not for self-eval since srcsum2 used, 6/30/16. Taken from fig_stoconvK1.
    if nargin<8, close=0; end              % default is plain native quadr
    N = numel(s.x);
    z = struct('x',z); 
    sig = co(1:2*N); psi = co(2*N+1:end);  % split into sig (density) & psi (proxy)
    if close
        % D = @(t,s,mu,dens) StoDLP_closepanel(t,s,mu,dens,'i'); 
        D = @StoDLP_closepanel;
    else
        D = @StoDLP; 
    end
    if nargout==1                          % don't want pressure output
        u = proxyrep(z,pr,mu,psi);           % init sol w/ proxies (always far)
        % u = u + srcsum2(D,U.trlist,[],z,s,mu,sig);
        u = u + srcsum(D,U.trlist,[],z,s,mu,sig);
    else  
        [u, p] = proxyrep(z,pr,mu,psi);       % init sol w/ proxies (always far)
        [uD, pD] = srcsum2(D,U.trlist,[],z,s,mu,sig);
        u = u + uD;
        p = p + pD;
    end 
end

function [x, w, D] = gauss_here(N) 
    % Gauss-Legendre nodes and weights on [-1,1]
    % Tidied up from lgwt.m code by Greg von Winckel 02/25/2004, via Gillman.
    % Included spectral diff matrix 10/23/2014, Bowei Wu
    N=N-1; N1=N+1; N2=N+2;
    xu=linspace(-1,1,N1)';
    y=cos((2*(0:N)'+1)*pi/(2*N+2))+(0.27/N1)*sin(pi*xu*N/N2); % Initial guess
    L=zeros(N1,N2);   % Legendre-Gauss Vandermonde Matrix
    Lp=zeros(N1,N2);  % Derivative of LGVM
    % Compute the zeros of the N+1 Legendre Polynomial
    % using the recursion relation and the Newton-Raphson method
    y0=2; % Iterate until new points are uniformly within epsilon of old points:
    while max(abs(y-y0))>eps      
      L(:,1)=1; Lp(:,1)=0;
      L(:,2)=y; Lp(:,2)=1;
      for k=2:N1, L(:,k+1)=( (2*k-1)*y.*L(:,k)-(k-1)*L(:,k-1) )/k; end
      Lp=(N2)*( L(:,N1)-y.*L(:,N2) )./(1-y.^2);       
      y0=y;
      y=y0-L(:,N2)./Lp;
    end
    x = y(end:(-1):1);
    w=2./((1-y.^2).*Lp.^2)*(N2/N1)^2; % Compute the weights
     
    if nargout > 2
        N = N1;
        index = (1:N)';
        % Construct differentiation matrix (see Fornberg book, p. 51):
          D = zeros(N,N); a = zeros(N,1);
          for k = 1:N
            notk = index~=k;
            a(k) = prod(x(k)-x(notk));
          end
          for k = 1:N
            notk = index~=k;
            D(notk,k) = (a(notk)/a(k))./(x(notk)-x(k));
            D(k,k) = sum(1./(x(k)-x(notk)));
          end
    end
end
