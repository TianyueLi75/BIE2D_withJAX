% singly-periodic in 2D velocity-BC Stokes BVP, ie "pipe" geom and
% particles under rigid body motion. Choco 2/16/2026

Num_panels = 10;
p = 10; % order on panel
N_perwall = p * Num_panels; qtype = 'p'; qntype = 'G'; 
Nptcl = 400;
Nlr = 80;
Nprx = 2*Nlr;    % # proxy pts (2 force comps per pt, so 2M dofs)

% set up upper and lower walls
uc.e1 = 2*pi;   % unitcell, e1=lattice vector as a complex number
U.Z = @(t) 2*pi-t + 1i*(1+0.3*sin(2*pi-t)); U.Zp = @(t) -1 - 0.3i*cos(2*pi-t); U.Zpp = @(t) -0.3i*sin(2*pi-t);
% U.Z = @(t) 2*pi-t + 1i; U.Zp = @(t) -1+0*t; U.Zpp = @(t) 0*t;
D.Z = @(t) t + 1i*(-1+0.3*sin(t)); D.Zp = @(t) 1 + 0.3i*cos(t); D.Zpp = @(t) - 0.3i*sin(t);
% D.Z = @(t) t - 1i; D.Zp = @(t) 1+0*t; D.Zpp = @(t) 0*t;
% Panel based quadrature using GL grids, use axigeom functions
U.p = p; D.p = p;
[U,~] = quadr(U, N_perwall, qtype, qntype); 
U.trlist = [-uc.e1,uc.e1];
U.tpan = [U.tlo;U.thi(end)];
U.cw = U.wxp; 
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
UX = @(x) x + 1i*(1+0.3*sin(x)); % Functions in left-to-right orientation in x for target selecting
DX = @(x) x + 1i*(-1+0.3*sin(x));
% UX = @(x) x + 1i;
% DX = @(x) x - 1i;

ptcl.Z = @(t) 1 + 0.2*cos(t) + 1j*(0.3*sin(t)+0.25); % changed center to y=0 for symmetric setup (for debugging)
ptcl.Zp = @(t) - 0.2*sin(t) + 1j*0.3*cos(t);
ptcl.Zpp = @(t) - 0.2*cos(t) - 1j*0.3*sin(t);
ptcl = setupquad(ptcl, Nptcl);
% ptcl = wobblycurve(0.2,0.1,5,Nptcl);
% ptcl.x = ptcl.x + 1+0.25j;
ptcl.a = 1+0.25j;

% ptcl2.Z = @(t) 5 + 0.2*cos(t) + 1j*(0.2*sin(t)+0);
% ptcl2.Zp = @(t) - 0.2*sin(t) + 1j*0.2*cos(t);
% ptcl2.Zpp = @(t) - 0.2*cos(t) - 1j*0.2*sin(t);
% ptcl2 = setupquad(ptcl2, Nptcl);
% ptcl2.a = 5+0j;
% ptcl_cell = {ptcl,ptcl2};

ptcl_cell = {ptcl};

% Collection of all panels on all particles
ptcl_tot = ptcl;
for ptcl_ind=2:numel(ptcl_cell)
    ptcl_tot = mergesegquads([ptcl_tot, ptcl_cell{ptcl_ind}]);
end
% inside = @(z) imag(z-DX(real(z)))>0 & imag(z-UX(real(z)))<0 & ~inpolygon(real(z),imag(z),real(ptcl.x),imag(ptcl.x))& ~inpolygon(real(z),imag(z),real(ptcl2.x),imag(ptcl2.x));
inside = @(z) imag(z-DX(real(z)))>0 & imag(z-UX(real(z)))<0 & ~inpolygon(real(z),imag(z),real(ptcl.x),imag(ptcl.x));
% TODO: more generic "inside" function


zt.x = [2+0.2i; 4+0.1i];    % point to test u soln at

% set up left and right walls
L = []; R = [];
uc.nei = 1; % how many nei copies either side (use eg 1e3 to test A w/o AP)
uc.trlist = uc.e1*(-uc.nei:uc.nei);  % list of translations for direct images
[x w] = gauss(Nlr); x = (1+x)/2; w = w'/2; % quadr on [0,1]
H = U.x(end)-D.x(1); L.x = D.Z(0) + H*x; L.nx = 0*L.x+1; L.w = H*w; % left side
R = L; R.x = L.x+uc.e1; % right side
uc.L = L; uc.R = R;

% set up aux periodizing basis
P = [];
proxyrep = @StoSLP;      % sets proxy pt type via a kernel function call
Rp = 1.1*2*pi; 
P.x = pi + Rp*exp(2i*pi*(0:Nprx-1)'/Nprx); P = setupquad(P);     % proxy pts

mu = 0.7;   % fluid viscosity
vrhs = zeros(2*numel(s.x),1);

B1 = 1.23;
B2 = -0.73;
vrhs_ptcl = get_vslip(B1,B2,ptcl_cell); 

jump = 0;

[E, A1, B1,C1,Q1] = ELSmatrix_rbm(s,ptcl_cell,P,proxyrep,mu,uc); 

Tjump = -jump * [real(R.nx);imag(R.nx)]; % traction driving growth (vector func)
erhs = [vrhs; vrhs_ptcl; zeros(3*numel(ptcl_cell),1); zeros(2*Nlr,1);Tjump]; 

warning('off','MATLAB:nearlySingularMatrix')  % backward-stable ill-cond is ok!
warning('off','MATLAB:rankDeficientMatrix')
lso.RECT = true;  % linsolve opts, forces QR even when square
co = linsolve(E,erhs,lso);                           % direct bkw stable solve
fprintf('resid norm = %.3g\n',norm(E*co - erhs))
sig = co(1:2*numel(s.x)); 
sig_ptcl = co(1+2*numel(s.x):2*numel(s.x)+2*numel(ptcl_tot.x));
U_ptcl_all = co(2*numel(s.x)+2*numel(ptcl_tot.x) + 1 : 2*numel(s.x)+2*numel(ptcl_tot.x) + 2*numel(ptcl_cell)); % 2 U entries per ptcl
Omega_ptcl_all = co(2*numel(s.x)+2*numel(ptcl_tot.x)+2*numel(ptcl_cell)+1:2*numel(s.x)+2*numel(ptcl_tot.x)+3*numel(ptcl_cell)); % 1 Omega entry per ptcl
psi = co(2*numel(s.x)+2*numel(ptcl_tot.x)+3*numel(ptcl_cell)+1:end);
fprintf('density norm = %.3g, ptcl density norm = %.3g, proxy norm = %.3g\n',norm(sig)/numel(sig), norm(sig_ptcl)/numel(sig_ptcl), norm(psi)/numel(psi));
U_ptcl_all
Omega_ptcl_all
[ut, pt] = evalsol(s,ptcl_tot,ptcl_cell,P,proxyrep,mu,uc,zt.x,co);

nx = 160; gx = 2*pi*((1:nx)-0.5)/nx; ny = nx; 
gy = gx - pi; % plotting grid
% gy = 8*((1:ny)/ny-0.5);
[xx yy] = meshgrid(gx,gy); t.x = xx(:)+1i*yy(:); Mt = numel(t.x);
di = reshape(inside(t.x),size(xx));  % boolean if inside domain
ug = nan(size([t.x;t.x])); pg = nan(size(t.x)); ii = inside(t.x);
[ug([ii;ii]), pg(ii)] = evalsol(s,ptcl_tot,ptcl_cell,P,proxyrep,mu,uc,t.x(ii),co);

u1 = reshape(ug(1:Mt),size(xx)); u2 = reshape(ug(Mt+(1:Mt)),size(xx));
pp = reshape(pg,size(xx)); pp = pp - pp(ceil(ny/2),1); % zero p mid left edge
figure; 
magvals = sqrt(u1.^2 + u2.^2);
imagesc(gx,gy,log10(magvals)); % colormap(jet(256)); 
colorbar; hold on; 
% quiver(gx,gy, u1,u2,2); 
[startX, startY] = meshgrid(gx(1:5:end), gy(1:5:end));
verts = stream2(gx,gy,u1,u2,startX,startY);
streamline(verts); 
title('soln u, with close-eval scheme')
% showsegment({U,D,ptcl_tot}); 
plot(U.x);
plot(D.x);
for ptcl_ind=1:numel(ptcl_cell)
    plot(ptcl_cell{ptcl_ind}.x);
end
showsegment({L,R}); plot(zt.x,'go'); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vslip_tot = get_vslip(B1,B2,ptcl_cell)
    vslip_tot = [];
    for ptcl_ind=1:numel(ptcl_cell)
        s_sqr = ptcl_cell{ptcl_ind};
        Xc = sum(real(s_sqr.x))/numel(s_sqr.x) + 1j*sum(imag(s_sqr.x))/numel(s_sqr.x);
        XmXc = s_sqr.x-Xc; 
        theta = atan2(imag(XmXc), real(XmXc));
        u_theta = B1 * sin(theta) + B2 * sin(theta) .* cos(theta);
        % vslip = [-u_theta .* sin(theta); u_theta .* cos(theta)];
        vslip = [u_theta .* real(s_sqr.tang); u_theta .* imag(s_sqr.tang)];
        vslip_tot = [vslip_tot; vslip];
    end
end

% Evaluation function (includes close eval on panel-based wall and global_quadr particle
function [u, p] = evalsol(s,ptcl_tot,ptcl_cell,pr,proxyrep,mu,U,z,co) % eval soln rep u,p
    z = struct('x',z);                     % make targets z a segment struct
    N = numel(s.x);
    sig = co(1:2*N);
    sig_ptcl_tot = co(1+2*N:2*N+2*numel(ptcl_tot.x));
    % Ux, Uy, Omega_z
    psi = co(2*N+2*numel(ptcl_tot.x)+3*numel(ptcl_cell)+1:end);  
    if nargout==1                          % don't want pressure output
        u = proxyrep(z,pr,mu,psi);           % init sol w/ proxies (always far)
        u = u + srcsum(@StoDLP_closepanel,U.trlist,[],z,s,mu,sig);
        % u = u + srcsum(@StoSLP,U.trlist,[],z,ptcl,mu,sig_ptcl) + srcsum(@StoDLP,U.trlist,[],z,ptcl,mu,sig_ptcl);
        start_ind = 0;
        ptcl_DL = @(t,s,mu,sigma) StoDLP_closeglobal(t,s,mu,sigma,'e');
        ptcl_SL = @(t,s,mu,sigma) StoSLP_closeglobal(t,s,mu,sigma,'e');
        for ptcl_ind=1:numel(ptcl_cell)
            ptcl_here = ptcl_cell{ptcl_ind};
            sig_ptcl = [sig_ptcl_tot(start_ind + (1:numel(ptcl_here.x))); sig_ptcl_tot(numel(ptcl_tot.x) + start_ind + (1:numel(ptcl_here.x)))];
            u = u + srcsum(ptcl_SL,U.trlist,[],z,ptcl_here,mu,sig_ptcl) + srcsum(ptcl_DL,U.trlist,[],z,ptcl_here,mu,sig_ptcl);
            start_ind = start_ind + numel(ptcl_here.x);
        end
    else  
        [u, p] = proxyrep(z,pr,mu,psi);       % init sol w/ proxies (always far)
        [uD, pD] = srcsum(@StoDLP_closepanel,U.trlist,[],z,s,mu,sig);
        % [uDptcl pDptcl] = srcsum(@StoDLP,U.trlist,[],z,ptcl,mu,sig_ptcl);
        % [uSptcl pSptcl] = srcsum(@StoSLP,U.trlist,[],z,ptcl,mu,sig_ptcl);
        u = u + uD;
        p = p + pD;
        start_ind = 0;
        ptcl_DL = @(t,s,mu,sigma) StoDLP_closeglobal(t,s,mu,sigma,'e');
        ptcl_SL = @(t,s,mu,sigma) StoSLP_closeglobal(t,s,mu,sigma,'e');
        for ptcl_ind=1:numel(ptcl_cell)
            ptcl_here = ptcl_cell{ptcl_ind};
            sig_ptcl = [sig_ptcl_tot(start_ind + (1:numel(ptcl_here.x))); sig_ptcl_tot(numel(ptcl_tot.x) + start_ind + (1:numel(ptcl_here.x)))];
            [uDptcl, pDptcl] = srcsum(ptcl_DL,U.trlist,[],z,ptcl_here,mu,sig_ptcl);
            [uSptcl, pSptcl] = srcsum(ptcl_SL,U.trlist,[],z,ptcl_here,mu,sig_ptcl);
            u = u + uDptcl + uSptcl;
            p = p + pDptcl + pSptcl;
            start_ind = start_ind + numel(ptcl_here.x);
        end
    end
end

function [E, BCgammaMat, B, C, Q] = ELSmatrix_rbm(s,ptcl_cell,P,proxyrep,mu,uc)
    % builds matrix blocks for Stokes extended linear system, D rep only
    N = numel(s.x);
    % Collect info from all cells for far evals
    ptcl_tot = ptcl_cell{1};
    if numel(ptcl_cell)>1
        for cellind=2:numel(ptcl_cell)
            ptcl_tot = mergesegquads([ptcl_tot, ptcl_cell{cellind}]);
        end
    end
    % Source: wall.
    A11 = -eye(2*N)/2 + srcsum(@StoDLP,uc.trlist,[],s,s,mu); % Wall to wall
    [A21,~,A21_T] = srcsum(@StoDLP_closepanel,uc.trlist,[],ptcl_tot,s,mu); % wall to ptcl
    % [A21,~,A21_T] = srcsum(@StoDLP,uc.trlist,[],ptcl_tot,s,mu); % wall to ptcl
    % Source: particle.
    A12 = srcsum_ptcl_wrapper(@StoDLP_closeglobal,uc.trlist,s,ptcl_cell,mu) + srcsum_ptcl_wrapper(@StoSLP_closeglobal,uc.trlist,s,ptcl_cell,mu); % all particle to wall
    % A12 = srcsum(@StoDLP,uc.trlist,[],s,ptcl_tot,mu) + srcsum_ptcl_wrapper(@StoSLP,uc.trlist,s,ptcl_cell,mu); % all particle to wall
    [A22_dl,~,~] = srcsum_ptclself_wrapper(@StoDLP_closeglobal,@StoDLP,uc.trlist,ptcl_cell,mu); % particle all to all DL
    % [A22_dl,~,~] = srcsum(@StoDLP,uc.trlist,[],ptcl_tot,ptcl_tot,mu); % particle all to all DL
    % Near copies DL still contribute to traction.
    ptcl_l = ptcl_tot;
    ptcl_l.x = ptcl_tot.x + uc.trlist(1);
    [~,~,A22_dl_Tl] = StoDLP(ptcl_tot, ptcl_l, mu);
    ptcl_l.x = ptcl_tot.x + uc.trlist(3); % hardcoded for 3 copies only
    [~,~,A22_dl_Tr] = StoDLP(ptcl_tot, ptcl_l, mu);
    % All 3 center copies SL contribute to traction
    [A22_sl,~,A22_slT] = srcsum_ptclself_wrapper(@StoSLP_closeglobal,@StoSLP,uc.trlist,ptcl_cell,mu); % particle all to all SL
    % [A22_sl,~,A22_slT] = srcsum_ptclself_wrapper(@StoSLP,uc.trlist,ptcl_cell,mu); % particle all to all SL
    A22 = eye(2*numel(ptcl_tot.x))/2 + A22_dl + A22_sl; % ptcl all to all DL+SL+jump
    A = [A11 A12; A21 A22];
    
    % Source: proxy
    B1 = proxyrep(s,P,mu);     % proxy to wall
    [B2,~,B2_T] = proxyrep(ptcl_tot,P,mu); % proxy to all particles
    B = [B1;B2];
    
    % Top left matrix incorporating U and Omega in B.C. on particles.
    BCgammaMat = zeros(numel(s.x)*2 + numel(ptcl_tot.x)*2, numel(s.x)*2 + numel(ptcl_tot.x)*2 + 3*numel(ptcl_cell));
    BCgammaMat(:, 1:(numel(s.x)*2+numel(ptcl_tot.x)*2)) = A; % wall+ptcl -> wall+ptcl
    BC_sqr = zeros(numel(ptcl_tot.x)*2,3*numel(ptcl_cell)); % sub matrix containing U, Omega for particle B.C. only.
    XmXc_tot = zeros(numel(ptcl_tot.x),1); % X-Xc collected, complex style
    for ptcl_ind=1:numel(ptcl_cell)
        s_sqr = ptcl_cell{ptcl_ind};
        Xc = sum(real(s_sqr.x))/numel(s_sqr.x) + 1j*sum(imag(s_sqr.x))/numel(s_sqr.x);
        XmXc = s_sqr.x-Xc; 
        XmXc_tot((ptcl_ind-1)*numel(s_sqr.x)+(1:numel(s_sqr.x))) = XmXc;
        BC_sqr((ptcl_ind-1)*numel(s_sqr.t)+(1:numel(s_sqr.t)), (ptcl_ind-1)*2 + 1) = -1.*ones(1,numel(s_sqr.t)); % Ux -> BCx
        BC_sqr(numel(ptcl_tot.x)+(ptcl_ind-1)*numel(s_sqr.t)+(1:numel(s_sqr.t)), (ptcl_ind-1)*2+2) = -1.*ones(1,numel(s_sqr.t)); % Uy -> BCy
        BC_sqr((ptcl_ind-1)*numel(s_sqr.t)+(1:numel(s_sqr.t)), numel(ptcl_cell)*2+ptcl_ind) = -imag(XmXc); % Omega -> BCx
        BC_sqr(numel(ptcl_tot.x)+(ptcl_ind-1)*numel(s_sqr.t)+(1:numel(s_sqr.t)), numel(ptcl_cell)*2+ptcl_ind) = real(XmXc); % Omega -> BCy
    end
    BCgammaMat((1+numel(s.x)*2):end, 1+numel(s.x)*2+numel(ptcl_tot.x)*2:end) = BC_sqr;
    
    % Force from all ptcls, wall, and proxy; integrate over particle
    % surfaces -> .* by integrating weights.
    T3 = A22_slT + A22_dl_Tl + A22_dl_Tr;
    fMat_sqr = -(-eye(size(T3))/2 + T3); % Exterior problem traction jump condition
    fMat_wall = - A21_T;
    fMat_prx = - B2_T; 

    num_ptcls = numel(ptcl_cell);
    n_total_pts = numel(ptcl_tot.x);
    
    % Pre-allocate sub-blocks
    WFx = zeros(num_ptcls, n_total_pts); % Maps fx to Fx_total
    WFy = zeros(num_ptcls, n_total_pts); % Maps fy to Fy_total
    WTx = zeros(num_ptcls, n_total_pts); % Maps fx to T (contribution)
    WTy = zeros(num_ptcls, n_total_pts); % Maps fy to T (contribution)
    
    current_idx = 0;
    for p = 1:num_ptcls
        ptcl = ptcl_cell{p};
        n_p = numel(ptcl.x);
        idx = current_idx + (1:n_p);
        
        % Centroid and relative distance
        Xc = sum(ptcl.x) / n_p;
        XmXc = ptcl.x - Xc;
        
        % Force weights: Integral(f_i)
        WFx(p, idx) = ptcl.w(:).';
        WFy(p, idx) = ptcl.w(:).';
        
        % Torque weights: Integral( -imag(XmXc)*fx + real(XmXc)*fy )
        WTx(p, idx) = -ptcl.w(:).' .* imag(XmXc(:).');
        WTy(p, idx) =  ptcl.w(:).' .* real(XmXc(:).');
        
        current_idx = current_idx + n_p;
    end

    intF = [ WFx,           zeros(size(WFx));   % Outputs Fx1, Fx2...
          zeros(size(WFy)), WFy] * ...
        [fMat_wall, fMat_sqr, zeros(2*numel(ptcl_tot.x), 3*numel(ptcl_cell)), fMat_prx];
    intT = [WTx,WTy] * ...
        [fMat_wall, fMat_sqr, zeros(2*numel(ptcl_tot.x), 3*numel(ptcl_cell)), fMat_prx];
    % intF = [ptcl_tot.w(:)', zeros(1,numel(ptcl_tot.w)); zeros(1,numel(ptcl_tot.w)), ptcl_tot.w(:)'] * ...
    %     [fMat_wall, fMat_sqr, zeros(2*numel(ptcl_tot.x), 3*numel(ptcl_cell)), fMat_prx];
    % intT = [-ptcl_tot.w(:)'.*imag(XmXc_tot)', ptcl_tot.w(:)'.* real(XmXc_tot)'] * ...
    %     [fMat_wall, fMat_sqr, zeros(2*numel(ptcl_tot.x), 3*numel(ptcl_cell)), fMat_prx];
    
    d = uc.e1*uc.nei;
    [CLD,~,TLD] = srcsum(@StoDLP,d,[],uc.L,s,mu); 
    [CRD,~,TRD] = srcsum(@StoDLP,-d,[],uc.R,s,mu);
    C1 = [CRD-CLD; TRD-TLD];
    [CLD,~,TLD] = srcsum(@StoDLP,d,[],uc.L,ptcl_tot,mu); 
    [CLS,~,TLS] = srcsum(@StoSLP,d,[],uc.L,ptcl_tot,mu);
    [CRD,~,TRD] = srcsum(@StoDLP,-d,[],uc.R,ptcl_tot,mu);
    [CRS,~,TRS] = srcsum(@StoSLP,-d,[],uc.R,ptcl_tot,mu);
    C2 = [CRD-CLD + CRS-CLS; TRD-TLD + TRS-TLS];
    C = [C1 C2];
    
    [QL,~,QLt] = proxyrep(uc.L,P,mu); [QR,~,QRt] = proxyrep(uc.R,P,mu); % vel, tract
    Q = [QR-QL; QRt-QLt];
    
    E = [BCgammaMat B; intF; intT; C zeros(numel(uc.L.x)*4,3*numel(ptcl_cell)) Q];
end

% for multiple particles contained in ptcl_cell, create large matrix with
% block structure [A11 A12 ...; A21 A22 ...] 
% Necessary since SLP self eval uses fft so requires input as one enclosed
% particle.
function [U, P, T] = srcsum_ptclself_wrapper(kernel,kernel_self,trlist,ptcl_cell,mu)
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
    P = zeros(ptcl_tot_size,ptcl_tot_size*2);
    T = zeros(ptcl_tot_size*2);
    
    for i=1:num_ptcl
        ptcl_i = ptcl_cell{i};
        srcind = [ptcl_idx(i):(ptcl_idx(i+1)-1),(ptcl_tot_size+ptcl_idx(i)) : (ptcl_tot_size+ptcl_idx(i+1)-1)];
        for j=1:num_ptcl
            trgind = [ptcl_idx(j):(ptcl_idx(j+1)-1),(ptcl_tot_size+ptcl_idx(j)) : (ptcl_tot_size+ptcl_idx(j+1)-1)];
            if i==j
                % ptcl_i self eval
                if nargout == 1
                    u = srcsum(kernel_self,trlist,[],ptcl_i,ptcl_i,mu);
                elseif nargout == 2
                    [u, pres] = srcsum(kernel_self,trlist,[],ptcl_i,ptcl_i,mu);
                elseif nargout == 3
                    [u, pres, trac] = srcsum(kernel_self,trlist,[],ptcl_i,ptcl_i,mu);
                end
            else
                ptcl_j = ptcl_cell{j};
                % particle i (with nbrs) to ptcl j
                % u = srcsum(kernel,trlist,[],ptcl_j,ptcl_i,mu);
                if nargout == 1
                    u = srcsum(kernel,trlist,[],ptcl_j,ptcl_i,mu);
                elseif nargout == 2
                    [u, pres] = srcsum(kernel,trlist,[],ptcl_j,ptcl_i,mu);
                elseif nargout == 3
                    [u, pres, trac] = srcsum(kernel,trlist,[],ptcl_j,ptcl_i,mu);
                end
            end
            U(trgind,srcind) = U(trgind, srcind) + u;
            if nargout > 1
                P(trgind(1:end/2),srcind) = P(trgind(1:end/2),srcind) + pres;
            end
            if nargout > 2
                T(trgind,srcind) = T(trgind,srcind) + trac;
            end
    
        end
    end
end

function U = srcsum_ptcl_wrapper(kernel,trlist,trg,ptcl_cell,mu)
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
    U = zeros(2*numel(trg.x),ptcl_tot_size*2);
    
    for i=1:num_ptcl
        ptcl_i = ptcl_cell{i};
        srcind = [ptcl_idx(i):(ptcl_idx(i+1)-1),(ptcl_tot_size+ptcl_idx(i)) : (ptcl_tot_size+ptcl_idx(i+1)-1)];
        u = srcsum(kernel,trlist,[],trg,ptcl_i,mu);
        U(:,srcind) = u;
    end
end