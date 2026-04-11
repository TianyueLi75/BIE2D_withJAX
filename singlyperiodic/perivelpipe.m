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
    
    % N=100; % Note that 40 is too low for close-eval to be effective, should be ~300.
    % TODO: even at 300, certain values around edge still blows up.
    m=80;
    
    % set up upper and lower walls
    uc.e1 = 2*pi;   % unitcell, e1=lattice vector as a complex number

    % U.Z = @(t) t + 1i*(1+0.3*sin(t)); U.Zp = @(t) 1 + 0.3i*cos(t); U.Zpp = @(t) -0.3i*sin(t);
    % U = setupquad(U,N);   % t=0 is left end, t=2pi right end
    % U.nx = -U.nx; U.cur = -U.cur; U.xp = -U.xp; U.tang=-U.tang; U.cw=-U.cw; % correct for sense of U, opp from periodicdirpipe
    % D.Z = @(t) t + 1i*(-1+0.3*sin(t)); D.Zp = @(t) 1 + 0.3i*cos(t); D.Zpp = @(t) - 0.3i*sin(t);
    % D = setupquad(D,N);   % same dir (left to right); pipe wall normals outwards
    % s = mergesegquads([U,D]);

    % Panel based quadr on wall
    U = []; D = []; % clear structs
    U.Z = @(t) (2*pi-t) + 1i*(1+0.3*sin(2*pi-t)); U.Zp = @(t) -1 - 0.3i*cos(2*pi-t); U.Zpp = @(t) -0.3i*sin(2*pi-t);
    D.Z = @(t) t + 1i*(-1+0.3*sin(t)); D.Zp = @(t) 1 + 0.3i*cos(t); D.Zpp = @(t) - 0.3i*sin(t);
    
    % Panel based quadrature using GL grids, use axigeom functions
    Num_panels = 10;
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
    % inside = @(z) imag(z-DX(real(z)))>0 & imag(z-UX(real(z)))<0; 
    
    % Choco Dec 2025: add particle
    Nptcl = 150;
    num_ptcl = 2;
    ptcl.Z = @(t) 1 + 0.2*cos(t) + 1j*(0.3*sin(t)+0.25);
    ptcl.Zp = @(t) - 0.2*sin(t) + 1j*0.3*cos(t);
    ptcl.Zpp = @(t) - 0.2*cos(t) - 1j*0.3*sin(t);
    % ptcl.Z = @(t) 1 + 0.2*cos(t) + 1j*(0.3*sin(t)+0.25);
    % ptcl.Zp = @(t) - 0.2*sin(t) + 1j*0.3*cos(t);
    % ptcl.Zpp = @(t) - 0.2*cos(t) - 1j*0.3*sin(t);
    ptcl = setupquad(ptcl, Nptcl);
    % ptcl = wobblycurve(0.2,0.1,5,Nptcl);
    % ptcl.x = ptcl.x + 1+0.25j;
    ptcl.a = 1+0.25i; % a point in interior, use center. (needed for LapSLP close eval)
    % ptcl.nx = -ptcl.nx; ptcl.cur = -ptcl.cur; ptcl.xp = -ptcl.xp; ptcl.tang=-ptcl.tang; ptcl.cw=-ptcl.cw; % correct for sense of ptcl, opp from periodicdirpipe
    % Add geometric properties for near/far target separation (Choco Mar
    % 2026), also used in inside()..
    ptcl.xc = 1+0.25i;
    ptcl.max_r = 0.3;

    % zt = [];
    % % zt.x = ptcl.x(3:4) + 0.01 * ptcl.nx(3:4);
    % zt.x = [2+1.2j;1+0.8j];
    % zt.nx = [1+1j; 1+1j];
    % mu = 0.7;
    % [u, p, T] = StoDLP_closeglobal(zt, ptcl, mu, eye(2*numel(ptcl.x)), 'e');
    % save("./ExpectedMatrices/stodl_global.mat",'u', 'p', 'T');
    % [v, vp, vpp] = Cau_closeglobal(zt.x, ptcl, eye(numel(ptcl.x)), 'e');
    % save("./ExpectedMatrices/cau_global.mat", 'v', 'vp','vpp');
    % [v, dv1, dv2, ddv1, ddv12, ddv2] = LapSLP_closeglobal(zt, ptcl, eye(numel(ptcl.x)), 'e');
    % save("./ExpectedMatrices/lapsl_global.mat", 'v', 'dv1','dv2', 'ddv1','ddv12','ddv2');
    % [v, dv1, dv2] = LapDLP_closeglobal(zt, ptcl, eye(numel(ptcl.x)), 'e');
    % save("./ExpectedMatrices/lapdl_global.mat", 'v', 'dv1','dv2');
    % [u, p, T] = StoSLP_closeglobal(zt, ptcl, mu, eye(2*numel(ptcl.x)), 'e');
    % save("./ExpectedMatrices/stosl_global.mat", 'u', 'p','T');
    % [u,p,T] = StoDLP_closepanel(zt, s, mu, eye(2*numel(s.x)),'i');
    % save("./ExpectedMatrices/stodl_panel.mat", 'u', 'p','T');

    if num_ptcl > 1
        ptcl2.Z = @(t) 5 + 0.2*cos(t) + 1j*(0.2*sin(t)+0);
        ptcl2.Zp = @(t) - 0.2*sin(t) + 1j*0.2*cos(t);
        ptcl2.Zpp = @(t) - 0.2*cos(t) - 1j*0.2*sin(t);
        ptcl2 = setupquad(ptcl2, Nptcl);
        ptcl2.a = 5; % a point in interior, use center. (needed for LapSLP close eval)
        % ptcl2.nx = -ptcl2.nx; ptcl2.cur = -ptcl2.cur; ptcl2.xp = -ptcl2.xp; ptcl2.tang=-ptcl2.tang; ptcl2.cw=-ptcl2.cw; % correct for sense of ptcl, opp from periodicdirpipe
        ptcl2.xc = 5;
        ptcl2.max_r = 0.2;
        % ptcl2.inside = @(z) abs(z-ptcl2.xc) < ptcl2.max_r;

        ptcl_cell = {ptcl,ptcl2};
        ptcl_tot = mergesegquads([ptcl,ptcl2]);
        inside = @(z) imag(z-DX(real(z)))>0 & imag(z-UX(real(z)))<0 & abs(z-ptcl.xc)>ptcl.max_r & abs(z-ptcl2.xc)>ptcl2.max_r;
    else
        ptcl_cell = {ptcl};
        ptcl_tot = ptcl;
        inside = @(z) imag(z-DX(real(z)))>0 & imag(z-UX(real(z)))<0 & abs(z-ptcl.xc)>ptcl.max_r;
    end
    
    zt.x = [2+0.2i; 4+0.1i];    % point to test u soln at (expt='t' only)
    % zt.x = [(U.x(16) -0.005*U.nx(16)); (D.x(24)-0.005*D.nx(24))]; % 2nd panel from the right on U, 3rd panel from left on D
    
    % set up left and right walls
    uc.nei = 1; % how many nei copies either side (use eg 1e3 to test A w/o AP)
    uc.trlist = uc.e1*(-uc.nei:uc.nei);  % list of translations for direct images
    [x w] = gauss(m); x = (1+x)/2; w = w'/2; % quadr on [0,1]
    % Note: assumes U.x(end) and D.x(0) match at same x..
    H = U.x(end)-D.x(1); L.x = D.Z(0) + H*x; L.nx = 0*L.x+1; L.w = H*w; % left side
    R = L; R.x = L.x+uc.e1; % right side
    uc.L = L; uc.R = R;
    
    % set up aux periodizing basis
    P = [];
    proxyrep = @StoSLP;      % sets proxy pt type via a kernel function call
    Rp = 1.1*2*pi; 
    M = 2*m;    % # proxy pts (2 force comps per pt, so 2M dofs)
    P.x = pi + Rp*exp(2i*pi*(0:M-1)'/M); P = setupquad(P);     % proxy pts
    % if v>1, figure; showsegment({U,D,ptcl_tot},uc.trlist); showsegment({L,R}); plot(P.x,'r+'); plot(zt.x,'go'); 
        % plot(ptcl.x);  
    % end
    
    mu = 0.7;                                           % fluid viscosity
    if expt=='t' % Exact soln: either periodic or plus fixed pressure drop / period:
      % ue = @(x) [1+0*x; -2+0*x]; pe = @(x) 0*x; % the exact soln: uniform rigid flow, constant pressure everywhere (no drop)
      h=.2; ue = @(x) h*[imag(x).^2;0*x]; pe = @(x) h*2*mu*real(x); % horiz Poisseuil flow (has pres drop)
      % ue = @(x) [1+0*x;0.5+0*x]; pe = @(x) 0*x;
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
    [E,A,B,C,Q] = ELSmatrix_multi(s,ptcl_cell,P,proxyrep,mu,uc); 
    save("mat_temp.mat",'A','B','C','Q');
    
    
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
    % psi = co(1+2*numel(s.x):end);
    % fprintf("density norm = %.3g, proxy norm = %.3g\n", norm(sig), norm(psi));
    sig_ptcl = co(1+2*numel(s.x):2*numel(s.x)+2*numel(ptcl_tot.x));
    psi = co(2*numel(s.x)+2*numel(ptcl_tot.x)+1:end);
    fprintf('density norm = %.3g, ptcl denstiy norm = %.3g, proxy norm = %.3g\n',norm(sig)/numel(sig), norm(sig_ptcl)/numel(sig_ptcl), norm(psi)/numel(psi))
    [ut, pt] = evalsol_sldl_near(s,ptcl_tot,ptcl_cell,P,proxyrep,mu,uc,zt.x,co);
    % [ut, pt] = evalsol_sldl(s,ptcl_tot,P,proxyrep,mu,uc,zt.x,co);
    % [ut,pt] = evalsol(s,P,proxyrep,mu,uc,zt.x,co);
    
    format longE
    % ut
    if expt=='t'                         % check vs known soln at the point zt
        fprintf('u velocity err at zt = %.3g \n', norm(ut-ue(zt.x)))
        fprintf('perr at zt: ')
        pt-pe(zt.x)
    else
        % fprintf('u velocity at zt = [%.15g, %.15g]; p at zt = %.15g \n', ut(1),ut(2),pt);
        fprintf('u velocity at zt: ')
        ut
        fprintf('p at zt: ')
        pt
    end

    % % Compare all near vs far evals, at these targets they should be the same.
    % fprintf("u difference closeglobal vs regular: %.3g \n", norm(ut_near - ut));
    % fprintf("p difference closeglobal vs regular: %.3g \n", norm(pt_near - pt));
    
    
    if v   % plots
        nx = 160; gx = 2*pi*((1:nx)-0.5)/nx; ny = nx; gy = gx - pi; % plotting grid
        % gy = 6.0/5.0 * (((1:ny)-0.5)/ny - 0.5); % Choco Jan2026: to avoid
        % near eval errors and visualize actual interior to sin pile.
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
        % showsegment({U,D,ptcl_tot},uc.trlist); showsegment({L,R}); plot(P.x,'r+'); plot(zt.x,'go'); 
        % text(-.5,0,'L');text(2*pi+.5,0,'R');text(4,-1,'D');text(pi,0.5,'U');
        % title('geom'); if expt=='t',title('geom and (u,p) known soln'); end
        % eval and plot soln...
        ug = nan(size([t.x;t.x])); pg = nan(size(t.x)); ii = inside(t.x);
        
        % [ug([ii;ii]) pg(ii)] = evalsol_sldl(s,ptcl_tot,P,proxyrep,mu,uc,t.x(ii),co);
        [ug([ii;ii]) pg(ii)] = evalsol_sldl_near(s,ptcl_tot,ptcl_cell,P,proxyrep,mu,uc,t.x(ii),co);
        
        u1 = reshape(ug(1:Mt),size(xx)); u2 = reshape(ug(Mt+(1:Mt)),size(xx));
        pp = reshape(pg,size(xx)); pp = pp - pp(ceil(ny/2),1); % zero p mid left edge
        figure; 
        % imagesc(gx,gy, pp); colormap(jet(256));
        % caxis(sort([0 jump])); colorbar;
        % hold on; 
        magvals = sqrt(u1.^2 + u2.^2);
        imagesc(gx,gy,log10(magvals)); % colormap(jet(256)); 
        colorbar; hold on; % Choco jan2026: plot magnitudes to see if no-slip was respected.
        quiver(gx,gy, u1,u2); 
        [startX, startY] = meshgrid(gx(1:10:end), gy(1:10:end));
        verts = stream2(gx,gy,u1,u2,startX,startY);
        streamline(verts); 

        title('soln (u,p), with close-eval scheme')
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

            eg3 = sum([u1(ii)-ue1(ii),u2(ii)-ue2(ii)].^2,2);
            norm(eg3)
        end
    end
    % %}
end
    % keyboard % don't forget to use dbquit to finish otherwise trapped in debug mode
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function [u, p] = evalsol(s,pr,proxyrep,mu,U,z,co,close) % eval soln rep u,p
    % z = list of targets as C values. pr = proxy struct, s = source curve struct
    % co = full coeff vec, U = unit cell struct, mu = viscosity. DLP only.
    % Note: not for self-eval since srcsum2 used, 6/30/16. Taken from fig_stoconvK1.
    % TODO: implement close eval.
    if nargin<8
        close=0; 
    end              % default is plain native quadr
    if close
        error('close not implemented for open segments!'); 
    end
    z = struct('x',z);                     % make targets z a segment struct
    N = numel(s.x);
    sig = co(1:2*N); psi = co(2*N+1:end);  % split into sig (density) & psi (proxy)
    if close
      % D = @(t,s,mu,dens) StoDLP_closeglobal(t,s,mu,dens,'e'); % exterior
      D = @(t,s,mu,dens) StoDLP_closeglobal(t,s,mu,dens,'e');
    % else, D = @StoDLP; end                 % NB now native & close have same 4 args
    else
        D = @StoDLP; 
    end
    if nargout==1                          % don't want pressure output
      u = proxyrep(z,pr,mu,psi);           % init sol w/ proxies (always far)
      u = u + srcsum2(D,U.trlist,[],z,s,mu,sig);
    else  
      [u p] = proxyrep(z,pr,mu,psi);       % init sol w/ proxies (always far)
      [uD pD] = srcsum2(D,U.trlist,[],z,s,mu,sig);
      u = u + uD;
      p = p + pD;
    end
end
    
% Choco Jan2026: DL+SL formulation, with same density
function [u, p] = evalsol_sldl(s,ptcl,pr,proxyrep,mu,U,z,co) % eval soln rep u,p
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
end
    
    
function [u, p] = evalsol_sldl_near(s,ptcl_tot,ptcl_cell,pr,proxyrep,mu,U,z,co) % eval soln rep u,p

    z = struct('x',z);                     % make targets z a segment struct
    N = numel(s.x);
    sig = co(1:2*N);
    sig_ptcl_tot = co(2*N + (1:2*numel(ptcl_tot.x)));
    psi = co(2*N+2*numel(ptcl_tot.x)+1:end);  
    if nargout==1                          % don't want pressure output
        u = proxyrep(z,pr,mu,psi);           % init sol w/ proxies (always far)
        u = u + srcsum(@StoDLP_closepanel,U.trlist,[],z,s,mu,sig); 
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
end

% Choco Jan 2026: dl+sl formulation; multi particle support
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

    A12 = srcsum_ptcl_wrapper(@StoDLP_closeglobal,uc.trlist,s,ptcl_cell,mu) + srcsum_ptcl_wrapper(@StoSLP_closeglobal,uc.trlist,s,ptcl_cell,mu); % all particle to wall
    A21 = srcsum(@StoDLP_closepanel,uc.trlist,[],ptcl_tot,s,mu); % wall to ptcl
    A22 = eye(2*numel(ptcl_tot.x))/2 + srcsum_ptclself_wrapper(@StoDLP_closeglobal,@StoDLP,uc.trlist,ptcl_cell,mu) + srcsum_ptclself_wrapper(@StoSLP_closeglobal,@StoSLP,uc.trlist,ptcl_cell,mu);

    % A12 = srcsum_ptcl_wrapper(@StoDLP,uc.trlist,s,ptcl_cell,mu) + srcsum_ptcl_wrapper(@StoSLP,uc.trlist,s,ptcl_cell,mu); % all particle to wall
    % A21 = srcsum(@StoDLP,uc.trlist,[],ptcl_tot,s,mu); % wall to ptcl
    % A22 = eye(2*numel(ptcl_tot.x))/2 + srcsum_ptclself_wrapper(@StoDLP,@StoDLP,uc.trlist,ptcl_cell,mu) + srcsum_ptclself_wrapper(@StoSLP,@StoSLP,uc.trlist,ptcl_cell,mu);
    
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
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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