% Self convergence for solver with close eval in eval stage to test 
% ... 1) solve with increasing discretization degrees
% ... 2) eval convergence
% ... 3) close eval convergences.

% distances = 10.^(-1:-1:-5);
distances = [0.00001]; % first check self conv for far target.
Nord_lst = 2.^(6:9);
p = 10;
Nlr = 140;
Nprx = 2*Nlr;
mu = 0.7;
qtype = 'p'; qntype = 'G';
jump = -1; % pressure jump

UX = @(x) x + 1i*(1+0.3*sin(x));
DX = @(x) x + 1i*(-1+0.3*sin(x));
inside = @(z) imag(z-DX(real(z)))>0 & imag(z-UX(real(z)))<0; 

% set up left and right walls
uc.e1 = 2*pi;
uc.nei = 1; % how many nei copies either side (use eg 1e3 to test A w/o AP)
uc.trlist = uc.e1*(-uc.nei:uc.nei);  % list of translations for direct images
[x, w] = gauss_here(Nlr); % GL grid..
x = (1+x)/2; w = w'/2; % quadr on [0,1]
H = UX(0)-DX(0); L.x = DX(0) + H*x; L.nx = 0*L.x+1; L.w = H*w; % left side
R = L; R.x = L.x+uc.e1; % right side
uc.L = L; uc.R = R;
% B.C. on R-L
Tjump = -jump * [real(R.nx);imag(R.nx)]; % traction driving growth (vector func)

% set up aux periodizing basis
proxyrep = @StoSLP;      % sets proxy pt type via a kernel function call
Rp = 1.1*2*pi; 
P.x = pi + Rp*exp(2i*pi*(0:Nprx-1)'/Nprx); P = setupquad(P);     % proxy pts

% Allocate array for errors
erru = zeros(numel(Nord_lst),numel(distances));
errp = zeros(numel(Nord_lst),numel(distances));
ut_exact_lst = zeros(numel(distances),1); % store max order ut and pt values for self convergence
pt_exact_lst = zeros(numel(distances),1);
for np_ind = numel(Nord_lst):-1:1
    Nord = Nord_lst(np_ind);
    N_perwall = p * Nord;  

    U = []; D = []; % clear structs
    U.Z = @(t) (2*pi-t) + 1i*(1+0.3*sin(2*pi-t)); U.Zp = @(t) -1 - 0.3i*cos(2*pi-t); U.Zpp = @(t) -0.3i*sin(2*pi-t);
    D.Z = @(t) t + 1i*(-1+0.3*sin(t)); D.Zp = @(t) 1 + 0.3i*cos(t); D.Zpp = @(t) - 0.3i*sin(t);
    % Panel based quadrature using GL grids, use axigeom functions
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

    h=.2; ue = @(x) h*[imag(x).^2;0*x]; pe = @(x) h*2*mu*real(x);
    vrhs = ue(s.x); 
    jump = pe(uc.e1)-pe(0); 
    Tjump = -jump * [real(R.nx);imag(R.nx)];
    % vrhs = zeros(2*numel(s.x),1);     % no-slip BCs, ie homog vel data on U,D
    
    [E,A,B,C,Q] = ELSmatrix(s,P,proxyrep,mu,uc); 
    erhs = [vrhs; zeros(2*Nlr,1);Tjump];       % expanded lin sys RHS
    dens = E \ erhs;
    fprintf('resid norm = %.3g\n',norm(E*dens - erhs))
    sig = dens(1:2*numel(s.x)); 
    psi = dens(1+2*numel(s.x):end);
    fprintf('density norm = %.3g, proxy norm = %.3g\n',norm(sig)/numel(sig), norm(psi)/numel(psi));
    
    for dist_ind = 1:numel(distances)
        near_dist = distances(dist_ind);
        fprintf("distance of target: %f\n", near_dist);
        % zt.x = [U.Z(pi) - near_dist.*U.Zp(pi);
        %     D.Z(pi) - near_dist.*D.Zp(pi)]; % a set of targets near the wall
        nx_temp = [imag(U.Zp(pi));-real(U.Zp(pi))]; % normal from tangent
        nx_normed = nx_temp / sqrt(nx_temp(1)^2 + nx_temp(2)^2);
        nx_normed = nx_normed(1) + 1j*nx_normed(2);
        zt.x = [U.Z(pi) - near_dist.*nx_normed];

        [ut, pt] = evalsol(s,P,proxyrep,mu,uc,zt.x,dens,1);
    
        % if np_ind == numel(Nord_lst)
        %     ut_exact_lst(dist_ind) = ut(1:end/2) + 1j*ut(1+end/2:end);
        %     pt_exact_lst(dist_ind) = pt;
        % else
        %     erru(np_ind,dist_ind) = max(abs(ut_exact_lst(dist_ind) - (ut(1:end/2)+1j*ut(1+end/2:end))));
        %     % TODO: error p subtract by mean first. -- can't do when only
        %     % one target point..
        %     % errp(np_ind,dist_ind) = max(abs(pt_exact_lst(dist_ind) - pt));
        % end
        erru(np_ind,dist_ind) = max(abs(ut - ue(zt.x)));
    end

end

figure; 
% subplot(1,2,1);
hold on;
legend_lst = [];
for dist_ind=1:numel(distances)
    plot(Nord_lst, log10(erru(:,dist_ind)),"*-");
    legend_lst = [legend_lst; strcat("1e-",num2str(dist_ind))];
end
legend(legend_lst); hold off;
title("close eval at different distances, self convergence by Npanels, mag(u)");

% subplot(1,2,2);
% for dist_ind=1:numel(distances)
%     plot(Nord_lst, log10(errp(:,dist_ind)),"*-");
% end
% legend(legend_lst); hold off;
% title("close eval at different distances, self convergence by Npanels, p");


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
