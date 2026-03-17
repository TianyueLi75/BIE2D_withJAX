function [A, P, T] = StoDLP_closepanel(t, s, mu, sigma, side)
% TODO: also need to return pressure close eval..
    if nargin==0
        % test_StoDLP_closepanel;
        test_StoDLP_closepanel_near;
        return;
    end

    if nargin < 5
        side = 'i'; % since intended to be used with channels, problem likely on the interior.
    end
    if nargin < 4
        sigma = [];
    end

    N=size(s.x,1); M=size(t.x,1);       % # srcs, # targs
    mat = isempty(sigma);
    if mat, sigma=eye(2*N); end         % case of dense matrix
    sigma = sigma(1:N,:)+1i*sigma(N+1:end,:);  % put into complex notation, N x 2N
    Nc = size(sigma,2); 

    % Below taken from AxiStokesSpecialMat... by Hai, AxiStokesPeriodicMat
    % Amat will be size Ntrg x 1 if density given (assuming vector density
    % not matrix), Ntrg x Nsrc if not.
    Amat = zeros(numel(t.x),Nc);
    P = zeros(numel(t.x),Nc);
    Tmat = zeros(numel(t.x),Nc);
    for k=1:s.np
        % kth panel...
        pidx = (k-1)*s.p+(1:s.p);
        sk = []; 
        sk.p = s.p; 
        % sk.tlo = s.tlo(k); 
        % sk.thi = s.thi(k);
        sk.np = 1; 
        sk.xlo = s.xlo(k); 
        sk.xhi = s.xhi(k); 
        sk.w = s.w(pidx);
        sk.x = s.x(pidx); 
        % sk.xp = s.xp(pidx); 
        % sk.xpp = s.xpp(pidx); 
        % sk.sp = s.sp(pidx); 
        % sk.tang = s.tang(pidx); 
        sk.nx = s.nx(pidx); 
        sk.cur = s.cur(pidx); 
        sk.ws = s.ws(pidx);
        % sk.t = s.t(pidx); 
        % sk.wxp = s.wxp(pidx); 
        sk.cw = s.cw(pidx);

        sigma_k = sigma(pidx,:); % p x Nc columns of density values corresponding to this panel nodes

        % close target index...
        panlen = sum(sk.ws); c=2; 
        ik = (abs(t.x - sk.xlo) + abs(t.x - sk.xhi)) < c*panlen;
        % ik = (abs(t.x - sk.xlo) + abs(t.x - sk.xhi)) >-100; % make all near
        tk = []; tk.x = t.x(ik(:)); 
        
        % build interaction matrix for close targets...
        if ~isempty(tk.x)
            % fprintf("Number of near targets on panel %d is %d\n", k, numel(tk.x));
            [Ak, L1, L2, ~, ~] = Dspecialquad(tk, sk, sk.xlo, sk.xhi, side); 

            tauf = bsxfun(@times, sigma_k, real(sk.nx)./(sk.nx));    % undo n_y rotation
            I1x1 = real(Ak*(tauf));
            tauf = bsxfun(@times, sigma_k, imag(sk.nx)./(sk.nx));    % undo n_y rotation
            I1x2 = real(Ak*(tauf));
            I1 = I1x1+1i*I1x2;
            
            % find I_2
            tau = real(bsxfun(@times,sk.x,conj(sigma_k)));
            I2x1 = L1 * tau;
            I2x2 = L2 * tau;
            I2 = I2x1+1i*I2x2;
            
            % find I_3 and I_4
            I3x1 = L1 * real(sigma_k);
            I3x2 = L2 * real(sigma_k);
            I4x1 = L1 * imag(sigma_k);
            I4x2 = L2 * imag(sigma_k);
            I3 = bsxfun(@times, real(tk.x), I3x1+1i*I3x2);
            I4 = bsxfun(@times, imag(tk.x), I4x1+1i*I4x2);

            u = I1+I2-I3-I4;
            Ak = u; % complex, Nclose x Nc
        end
    
        % naive implementation for far targets interaction...
        tfar = []; tfar.x = t.x(~ik(:)); 
        if ~isempty(tfar.x)
            if nargout > 2
                if ~isfield(t,'nx')
                    error("\n Need target normal for DLP traction.");
                end
                tfar.nx = t.nx(~ik(:));
                [Afar,Pfar,Tfar] = StoDLP(tfar,sk,mu, [real(sigma_k); imag(sigma_k)]); 
            elseif nargout > 1
                [Afar,Pfar] = StoDLP(tfar,sk,mu, [real(sigma_k); imag(sigma_k)]);
            else
                Afar = StoDLP(tfar,sk,mu, [real(sigma_k); imag(sigma_k)]);
            end
        end

        if ~isempty(tk.x)
            Amat(ik(:),:) = Amat(ik(:),:) + Ak;
        end
        if ~isempty(tfar.x)
            Amat(~(ik(:)),:) = Amat(~(ik(:)),:) + Afar(1:end/2,:) + 1j * Afar(1+end/2 : end, :);
        end

        if nargout > 1
            if ~isempty(tk.x)
                P(ik(:),:) = P(ik(:),:) - 2*mu*(I3x1 + I4x2);
            end
            if ~isempty(tfar.x)
                P(~ik(:),:) = P(~(ik(:)),:) + Pfar;
            end
        end

        if nargout > 2
            if ~isempty(tk.x)
                tk.nx = t.nx(ik(:));
                Tcorr = mu*stokespanelcorm(tk, sk, sk.xlo, sk.xhi, side);
                Tmat(ik(:),:) = Tmat(ik(:),:) + Tcorr * [real(sigma_k);imag(sigma_k)];
            end
            if ~isempty(tfar.x)
                Tmat(~ik(:),:) = Tmat(~ik(:),:) + (Tfar(1:end/2,:) + 1j*Tfar(1+end/2:end,:));
            end

        end
    end
    
    A = [real(Amat);imag(Amat)];
    T = [real(Tmat);imag(Tmat)];

end


function test_StoDLP_closepanel_near
    fprintf('tests on StoDLP near eval\n')

    % U.Z = @(t) t + 1j;
    % U.Zp = @(t) 1 + 0.*t;
    % U.Zpp = @(t) 0.*t;
    U.Z = @(t) 2*pi - t + 1j;
    U.Zp = @(t) -1 + 0.*t;
    U.Zpp = @(t) 0.*t;

    mu = 0.9;       % viscosity (real, pos)
    side = 'i';

    % Set source density
    tau = @(t) [0.7+sin(t); -0.4+cos(t)];  % pick smooth density w/ nonzero mean
    % tau = @(t) [zeros(numel(t),1); cos(t)];
    % tau = @(t) [ones(numel(t),1); zeros(numel(t),1)]; % simple (1,0) density
    
    % Panel based quadrature using GL grids, use axigeom functions
    Num_panels = 20;
    p = 10; % order on panel
    N_perwall = p * Num_panels; qtype = 'p'; qntype = 'G'; U.p = p;
    [U,~] = quadr(U, N_perwall, qtype, qntype); 
    U.trlist = [-2*pi,2*pi];
    U.tpan = [U.tlo;U.thi(end)];
    U.cw = U.wxp; 

    t.x = [1 + 0.65i; 4+0.7i];
    t.nx = [1+1i; 1+1i];

    [uc, pc, Tc] = StoDLP_closepanel(t,U,mu,tau(U.t),side);


    % Compute using higher discretization from naive setting
    np_dense = 40;
    N_perwall_dense = p * np_dense; 
    U_dense = [];
    U_dense.p = p; 
    U_dense.Z = U.Z; U_dense.Zp = U.Zp; U_dense.Zpp = U.Zpp;
    [U_dense,~] = quadr(U_dense, N_perwall_dense, qtype, qntype); 
    U_dense.trlist = [-2*pi,2*pi];
    U_dense.tpan = [U_dense.tlo;U_dense.thi(end)];
    U_dense.cw = U_dense.wxp; 
    % U_dense.nx = -U_dense.nx; U_dense.cur = -U_dense.cur; U_dense.xp = -U_dense.xp; U_dense.tang=-U_dense.tang; U_dense.cw=-U_dense.cw; % correct for sense of U, opp from periodicdirpipe

    [u, p, T] = StoDLP(t,U_dense,mu,tau(U_dense.t));    % eval given density cases...

    fprintf('Sto DLP density eval, max abs err in u cmpts, p, T:\n')
    disp([max(abs(u-uc)), max(abs(p-pc)), max(abs(T-Tc))]);

    % Ac = StoDLP_closepanel(t,s,mu,[],side);
    % A = StoDLP(t,s,mu);   % compare matrix els...
    % fprintf('Compare matrices: \n')
    % disp(max(abs(A(:)-Ac(:))))

    return;
end

function Acorr = stokespanelcorm(tx, sx, a, b, side)
    % STOKESPANELCOR - Panel correction matrix for Stokes potentials
    %
    % Acorr = stokespanelcor(tx, rx, a, b, lptype, side) gives special
    % quadrature value matrix for panel correction.
    % Inputs: tx = target node, with or without struct both will work.
    %         sx = source node, with or without struct both will work.
    %         a = panel start, b = panel end, in complex plane.
    %         side = interior, 'i', or exterior, 'e'.
    % Output: Acorr is special quadrature value at target node px.
    %         Note: Acorr maps density vals (be*p) to target velocities.
    % Efficient only if multiple targs, since O(p^3).
    % See Helsing-Ojala 2008 (special quadr Sec 5.1-2), Helsing 2009 mixed (p=16),
    % and Helsing's tutorial demo11b.m LGIcompRecFAS()
    % Hai 08/28/16, modified 11/29/18, use sf to avoid iterpolation multiple times
    % AHB moved sub-funcs to separate files, removed old code. 4/4/19.
    
    be=2; % upsampling rate

    t = []; if isstruct(tx), t = tx; else, t.x = tx; end     % form target nodes struct
    s = []; if isstruct(sx), s = sx; else, s.x = sx; end     % form source nodes struct
    % get struct for s with geometry info
    if ~isfield(s,'nx')
        error("\n Need s.nx for traction.");
    end   
    sf = quadr_panf(s, be, 'G'); % struct with geometry info at fine nodes for close eval
    num = numel(s.x); 
    Imn = interpmat(num, ceil(num*be), 'G');     % interpolation matrix
    % double-layer traction kernel
    % Hai 11/11/2018
    % use Az, Azz from Laplace DLP, assemble each splitted term
    [~, Az, Azz] = Dspecialquad(t,sf,a,b,side);

    hx11   = -4*real(Azz).*real((t.x-sf.x.').*(conj(t.nx)*ones(1,numel(sf.x))));  % (s.ws(testn)*(-Hx)/pi)
    dxr11  =  real(Az).*(real(t.nx)*ones(1,numel(sf.x)));   % (Dxr*s.ws(testn)/(2*pi).*nx(:,1))
    dxi11  = -3*imag(Az).*(imag(t.nx)*ones(1,numel(sf.x)));    % (-3*Dxi*s.ws(testn)/(2*pi).*nx(:,2))
    dxrc11 =  real(Az.*(ones(numel(t.x),1)*(conj(sf.nx)./sf.nx).')).*(real(t.nx)*ones(1,numel(sf.x)));    % (Dxrc*s.ws(testn)/(2*pi).*nx(:,1))
    dxic11 = -imag(Az.*(ones(numel(t.x),1)*(conj(sf.nx)./sf.nx).')).*(imag(t.nx)*ones(1,numel(sf.x)));    % (-(Dxic*s.ws(testn)/(2*pi).*nx(:,2)))

    T11 = hx11+dxr11+dxi11+dxrc11+dxic11;

    hx12   =  4*imag(Azz).*real((t.x-sf.x.').*(conj(t.nx)*ones(1,numel(sf.x))));
    dxr12  = -real(Az).*(imag(t.nx)*ones(1,numel(sf.x)));
    dxi12  =  imag(Az).*(real(t.nx)*ones(1,numel(sf.x))); 
    dxrc12 = -real(Az.*(ones(numel(t.x),1)*(conj(sf.nx)./sf.nx).')).*(imag(t.nx)*ones(1,numel(sf.x)));
    dxic12 = -imag(Az.*(ones(numel(t.x),1)*(conj(sf.nx)./sf.nx).')).*(real(t.nx)*ones(1,numel(sf.x)));

    T12 = hx12+dxr12+dxi12+dxrc12+dxic12;


    hx22   =  4*real(Azz).*real((t.x-sf.x.').*(conj(t.nx)*ones(1,numel(sf.x))));
    dxr22  =  3*real(Az).*(real(t.nx)*ones(1,numel(sf.x)));
    dxi22  = -imag(Az).*(imag(t.nx)*ones(1,numel(sf.x)));
    dxrc22 = -real(Az.*(ones(numel(t.x),1)*(conj(sf.nx)./sf.nx).')).*(real(t.nx)*ones(1,numel(sf.x)));
    dxic22 =  imag(Az.*(ones(numel(t.x),1)*(conj(sf.nx)./sf.nx).')).*(imag(t.nx)*ones(1,numel(sf.x)));

    T22 = hx22+dxr22+dxi22+dxrc22+dxic22;

    Acorr = [T11*Imn,T12*Imn]+1i*[T12*Imn,T22*Imn];
        
end

function sf = quadr_panf(s, be, qntype)  
    % set up quadrature on a closed segment
    % QUADR_panf - set up quadrature (either coarse or fine nodes) on a segment struct
    %
    % sf = quadr_panf(s, be) gives quadrature on coarse or fine nodes.
    % Inputs: s  = segment struct containing parametrization
    %         be = factor by which to increase panel nodes
    %  
    % Outputs: sf - segment struct on fine node
    if be == 1
        sf = s;
        if ~isfield(s,'p')
        s.p=16; 
        end
        p = s.p; % default panel order
        if qntype=='G', [~, w, D] = gauss_here(p); else, [~, w, D] = cheby(p); end 
        sf.xp = D*sf.x;
        sf.xpp = D*sf.xp;   % acceleration Z''(sf.x)
        sf.w = w;
        sf.sp = abs(sf.xp); sf.tang = sf.xp./sf.sp; sf.nx = -1i*sf.tang;    % outward unit norma
    else
        if ~isfield(s,'p')
            s.p=16; 
        end
        p = s.p; % default panel order
        sf=[]; sf.p=ceil(be*s.p); pf=sf.p;
        Imn = interpmat(p, pf, qntype);
        sf.x = Imn*s.x;
        if qntype=='G', [xx, w, D] = gauss_here(pf); else, [xx, w, D] = cheby(pf); end 
        if ~isfield(s,'Zp') 
            if ~isfield(s,'xp'), sf.xp = D*sf.x;  else, sf.xp = Imn*s.xp*(s.thi-s.tlo)/2; end  % velocities Z'(sf.x)
        else
            sf.xp = 1/2*(s.thi-s.tlo)*s.Zp(s.tlo + (1+xx)/2*(s.thi-s.tlo));
        end
        if ~isfield(s,'Zpp')
            if ~isfield(s,'xpp'), sf.xpp = D*sf.xp;  else, sf.xpp = Imn*s.xpp*(s.thi-s.tlo)/2; end  % acceleration Z''(sf.x)
        else
            sf.xpp = 1/2*(s.thi-s.tlo)*s.Zpp(s.tlo + (1+xx)/2*(s.thi-s.tlo));
        end
        sf.w = w;
        sf.sp = abs(sf.xp); sf.tang = sf.xp./sf.sp; sf.nx = -1i*sf.tang;    % outward unit normals
        sf.cur = -real(conj(sf.xpp).*sf.nx)./sf.sp.^2;
        sf.ws = sf.w.*sf.sp; % speed weights
        sf.cw = sf.w.*sf.xp; % complex speed weights (Helsing's wzp)
    end
end

function P = interpmat(n,m, qntype) % interpolation matrix from n-pt to m-pt Gauss nodes
    % INTERPMAT - create interpolation matrix from n-pt to m-pt Gauss nodes
    %
    % P = interpmat(n,m) returns a m*n matrix which maps func values on n-pt Gauss-
    % Legendre nodes on [-1,1] to values on m-pt nodes.
    % Does it the Helsing way via backwards-stable ill-cond Vandermonde solve.
    if m==n
        P = eye(n); 
        return
    end
    if qntype=='G'
        x = gauss_here(n); 
        y = gauss_here(m); 
    else
        x = cheby(n); 
        y = cheby(m); 
    end 
    V = ones(n); 
    for j=2:n
        V(:,j) = V(:,j-1).*x; 
    end % Vandermonde, original nodes
    R = ones(m,n); 
    for j=2:n
        R(:,j) = R(:,j-1).*y; 
    end % monomial eval matrix @ y
    P = (V'\R')';                                       % backwards-stable solve
end

function [x, w, D] = gauss_here(N)  % Gauss-Legendre nodes and weights on [-1,1]
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