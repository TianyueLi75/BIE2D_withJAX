function [s, N, np] = quadr(s, N, qtype, qntype)  % set up quadrature on a closed segment
    % QUADR - set up quadrature (either global or panel-based) on a segment struct
    %
    % [s N] = quadr(s, N, qtype, qntype) adds quadrature input to segment struct s.
    % Inputs:
    %  s - segment struct containing parametrization
    %  s.Z - complex function, parameterization of the segment on [0,2\pi]
    %  s.T (optional) - reparameterization function mapping [0,2\pi] to itself.
    %  N - requested number of nodes
    %  qtype - quadrature type: 'g' global periodic trapezoid rule
    %                           'p' panel-wise quadr w/ s.p nodes per pan
    %  qntype - type of panel quadr: 'G' Gause-Legendre
    %                                'C' Cheyshev (Fejer's 1st rule)
    %
    % Outputs: s - amended segment struct
    %          N - actual number of nodes (is within s.p of requested N)
    %          np - number of panels
    % Notes: 1) Note sign change in normal sense vs periodicdirpipe.m
    % 2) Non-adaptive for now.  Barnett 6/10/13
    % - Added reparameterized panel. Bowei Wu 3/22/18
    
    if nargin < 3
        qtype = 'g'; 
    elseif nargin < 4 && qtype=='p'
        qntype = 'G'; 
    end
    
    if qtype=='g' % global (ptr) quadr
        if isfield(s,'Z') && ~isempty(N) % quadr nodes
            t = (1:N)'/N*2*pi;
            s.x = s.Z(t);
            s.tlo = 0; s.thi = 2*pi; s.p = N; s.w = 2*pi/N*ones(N,1); np=1; % 1 big panel
            s.xlo = s.Z(s.tlo); s.xhi = s.Z(s.thi);
        elseif isfield(s,'x')
            s.x = s.x(:);
        else
            error('Need to provide at least s.Z and N, or s.x. Neither found!');
        end
    elseif qtype=='p'
        if ~isfield(s,'Z'), error('Need to provide s.Z to build panels!'); end 
        if ~isfield(s,'p'), s.p=16; end       % default panel order
        p = s.p;
        np = 1;          % default, to prevent failure of line s.np=np below 4/4/19
        if isfield(s,'tpan')   % adaptive panel, See adaptive_panel.m
            s.tlo = s.tpan(1:end-1);
            s.thi = s.tpan(2:end);
            np = numel(s.tlo);
            N = np*p;
        elseif ~isfield(s,'tlo') || ~isfield(s,'thi') % if panels are not given
          if isempty(N), error('Need to provide N (approx num of pts) to build panels'), end
            np = ceil(N/p); N = p*np;      % np = # panels
            s.tlo = (0:np-1)'/np*2*pi; % panel start params
            s.thi = (1:np)'/np*2*pi; % panel end params
            if isfield(s,'T')   % adaptive panel via reparameterization, See reparam_mixed.m
                s.tlo = s.T(s.tlo);
                s.thi = s.T(s.thi);
            end
        end
        s.np = np;
        s.xlo = s.Z(s.tlo); % panel start locs
        s.xhi = s.Z(s.thi);  % panel end locs
        pt = s.thi - s.tlo;                  % panel size in parameter
        t = zeros(N,1); s.w = t;
        if qntype=='G', [x, w, D] = gauss_here(p); else, [x, w, D] = cheby(p); end  
        for i=1:np
            ii = (i-1)*p+(1:p); % indices of this panel
            t(ii) = s.tlo(i) + (1+x)/2*pt(i); s.w(ii) = w*pt(i)/2; % nodes weights this panel
        end
        s.x = s.Z(t); % quadr nodes
    end
    
    if N~=length(s.x), N = length(s.x); warning('N has changed!'); end 
    s.xp = zeros(length(s.x),1);
    s.xpp = zeros(length(s.x),1);
    
    if isfield(s,'Zp'), s.xp = s.Zp(t); % 1st dir of curve
    else
        if qtype == 'p'
            for i=1:np
                ii = (i-1)*p+(1:p); % indices of this panel
                s.xp(ii) = D*s.x(ii)*2/pt(i);
            end
        else
            s.xp = perispecdiff(s.x); % fourier spectral diff
        end
    end
    if isfield(s,'Zpp'), s.xpp = s.Zpp(t); % 2nd dir of curve
    else
        if qtype == 'p'
        for i=1:np
            ii = (i-1)*p+(1:p); % indices of this panel
            s.xpp(ii) = D*s.xp(ii)*2/pt(i);
        end
        else
            s.xpp = perispecdiff(s.xp); % fourier spectral diff
        end
    end
    
    s.sp = abs(s.xp); s.tang = s.xp./s.sp; s.nx = -1i*s.tang; % speed, tangent, normal
    s.cur = -real(conj(s.xpp).*s.nx)./s.sp.^2; % curvature
    s.ws = s.w.*s.sp; % speed weights
    s.t = t; s.wxp = s.w.*s.xp; % complex speed weights (Helsing's wzp)
    % ChocoMar2026: rename s.ws to s.w to be consistent with BIE2D
    % conventions
    s.w = s.ws;
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

function [x, w, D] = cheby(N)
    % based on chebyshev extreme cheb.m from spectral methods in matlab
    % Chebyshev nodes, weights, and spectral differentiation matrix on [-1,1]
    % 08/31/2016 Hai
    
    % chebyshev nodes
    theta = pi*(2*(1:N)'-1)/(2*N);
    x = -cos(theta);
    
    % chebyshev weights
    l = floor(N/2)+1;
    K = 0:N-l;   
    v = [2*exp(1i*pi*K/N)./(1-4*K.^2)  zeros(1,l)];
    w = real(ifft(v(1:N) + conj(v(N+1:-1:2))))';
    % j = 1:floor(N/2);
    % w = 2/N*(1 - 2*sum( cos( repmat(j,N,1).*repmat((2*(0:N-1)'+1)*pi/N,1,floor(N/2)))...
    %     .* (1./repmat(4*j.^2-1,N,1)), 2));
    
    % spectral differentiation matrix
    X = repmat(x,1,N);
    dX = X-X';  % x_i-x_j
    a = prod(dX+eye(N),2);  % cardinal function coeff
    D = (a*(1./a)')./(dX+eye(N));   % off diagonal element
    D = D - diag(sum(D,2)); % stable computation based on interpolation of constant function 1 (derivative 0)
end

function g = perispecdiff(f)
    % PERISPECDIFF - use FFT to take periodic spectral differentiation of vector
    %
    % g = PERISPECDIFF(f) returns g the derivative of the spectral interpolant
    %  of f, which is assumed to be the values of a smooth 2pi-periodic function
    %  at the N gridpoints 2.pi.j/N, for j=1,..,N (or any translation of such
    %  points).
    %
    % Barnett 2/18/14
    N = numel(f);
    if mod(N,2)==0   % even
      g = ifft(fft(f(:)).*[0 1i*(1:N/2-1) 0 1i*(-N/2+1:-1)].');
    else
      g = ifft(fft(f(:)).*[0 1i*(1:(N-1)/2) 1i*((1-N)/2:-1)].');
    end
    g = reshape(g,size(f));
end
