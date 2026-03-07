function [A, P] = StoDLP_closepanel(t, s, mu, sigma, side)
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
            fprintf("Number of near targets on panel %d is %d\n", k, numel(tk.x));
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
            if nargout > 1
                [Afar,Pfar] = StoDLP(tfar,sk,mu, [real(sigma_k); imag(sigma_k)]); % returns 2Ntrg x Nc.
            else
                Afar = StoDLP(tfar,sk,mu, [real(sigma_k); imag(sigma_k)]); % returns 2Ntrg x Nc.
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
    end
    
    A = [real(Amat);imag(Amat)];

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
    % U.nx = -U.nx; U.cur = -U.cur; U.xp = -U.xp; U.tang=-U.tang; U.cw=-U.cw; U.wxp = -U.wxp; % correct for sense of U, opp from periodicdirpipe

    t.x = [1 + 0.8i];
    t.nx = [1+1i];

    uc = StoDLP_closepanel(t,U,mu,tau(U.t),side);


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

    u = StoDLP(t,U_dense,mu,tau(U_dense.t));    % eval given density cases...

    fprintf('Sto DLP density eval, max abs err in u cmpts:\n')
    disp(max(abs(u-uc)))

    % Ac = StoDLP_closepanel(t,s,mu,[],side);
    % A = StoDLP(t,s,mu);   % compare matrix els...
    % fprintf('Compare matrices: \n')
    % disp(max(abs(A(:)-Ac(:))))

    return;
end