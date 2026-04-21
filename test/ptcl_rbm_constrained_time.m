% Constrained Particle RBM, TIME EVOLUTION Choco 4/20/2026

Num_panels = 10;
p = 10; % order on panel
qtype = 'p'; qntype = 'G';
N_perwall = p*Num_panels;
Nptcl = 120;
mu=1.0;

R = 1.;
wall.Z = @(t) R*cos(t) + 1j*R*sin(t);
wall.Zp = @(t) - R*sin(t) + 1j*R*cos(t);
wall.Zpp = @(t) - R*cos(t) - 1j*R*sin(t);
wall.p = p;
% wall = setupquad(wall, Nwall);
[wall,~] = quadr(wall, N_perwall, qtype, qntype);
wall.tpan = [wall.tlo;wall.thi(end)];
wall.cw = wall.wxp;
wall.a = 0+0j;

theta0 = pi/4;
a0 = 0.1; b0 = 0.1; % radii
c0 = 0.3; d0 = 0; % center
ptcl.Z = @(t) a0*cos(t)*cos(theta0) - b0*sin(t)*sin(theta0) + c0 ...
            + 1j * (a0*cos(t)*sin(theta0) + b0*sin(t)*cos(theta0) + d0);
ptcl.Zp = @(t) -a0*sin(t)*cos(theta0) - b0*cos(t)*sin(theta0) ...
            + 1j * (-a0*sin(t)*sin(theta0) + b0*cos(t)*cos(theta0));
ptcl.Zpp = @(t) -a0*cos(t)*cos(theta0) + b0*sin(t)*sin(theta0) ...
            + 1j * (-a0*cos(t)*sin(theta0) - b0*sin(t)*cos(theta0));
ptcl = setupquad(ptcl, Nptcl);
ptcl.a = c0+d0*1j;
ptcl.theta0 = theta0;
ptcl_cell = {ptcl};
ptcl_tot = ptcl;

B1 = 1;
beta = -1; % beta < 0: pusher, beta > 0: puller; beta = 0: neutral
B2 = B1 * beta;
jump = 0;

warning('off','MATLAB:nearlySingularMatrix')  % backward-stable ill-cond is ok!
warning('off','MATLAB:rankDeficientMatrix')
lso.RECT = true;  % linsolve opts, forces QR even when square

dt = 0.1;
T = 2;
Nt = T / dt;
M = struct('cdata',[],'colormap',[]);
figure_obj = figure('Visible', 'off');
% figure_obj = figure;
for tstep=1:Nt
    fprintf("\n Time step %d, ptcl at (%f,%f), angle = %f.\n", tstep, real(ptcl.a),imag(ptcl.a), ptcl.theta0);

    vrhs = zeros(2*numel(wall.x),1);
    
    vrhs_ptcl_cplx = get_vslip(B1,B2,ptcl_cell); 
    vrhs_ptcl = [real(vrhs_ptcl_cplx); imag(vrhs_ptcl_cplx)];
    erhs = [vrhs; vrhs_ptcl; zeros(3*numel(ptcl_cell),1)]; 

    Emat = getEmat(wall,ptcl_cell,mu); 
    Edens = linsolve(Emat,erhs,lso);                           % direct bkw stable solve
    fprintf('resid norm = %.3g\n',norm(Emat*Edens - erhs))
    dens_wall = Edens(1:2*numel(wall.x));
    dens_ptcl_tot = Edens((1+2*numel(wall.x)):2*(numel(wall.x)+numel(ptcl_tot.x)));
    U_ptcl_all = Edens(2*(numel(wall.x) + numel(ptcl_tot.x))+(1:2*numel(ptcl_cell)));
    Omega_ptcl_all = Edens(2*(numel(wall.x) + numel(ptcl_tot.x) + numel(ptcl_cell))+1:end);
    fprintf('density norm = %.3g, ptcl density norm = %.3g; U = %f,%f, Omega = %f\n', ...
        norm(dens_wall)/numel(dens_wall), norm(dens_ptcl_tot)/numel(dens_ptcl_tot), U_ptcl_all(1), U_ptcl_all(2), Omega_ptcl_all(1));

    % Plot current time step
    nx = 100; gx = 2.5*((1:nx)/nx-0.5); 
    ny = nx; gy = 2.5*((1:nx)/nx-0.5);
    [xx yy] = meshgrid(gx,gy); t.x = xx(:)+1i*yy(:); Mt = numel(t.x);
    inside = @(z) inpolygon(real(z),imag(z),real(wall.x),imag(wall.x));
    for pind=1:numel(ptcl_cell)
        s_sqr = ptcl_cell{pind};
        inside = @(z) inside(z) & ~inpolygon(real(z),imag(z),real(s_sqr.x),imag(s_sqr.x));
    end
    % di = reshape(inside(t.x),size(xx));  % boolean if inside domain
    ii = inside(t.x);
    tinside = []; tinside.x = t.x(ii);
    ug = nan(size([t.x;t.x])); 
    ux = ptcl_wrapper(@StoSLP_closeglobal,tinside,ptcl_cell,mu)*dens_ptcl_tot + ptcl_wrapper(@StoDLP_closeglobal,tinside,ptcl_cell,mu) * dens_ptcl_tot;
    % ux = ux + StoSLP_closeglobal(tinside,wall,mu,dens_wall,'i') + StoDLP_closeglobal(tinside,wall,mu,dens_wall,'i');
    ux = ux + StoDLP_closepanel(tinside,wall,mu,dens_wall,'i');
    ug([ii;ii]) = ux;
   
    clf;
    hold on; axis equal;
    clim([-5,1])
    u1 = reshape(ug(1:Mt),size(xx)); u2 = reshape(ug(Mt+(1:Mt)),size(xx));
    % figure; 
    magvals = sqrt(u1.^2 + u2.^2);
    imagesc(gx,gy,log10(magvals)); % colormap(jet(256)); 
    colorbar;
    [startX, startY] = meshgrid(gx(1:5:end), gy(1:5:end));
    verts = stream2(gx,gy,u1,u2,startX,startY);
    streamline(verts); 
    quiver(gx,gy, u1,u2,3); 
    plot(real(wall.x),imag(wall.x),'k','LineWidth',2);
    for pind=1:numel(ptcl_cell)
        plot(real(ptcl_cell{pind}.x),imag(ptcl_cell{pind}.x),'k','LineWidth',2);
        % quiver(real(ptcl_cell{ptcl_ind}.a),imag(ptcl_cell{ptcl_ind}.a),
        % cos(ptcl_cell{ptcl_ind}.theta0), sin(ptcl_cell{ptcl_ind}.theta0),
        % 0.25, 'LineWidth', 2); % plot orientation vector, which at this
        % point corresponds to direction of swimming.
    end
    hold off;
    drawnow;
    M(tstep) = getframe(figure_obj);

    % Update particle (hardcoded 1 particle)
    theta0 = ptcl.theta0 + Omega_ptcl_all(1) * dt;
    c0 = c0 + U_ptcl_all(1) * dt;
    d0 = d0 + U_ptcl_all(2) * dt;
    ptcl = [];
    ptcl.Z = @(t) a0*cos(t)*cos(theta0) - b0*sin(t)*sin(theta0) + c0 ...
                + 1j * (a0*cos(t)*sin(theta0) + b0*sin(t)*cos(theta0) + d0);
    ptcl.Zp = @(t) -a0*sin(t)*cos(theta0) - b0*cos(t)*sin(theta0) ...
                + 1j * (-a0*sin(t)*sin(theta0) + b0*cos(t)*cos(theta0));
    ptcl.Zpp = @(t) -a0*cos(t)*cos(theta0) + b0*sin(t)*sin(theta0) ...
                + 1j * (-a0*cos(t)*sin(theta0) - b0*sin(t)*cos(theta0));
    ptcl = setupquad(ptcl, Nptcl);
    ptcl.a = c0 + 1j*d0;
    ptcl.theta0 = theta0;
    
    % Check if any points on particle gets too close to walls
    buffer = 0.1*R;
    touch_container = any(R - abs(ptcl.x-wall.a) < buffer); % true if any point closer to R than buffer
    if (touch_container)
        % Terminate for now. 
        % fprintf("\n terminating.");
        % break;

        % Approach 1: remove normal component
        fprintf("\n Almost touching container, remove normal velocity.")
        % TODO: hardcoded one swimmer.
        % Udir = U_ptcl_all / norm(U_ptcl_all);
        Ndir = ptcl.a - wall.a; % outward normal for container (hardcoded circle) is vec{r} for ptcl center
        if abs(Ndir) > 0
            Ndir = Ndir / abs(Ndir);
            Ndir = [real(Ndir); imag(Ndir)]; % normal 2D vector
            Utang = U_ptcl_all - (U_ptcl_all.'*Ndir) * Ndir; % remove normal component
        else
            fprintf("\n ERROR: ptcl center and wall center overlaps, should not be touching container then.")
        end
        c0 = c0 - U_ptcl_all(1) * dt + Utang(1) * dt; % remove bad update, add in only tangential translation.
        d0 = d0 - U_ptcl_all(2) * dt + Utang(2) * dt;
        ptcl = [];
        ptcl.Z = @(t) a0*cos(t)*cos(theta0) - b0*sin(t)*sin(theta0) + c0 ...
                    + 1j * (a0*cos(t)*sin(theta0) + b0*sin(t)*cos(theta0) + d0);
        ptcl.Zp = @(t) -a0*sin(t)*cos(theta0) - b0*cos(t)*sin(theta0) ...
                    + 1j * (-a0*sin(t)*sin(theta0) + b0*cos(t)*cos(theta0));
        ptcl.Zpp = @(t) -a0*cos(t)*cos(theta0) + b0*sin(t)*sin(theta0) ...
                    + 1j * (-a0*cos(t)*sin(theta0) - b0*sin(t)*cos(theta0));
        ptcl = setupquad(ptcl, Nptcl);
        ptcl.a = c0 + 1j*d0;
        ptcl.theta0 = theta0;
    end
    
    ptcl_cell = {ptcl};
    ptcl_tot = ptcl;
    inside = @(z) inpolygon(real(z),imag(z),real(wall.x),imag(wall.x));
    for pind=1:numel(ptcl_cell)
        s_sqr = ptcl_cell{pind};
        inside = @(z) inside(z) & ~inpolygon(real(z),imag(z),real(s_sqr.x),imag(s_sqr.x));
    end

end

set(figure_obj, 'Visible', 'on');
movie(figure_obj, M, 2, 2);

% Write to video
v = VideoWriter('confined_pusher.mp4', 'MPEG-4');
v.FrameRate = 5; 
open(v);
writeVideo(v, M);
close(v);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end main %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function vslip_tot = get_vslip(B1,B2,ptcl_cell)
    vslip_tot = [];
    for ptcl_ind=1:numel(ptcl_cell)
        s_sqr = ptcl_cell{ptcl_ind};
        Xc = sum(real(s_sqr.x))/numel(s_sqr.x) + 1j*sum(imag(s_sqr.x))/numel(s_sqr.x);
        XmXc = s_sqr.x-Xc; 
        theta = atan2(imag(XmXc), real(XmXc));
        theta = theta - s_sqr.theta0; % Handle particle rotation to recenter theta to canonical shape
        u_theta = B1 * sin(theta) + B2 * sin(theta) .* cos(theta);
        % vslip = -u_theta .* sin(theta) + 1j* u_theta .* cos(theta);
        vslip = u_theta .* real(s_sqr.tang) + 1j*u_theta .* imag(s_sqr.tang);
        vslip_tot = [vslip_tot; vslip];
    end
end

function Emat = getEmat(wall, ptcl_cell, mu)
    % Collect info from all cells for far evals
    % New -- implement multi-swimmer later
    num_ptcls = numel(ptcl_cell);
    ptcl_tot = ptcl_cell{1};
    if num_ptcls>1
        for cellind=2:numel(ptcl_cell)
            ptcl_tot = mergesegquads([ptcl_tot, ptcl_cell{cellind}]);
        end
    end
    n_total_pts = numel(ptcl_tot.x);

    % Source wall
    [A_w2wdl,~,~] = StoDLP(wall,wall,mu);
    % [A_w2wsl,~,~] = StoSLP(wall,wall,mu);
    A_w2w = -eye(2*numel(wall.x))/2 + A_w2wdl; % interior problem for container particle
    % [A_w2pdl,~,T_w2pdl] = StoDLP(ptcl_tot,wall,mu);
    % [A_w2psl,~,T_w2psl] = StoSLP(ptcl_tot,wall,mu);
    [A_w2p,~,T_w2p] = StoDLP_closeglobal(ptcl_tot,wall,mu,[],'i');
    % [A_w2psl,~,T_w2psl] = StoSLP_closeglobal(ptcl_tot,wall,mu,[],'i');
    % A_w2p = A_w2pdl + A_w2psl;
    % T_w2p = T_w2pdl + T_w2psl;

    % Source ptcl
    % [A_p2pdl,~,~] = StoDLP(ptcl,ptcl,mu);
    % [A_p2psl,~,T_p2psl] = StoSLP(ptcl,ptcl,mu);
    % [A_p2wdl,~,~] = StoDLP(wall,ptcl,mu);
    % [A_p2wsl,~,~] = StoSLP(wall,ptcl,mu);
    % A_p2w = A_p2wdl + A_p2wsl;

    A_p2w = ptcl_wrapper(@StoDLP_closeglobal,wall,ptcl_cell,mu) + ptcl_wrapper(@StoSLP_closeglobal,wall,ptcl_cell,mu); % all particle to wall
    [A_p2pdl,~,~] = ptclself_wrapper(@StoDLP_closeglobal,@StoDLP,ptcl_cell,mu); % particle all to all DL
    [A_p2psl,~,T_p2psl] = ptclself_wrapper(@StoSLP_closeglobal,@StoSLP,ptcl_cell,mu); % particle all to all SL
    A_p2p = eye(2*numel(ptcl_tot.x))/2 + A_p2pdl + A_p2psl; % exterior problem

    A = [A_w2w, A_p2w;
         A_w2p, A_p2p];

    % intF
    fMat_ptcl = -(-eye(size(T_p2psl))/2 + T_p2psl); % Exterior problem traction jump
    fMat_wall = -T_w2p;
    
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
          zeros(size(WFy)), WFy] * [fMat_wall, fMat_ptcl];
    intT = [WTx,WTy] * [fMat_wall, fMat_ptcl];

    % Top left matrix incorporating U and Omega in B.C. on particles.
    BCgammaMat = zeros(size(A,1), size(A,2) + 3*num_ptcls);
    BCgammaMat(:, 1:size(A,2)) = A; % wall+ptcl -> wall+ptcl
    BC_ptcl = zeros(n_total_pts*2,3*num_ptcls); % sub matrix containing U, Omega for particle B.C. only.
    XmXc_tot = zeros(n_total_pts,1); % X-Xc collected, complex style
    for ptcl_ind=1:num_ptcls
        s_sqr = ptcl_cell{ptcl_ind};
        Xc = sum(real(s_sqr.x))/numel(s_sqr.x) + 1j*sum(imag(s_sqr.x))/numel(s_sqr.x);
        XmXc = s_sqr.x-Xc; 
        XmXc_tot((ptcl_ind-1)*numel(s_sqr.x)+(1:numel(s_sqr.x))) = XmXc;
        BC_ptcl((ptcl_ind-1)*numel(s_sqr.t)+(1:numel(s_sqr.t)), (ptcl_ind-1)*2 + 1) = -1.*ones(1,numel(s_sqr.t)); % Ux -> BCx
        BC_ptcl(n_total_pts+(ptcl_ind-1)*numel(s_sqr.t)+(1:numel(s_sqr.t)), (ptcl_ind-1)*2+2) = -1.*ones(1,numel(s_sqr.t)); % Uy -> BCy
        BC_ptcl((ptcl_ind-1)*numel(s_sqr.t)+(1:numel(s_sqr.t)), num_ptcls*2+ptcl_ind) = -imag(XmXc); % Omega -> BCx
        BC_ptcl(n_total_pts+(ptcl_ind-1)*numel(s_sqr.t)+(1:numel(s_sqr.t)), num_ptcls*2+ptcl_ind) = real(XmXc); % Omega -> BCy
    end
    BCgammaMat((1+numel(wall.x)*2):end, 1+size(A,2):end) = BC_ptcl;

    
    Emat = [BCgammaMat;
            intF, zeros(2*num_ptcls,3*num_ptcls);
            intT, zeros(num_ptcls,3*num_ptcls)];

end

function [U, P, T] = ptclself_wrapper(kernel,kernel_self,ptcl_cell,mu)
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
                    u = kernel_self(ptcl_i,ptcl_i,mu);
                elseif nargout == 2
                    [u, pres] = kernel_self(ptcl_i,ptcl_i,mu);
                elseif nargout == 3
                    [u, pres, trac] = kernel_self(ptcl_i,ptcl_i,mu);
                end
            else
                ptcl_j = ptcl_cell{j};
                % particle i (with nbrs) to ptcl j
                % u = srcsum(kernel,trlist,[],ptcl_j,ptcl_i,mu);
                if nargout == 1
                    u = kernel(ptcl_j,ptcl_i,mu);
                elseif nargout == 2
                    [u, pres] = kernel(ptcl_j,ptcl_i,mu);
                elseif nargout == 3
                    [u, pres, trac] = kernel(ptcl_j,ptcl_i,mu);
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

function U = ptcl_wrapper(kernel,trg,ptcl_cell,mu)
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
        u = kernel(trg,ptcl_i,mu);
        U(:,srcind) = u;
    end
end
