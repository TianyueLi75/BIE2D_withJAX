% singly-periodic in 2D velocity-BC Stokes BVP, ie "pipe" geom and
% particles under rigid body motion, PARAMETER GRID Choco 4/16/2026

Num_panels = 10;
p = 10; % order on panel
N_perwall = p * Num_panels; qtype = 'p'; qntype = 'G'; 
Nptcl = 100;
Nlr = 40;
Nprx = 2*Nlr;    % # proxy pts (2 force comps per pt, so 2M dofs)

% set up upper and lower walls
uc_0.e1 = 2*pi;   % unitcell, e1=lattice vector as a complex number
% U_0.Z = @(t) 2*pi-t + 1i*(1+0.3*sin(2*pi-t)); U.Zp = @(t) -1 - 0.3i*cos(2*pi-t); U.Zpp = @(t) -0.3i*sin(2*pi-t);
% D_0.Z = @(t) t + 1i*(-1+0.3*sin(t)); D.Zp = @(t) 1 + 0.3i*cos(t); D.Zpp = @(t) - 0.3i*sin(t);
U_0.Z = @(t) 2*pi-t + 1i; U_0.Zp = @(t) -1+0*t; U_0.Zpp = @(t) 0*t;
D_0.Z = @(t) t - 1i; D_0.Zp = @(t) 1+0*t; D_0.Zpp = @(t) 0*t;
% Panel based quadrature using GL grids, use axigeom functions
U_0.p = p; D_0.p = p;
[U_0,~] = quadr(U_0, N_perwall, qtype, qntype); 
U_0.trlist = [-uc_0.e1,uc_0.e1];
U_0.tpan = [U_0.tlo;U_0.thi(end)];
U_0.cw = U_0.wxp; 
[D_0,~] = quadr(D_0, N_perwall, qtype, qntype); 
D_0.trlist = [-uc_0.e1,uc_0.e1];
D_0.tpan = [D_0.tlo;D_0.thi(end)];
D_0.cw = D_0.wxp;
s_0 = mergesegquads([U_0,D_0]);
s_0.cw = [U_0.cw; D_0.cw];
s_0.np = U_0.np + D_0.np;
s_0.tlo = [U_0.tlo; D_0.tlo];
s_0.thi = [U_0.thi; D_0.thi];
s_0.xlo = [U_0.xlo; D_0.xlo];
s_0.xhi = [U_0.xhi; D_0.xhi];
s_0.ws = [U_0.ws; D_0.ws];
s_0.wxp = [U_0.wxp; D_0.wxp];
s_0.tpan = [U_0.tpan; D_0.tpan];
% UX = @(x) x + 1i*(1+0.3*sin(x)); % Functions in left-to-right orientation in x for target selecting
% DX = @(x) x + 1i*(-1+0.3*sin(x));
UX = @(x) x + 1i;
DX = @(x) x - 1i;

% set up left and right walls
L_0 = []; R_0 = [];
uc_0.nei = 1; % how many nei copies either side (use eg 1e3 to test A w/o AP)
uc_0.trlist = uc_0.e1*(-uc_0.nei:uc_0.nei);  % list of translations for direct images
[x, w] = gauss(Nlr); x = (1+x)/2; w = w'/2; % quadr on [0,1]
H = U_0.x(end)-D_0.x(1); L_0.x = D_0.Z(0) + H*x; L_0.nx = 0*L_0.x+1; L_0.w = H*w; % left side
R_0 = L_0; R_0.x = L_0.x+uc_0.e1; % right side
uc_0.L = L_0; uc_0.R = R_0;

% set up aux periodizing basis
P_0 = [];
proxyrep = @StoSLP;      % sets proxy pt type via a kernel function call
Rp = 1.1*2*pi; 
P_0.x = pi + Rp*exp(2i*pi*(0:Nprx-1)'/Nprx); P_0 = setupquad(P_0);     % proxy pts

% Flow properties and solver setup
mu = 0.9;   % fluid viscosity
jump = 0;
warning('off','MATLAB:nearlySingularMatrix')  % backward-stable ill-cond is ok!
warning('off','MATLAB:rankDeficientMatrix')
lso.RECT = true;  % linsolve opts, forces QR even when square

% SWIMMER PARAMETERS
theta0_0 = pi/7;
a_0 = 0.3; b_0 = 0.3; % a = x-axis, b = y-axis
c1_0 = 1.; c2_0 = 0.15; % center
B1_0 = 1.23;
B2_0 = -0.73; % swim modes

dt = 0.2;
T = 1;
Nt = T / dt;

% Varying starting positions and angles, sphere of radius 0.3
% c2_list = 0.:0.1:0.6
% theta0_list = 0:pi/10:pi/2
c2_list = 0:0.1:0.3;
theta0_list = 0:-pi/8:-pi/4;

rows = numel(theta0_list);
cols = numel(c2_list);
% margin = 0.02;
% top_gap = 0.05;
% side_pad = 0.03;

% figure('Units', 'normalized', 'OuterPosition', [0 0 1 1]);

for ic2=1:cols
    for jtheta0=1:rows
        theta0 = theta0_list(jtheta0);
        c2 = c2_list(ic2);
        % Reinitialize ptcl param in case previous shifts. 
        c1 = c1_0;
        B1 = B1_0; B2 = B2_0;
        a = a_0; b = b_0;
        U = U_0; D = D_0;
        s = s_0;
        L = L_0; R = R_0; uc = uc_0;
        P = P_0;

        ptcl.Z = @(t) a*cos(t)*cos(theta0) - b*sin(t)*sin(theta0) + c1 ...
                    + 1j * (a*cos(t)*sin(theta0) + b*sin(t)*cos(theta0) + c2);
        ptcl.Zp = @(t) -a*sin(t)*cos(theta0) - b*cos(t)*sin(theta0) ...
                    + 1j * (-a*sin(t)*sin(theta0) + b*cos(t)*cos(theta0));
        ptcl.Zpp = @(t) -a*cos(t)*cos(theta0) + b*sin(t)*sin(theta0) ...
                    + 1j * (-a*cos(t)*sin(theta0) - b*sin(t)*cos(theta0));
        ptcl = setupquad(ptcl, Nptcl);
        ptcl.a = c1 + 1j*c2;
        ptcl.theta0 = theta0;
        ptcl_cell = {ptcl};
        ptcl_tot = ptcl;
        inside = @(z) imag(z-DX(real(z)))>0 & imag(z-UX(real(z)))<0 & ~inpolygon(real(z),imag(z),real(ptcl.x),imag(ptcl.x));
        
        traj = zeros(floor(Nt), 2); % Nt x Nparam when param grid
        for tstep=1:Nt
            fprintf("\n Time step %d, ptcl at (%f,%f), angle = %f.\n", tstep, real(ptcl.a),imag(ptcl.a), ptcl.theta0);
            traj(tstep,1) = ptcl.a; % store swimmer center as trajectory of swimmer
            traj(tstep,2) = ptcl.theta0;
        
            vrhs = zeros(2*numel(s.x),1);
            Tjump = -jump * [real(uc.R.nx);imag(uc.R.nx)]; % traction driving growth (vector func)
            
            vrhs_ptcl_cplx = get_vslip(B1,B2,ptcl_cell); 
            vrhs_ptcl = [real(vrhs_ptcl_cplx); imag(vrhs_ptcl_cplx)];
            erhs = [vrhs; vrhs_ptcl; zeros(3*numel(ptcl_cell),1); zeros(2*Nlr,1);Tjump]; 
        
            [E, A, B,C,Q] = ELSmatrix_rbm(s,ptcl_cell,P,proxyrep,mu,uc); 
            co = linsolve(E,erhs,lso);                           % direct bkw stable solve
            fprintf('resid norm = %.3g\n',norm(E*co - erhs))
            sig = co(1:2*numel(s.x)); 
            sig_ptcl = co(1+2*numel(s.x):2*numel(s.x)+2*numel(ptcl_tot.x));
            U_ptcl_all = co(2*numel(s.x)+2*numel(ptcl_tot.x) + 1 : 2*numel(s.x)+2*numel(ptcl_tot.x) + 2*numel(ptcl_cell)); % 2 U entries per ptcl
            Omega_ptcl_all = co(2*numel(s.x)+2*numel(ptcl_tot.x)+2*numel(ptcl_cell)+1:2*numel(s.x)+2*numel(ptcl_tot.x)+3*numel(ptcl_cell)); % 1 Omega entry per ptcl
            psi = co(2*numel(s.x)+2*numel(ptcl_tot.x)+3*numel(ptcl_cell)+1:end);
            fprintf('density norm = %.3g, ptcl density norm = %.3g, proxy norm = %.3g\n',norm(sig)/numel(sig), norm(sig_ptcl)/numel(sig_ptcl), norm(psi)/numel(psi));
            fprintf("translation velocity: [%f, %f]; rotational: %f", U_ptcl_all(1), U_ptcl_all(2), Omega_ptcl_all(1));
        
            % Update particle (hardcoded 1 particle)
            % No update needed at final time step.
            if tstep < Nt
                theta0_new = ptcl.theta0 + Omega_ptcl_all(1) * dt;
                c1_new = c1 + U_ptcl_all(1) * dt;
                c2_new = c2 + U_ptcl_all(2) * dt;
                ptcl_new = [];
                ptcl_new.Z = @(t) a*cos(t)*cos(theta0_new) - b*sin(t)*sin(theta0_new) + c1_new ...
                            + 1j * (a*cos(t)*sin(theta0_new) + b*sin(t)*cos(theta0_new) + c2_new);
                ptcl_new.Zp = @(t) -a*sin(t)*cos(theta0_new) - b*cos(t)*sin(theta0_new) ...
                            + 1j * (-a*sin(t)*sin(theta0_new) + b*cos(t)*cos(theta0_new));
                ptcl_new.Zpp = @(t) -a*cos(t)*cos(theta0_new) + b*sin(t)*sin(theta0_new) ...
                            + 1j * (-a*cos(t)*sin(theta0_new) - b*sin(t)*cos(theta0_new));
                ptcl_new = setupquad(ptcl_new, Nptcl);
                
                % Check if any points on particle gets too close to walls
                updown_buffer = 0.01;
                touch_U = any(imag(UX(real(ptcl_new.x))-ptcl_new.x) < updown_buffer);
                touch_D = any(imag(ptcl_new.x - DX(real(ptcl_new.x))) < updown_buffer);
                if (touch_U || touch_D)
                    % Terminate for now. 
                    fprintf("\n terminating.");
                    % TODO: repulsive force update
                    break;
                end
        
                ptcl = ptcl_new;
                c1 = c1_new;
                c2 = c2_new;
                theta0 = theta0_new;
                ptcl.a = c1 + 1j*c2;
                ptcl.theta0 = theta0;
                ptcl_cell = {ptcl};
                ptcl_tot = ptcl;
                inside = @(z) imag(z-DX(real(z)))>0 & imag(z-UX(real(z)))<0 & ~inpolygon(real(z),imag(z),real(ptcl.x),imag(ptcl.x));
        
                % Shift frame if getting too close to a inlet/outlet
                inout_buffer = 0.2; % only shift half periods at a time, so can leave larger margin to shift.
                touch_out = any(real(ptcl.x) > real(uc.R.x(1)) - inout_buffer);
                touch_in = any(real(ptcl.x) < real(uc.L.x(1)) + inout_buffer);
                if (touch_in)
                    % Shift left by half period
                    fprintf("\n Shifting window to the left by 3/4 period");
                    [U, D, s, uc, P] = update_geom(-3*uc.e1/4, U, D, N_perwall, uc, Nlr, P);
                elseif touch_out
                    % Shift right by half period
                    fprintf("\n Shifting window to the right by 3/4 period");
                    [U, D, s, uc, P] = update_geom(3*uc.e1/4, U, D, N_perwall, uc, Nlr, P);
                end
            end
        end
        if tstep < Nt % should be <=?
            % terminated early, then final update at tstep Nt
            traj(tstep+1,1) = ptcl.a + U_ptcl_all(1) * dt + 1j * U_ptcl_all(2) * dt; 
            traj(tstep+1,2) = ptcl.theta0 + Omega_ptcl_all(1) * dt;
            tstep = tstep + 1;
        end
        
        % Plot final flow field
        nx = 100; gx = 2*pi*((1:nx)-0.5)/nx; ny = nx; 
        gy = gx - pi; % plotting grid
        % gy = 8*((1:ny)/ny-0.5);
        gx = gx + real(uc.L.x(1)); % shift to start of current vis lens
        [xx yy] = meshgrid(gx,gy); t.x = xx(:)+1i*yy(:); Mt = numel(t.x);
        di = reshape(inside(t.x),size(xx));  % boolean if inside domain
        ug = nan(size([t.x;t.x])); pg = nan(size(t.x)); ii = inside(t.x);
        [ug([ii;ii]), pg(ii)] = evalsol(s,ptcl_tot,ptcl_cell,P,proxyrep,mu,uc,t.x(ii),co);
        
        % clf;
        subplot(numel(theta0_list),numel(c2_list),(jtheta0-1)*numel(c2_list) + ic2);
        % w = (1 - 2*side_pad - margin*cols) / cols;
        % h = (1 - 2*top_gap - margin*rows) / rows;
        % x = side_pad + (ic2 - 1) * (w + margin);
        % y = (1 - top_gap) - (jtheta0 * (h + margin)) + margin;
        % subplot('Position', [x, y, w, h]);
        hold on;
        % xlim([real(uc.L.x(1)), real(uc.R.x(1))]);
        % xlim([0,4*pi]); % hardcoding the several periods for clear visualization
        % clim([-5,1]);
        u1 = reshape(ug(1:Mt),size(xx)); u2 = reshape(ug(Mt+(1:Mt)),size(xx));
        pp = reshape(pg,size(xx)); pp = pp - pp(ceil(ny/2),1); % zero p mid left edge
        % figure; 
        magvals = sqrt(u1.^2 + u2.^2);
        imagesc(gx,gy,log10(magvals)); % colormap(jet(256)); 
        colorbar;
        % quiver(gx,gy, u1,u2,2); 
        [startX, startY] = meshgrid(gx(1:5:end), gy(1:5:end));
        verts = stream2(gx,gy,u1,u2,startX,startY);
        streamline(verts); 
        % title('soln u, with close-eval scheme')
        % showsegment({U,D,ptcl_tot}); 
        plot(s.x(1:end/2));
        plot(s.x(1+end/2:end));
        for ptcl_ind=1:numel(ptcl_cell)
            plot(ptcl_cell{ptcl_ind}.x);
        end
        showsegment({uc.L,uc.R}); 
        % Plot trajectory
        for ti=1:tstep
            quiver(real(traj(ti,1)),imag(traj(ti,1)), cos(real(traj(ti,2))), sin(real(traj(ti,2))), 0.15, 'g', 'LineWidth', 2, 'AutoScaleFactor', 4); 
        end
        hold off;

        if ic2 == 1
            % y label for theta0 values
            ylabel("$\theta_0 = $"+theta0_list(jtheta0), 'Interpreter', 'latex');
        end
        if jtheta0 == numel(theta0_list)
            % x label with c2 values
            xlabel("$c_2 = $"+c2_list(ic2), 'Interpreter', 'latex');
        end

    end
end

sgtitle('Swimmer trajectory, $r=0.3$, starting height ($c_2$) vs angle ($\theta_0$).', 'Interpreter', 'latex');


% Shifts wall by <pi_shift> from the (0,2pi) original period
function [Unew, Dnew, s,uc,Pnew] = update_geom(pi_shift, U, D, N_perwall, uc, Nlr, P)
    Unew = []; Dnew = [];
    Unew.Z = @(t) U.Z(t-pi_shift);
    Dnew.Z = @(t) D.Z(t+pi_shift);
    Unew.Zp = @(t) U.Zp(t-pi_shift);
    Dnew.Zp = @(t) D.Zp(t+pi_shift);
    Unew.Zpp = @(t) U.Zpp(t-pi_shift);
    Dnew.Zpp = @(t) D.Zpp(t+pi_shift);
    Unew.p = U.p; Dnew.p = D.p;
    [Unew,~] = quadr(Unew, N_perwall, 'p', 'G'); 
    Unew.trlist = [-uc.e1,uc.e1];
    Unew.tpan = [Unew.tlo;Unew.thi(end)];
    Unew.cw = Unew.wxp; 
    [Dnew,~] = quadr(Dnew, N_perwall, 'p', 'G'); 
    Dnew.trlist = Unew.trlist;
    Dnew.tpan = [Dnew.tlo;Dnew.thi(end)];
    Dnew.cw = Dnew.wxp;
    s = mergesegquads([Unew,Dnew]);
    s.cw = [Unew.cw; Dnew.cw];
    s.np = Unew.np + Dnew.np;
    s.tlo = [Unew.tlo; Dnew.tlo];
    s.thi = [Unew.thi; Dnew.thi];
    s.xlo = [Unew.xlo; Dnew.xlo];
    s.xhi = [Unew.xhi; Dnew.xhi];
    s.ws = [Unew.ws; Dnew.ws];
    s.wxp = [Unew.wxp; Dnew.wxp];
    s.tpan = [Unew.tpan; Dnew.tpan];

    L = [];
    [x, w] = gauss(Nlr); x = (1+x)/2; w = w'/2; % quadr on [0,1]
    H = Unew.x(end)-Dnew.x(1); L.x = Dnew.Z(0) + H*x; L.nx = 0*L.x+1; L.w = H*w; % left side
    R = L; R.x = L.x+uc.e1; % right side
    uc.L = L; uc.R = R;
    
    Pnew = P;
    Pnew.x = P.x + pi_shift;

end




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