% Particle RBM tests

% TEST 1: spherical particle in free space: vslip = sin(s)/U should give
% translation in x direction.
Nptcl = 280;
mu=1.0;
ptcl.Z = @(t) 1 + 0.2*cos(t) + 1j*(0.3*sin(t)+0.25);
ptcl.Zp = @(t) - 0.2*sin(t) + 1j*0.3*cos(t);
ptcl.Zpp = @(t) - 0.2*cos(t) - 1j*0.3*sin(t);
ptcl = setupquad(ptcl, Nptcl);
ptcl.a = 1+0.25j;
% ptcl = wobblycurve(0.8,0.3,5,Nptcl); 
% ptcl.x = ptcl.x + 1;
% ptcl.a = mean(ptcl.x);

[A_dl,~,~] = StoDLP(ptcl,ptcl,mu);
[A_sl,~,T_sl] = StoSLP(ptcl,ptcl,mu);
A = eye(2*numel(ptcl.x))/2 + A_dl + A_sl; % exterior problem
% A = A_sl; % DEBUG: SL only.

% % Test DL near traction
% [~,~,T_dl_near] = StoDLP_closeglobal(ptcl, ptcl, mu, [], 'e');
% % TODO: check T_dl_near = 0 -- RESULT: NOPE

% intF
fMat_ptcl = -(-eye(size(T_sl))/2 + T_sl); % Exterior problem traction jump
intFx = ptcl.w(:).' * fMat_ptcl(1:end/2,:);
intFy = ptcl.w(:).' * fMat_ptcl(1+end/2:end,:);
Xc = sum(real(ptcl.x))/numel(ptcl.x) + 1j*sum(imag(ptcl.x))/numel(ptcl.x);
XmXc = ptcl.x - Xc; 
intT = [-ptcl.w(:)'.*imag(XmXc)', ptcl.w(:)'.*real(XmXc).'] * fMat_ptcl;

BCmatU = -1*[ones(numel(ptcl.x),1),zeros(numel(ptcl.x),1);
        zeros(numel(ptcl.x),1), ones(numel(ptcl.x),1)];

BCmatT = [-imag(XmXc);real(XmXc)]; % TODO: check size of XmXc

Emat = [A, BCmatU, BCmatT;
        intFx, zeros(1,3);
        intFy, zeros(1,3);
        intT, zeros(1,3)];

% % Resistance problem: known U and Omega
% % CASE 1: U = [0;0], Omega = omega0;
% omega0 = 1.5;
% vslip = [-omega0*imag(XmXc); omega0*real(XmXc)];
% dens = A \ vslip;
% fprintf("mag density = %.3g", norm(dens)/numel(dens));
% % Plot -- should be purely rotational flow.
% nx = 80; gx = 1 + 4*((1:nx)/nx-0.5); ny = nx; gy = 0.25 + 4*((1:nx)/nx-0.5); % plotting grid
% [xx yy] = meshgrid(gx,gy); t.x = xx(:)+1i*yy(:); Mt = numel(t.x);
% inside = @(x) (real(x)-1).^2 + (imag(x)-0.25).^2 > 0.3;
% ii = inside(t.x);
% tinside = []; tinside.x = t.x(ii);
% ug = nan(size([t.x;t.x]));
% ux = StoSLP_closeglobal(tinside,ptcl,mu,dens,'e') + StoDLP_closeglobal(tinside,ptcl,mu,dens,'e');
% ug([ii;ii]) = ux;
% u1 = reshape(ug(1:Mt),size(xx)); u2 = reshape(ug(Mt+(1:Mt)),size(xx));
% magvals = u1.^2 + u2.^2;
% imagesc(gx,gy,magvals); axis equal; colorbar;
% hold on; quiver(gx,gy,u1,u2);
% % CHECKED: rotational flow. 
% % TODO: maybe check magnitude? select t.x set radius away from surface,
% % measure torque/magnitude, check tangent to normal.

% Mobility problem: solve for U and Omega
B1 = 1;
B2 = -1;
theta = atan2(imag(XmXc), real(XmXc)); % Angle of each node
u_theta = B1 * sin(theta) + B2 * sin(theta) .* cos(theta);
vslip = [-u_theta .* sin(theta); u_theta .* cos(theta)]; % vslip checked, tangent and push toward -x.
Erhs = [vslip; 0; 0; 0];
Edens = Emat \ Erhs;
dens = Edens(1:2*numel(ptcl.x));
U = Edens(2*numel(ptcl.x)+(1:2));
Omega_z = Edens(end); 
fprintf("mag density = %.3g, U = %f,%f, Omega_z = %f.\n", norm(dens)/numel(dens), U(1), U(2), Omega_z);

% Plot: 
nx = 80; gx = 1 + 4*((1:nx)/nx-0.5); ny = nx; gy = 0.25 + 4*((1:nx)/nx-0.5); % plotting grid
[xx yy] = meshgrid(gx,gy); t.x = xx(:)+1i*yy(:); Mt = numel(t.x);
inside = @(z) ~inpolygon(real(z),imag(z),real(ptcl.x),imag(ptcl.x));
ii = inside(t.x);
tinside = []; tinside.x = t.x(ii);
ug = nan(size([t.x;t.x]));
ux = StoSLP_closeglobal(tinside,ptcl,mu,dens,'e') + StoDLP_closeglobal(tinside,ptcl,mu,dens,'e');
% ux = StoSLP_closeglobal(tinside,ptcl,mu,dens,'e'); % DEBUG: SL only
ug([ii;ii]) = ux;
u1 = reshape(ug(1:Mt),size(xx)); u2 = reshape(ug(Mt+(1:Mt)),size(xx));
magvals = u1.^2 + u2.^2;
imagesc(gx,gy,magvals); axis equal; colorbar;
hold on; quiver(gx,gy,u1,u2);
[startX, startY] = meshgrid(gx(1:5:end), gy(1:5:end));
verts = stream2(gx,gy,u1,u2,startX,startY);
streamline(verts); 
