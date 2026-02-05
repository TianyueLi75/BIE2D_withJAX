% Solver for Stokes exterior problem with TWO particles, each containing a point source. Exact solution.

% Choco Jan 2026

clear; 
mu = 0.7; 
a = .3; w = 5;
N = 100;

% plotting = true;
plotting = false;

p = []; p.x = [2+1.5i];

% geometry
s1 = wobblycurve(1,a,w,N); s1.a = mean(s1.x);
s2 = wobblycurve(1,a,w,N); s2.a = mean(s2.x); 
s2.x = s2.x + 3+3i; % shifted s2 particle up and right by 3 each.
s = mergesegquads([s1,s2]);
% s = s1;

% point charges and manufactured solutions
y1 = 0+0.15i; y2 = 2.9+3i; % close to center of each particle
psi1 = [-1;0.3]; psi2 = [1;-0.3];
% Flow generated from Stokeslets at y1 and y2 with strengths psi1 and psi2
exact_sol_func = @(x) StoSLPvelker(mu, x, y1, [])*psi1 + StoSLPvelker(mu, x, y2, [])*psi2;
% y1 = 0+0.15i; psi1 = [-1;0.3];
% exact_sol_func = @(x) StoSLPvelker(mu, x, y1, [])*psi1;

% Debug:
if plotting
    figure; hold on;
    % plot(real(s1.x),imag(s1.x)); plot(real(s2.x), imag(s2.x));
    plot(real(s.x),imag(s.x));
    plot(real(p.x),imag(p.x),'*');
    % TODO: plot exact soln in field..
end

% First try with merged object, should not work with Lap self.
Apre = eye(2*numel(s.x))/2 + StoDLP(s,s,mu) + StoSLP(s,s,mu);

% Second attempt: block matrix form
ptcl_cell = {s1,s2};
% ptcl_cell = {s1};
A = ptcl_wrapper(ptcl_cell,mu); % assumes DL+SL formulation

% norm(Apre - A)

rhs = exact_sol_func(s.x);
tau = A \ rhs;
ut = StoDLP(p,s,mu)*tau + StoSLP(p,s,mu)*tau;
uex = exact_sol_func(p.x);
err = norm(ut - uex);
fprintf('\n error at target p: %.3g', err);


function U = ptcl_wrapper(ptcl_cell,mu)
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

    for i=1:num_ptcl
        ptcl_i = ptcl_cell{i};
        srcind = [ptcl_idx(i):(ptcl_idx(i+1)-1),(ptcl_tot_size+ptcl_idx(i)) : (ptcl_tot_size+ptcl_idx(i+1)-1)];
        for j=1:num_ptcl
            trgind = [ptcl_idx(j):(ptcl_idx(j+1)-1),(ptcl_tot_size+ptcl_idx(j)) : (ptcl_tot_size+ptcl_idx(j+1)-1)];
            if i==j
                % ptcl_i self eval
                u = eye(2*numel(ptcl_i.x))/2 + StoSLP(ptcl_i,ptcl_i,mu)+StoDLP(ptcl_i,ptcl_i,mu);
                U(trgind,srcind) = U(trgind, srcind) + u;
            else
                ptcl_j = ptcl_cell{j};
                % particle i (with nbrs) to ptcl j
                u = StoSLP(ptcl_j,ptcl_i,mu)+StoDLP(ptcl_j,ptcl_i,mu);
                U(trgind, srcind) = U(trgind, srcind) + u;
            end
        end
    end
end
