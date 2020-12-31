%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution of the viscous Burgers equation, possibly with linear damping,
% using a direct application of the Carleman method combined with Euler's
% equation. The result is compared with solutions from the inbuilt matlab
% solver pdepe, as well as a direct applicatoin of Euler's method.
% 
% This script in its currrent state should reproduce the results
% in https://arxiv.org/abs/2011.03185, "Efficient quantum algorithm for
% dissipative nonlinear differential equations" by Jin-Peng Liu,
% Herman Øie Kolden, Hari K. Krovi, Nuno F. Loureiro, Konstantina Trivisa,
% Andrew M. Childs.
%
% Code written by Herman Øie Kolden in 2020.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plotting configuration
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
fontsize = 14;

%% Simulation parameters
Re0 = 20; % Desired Reynolds number
L0 = 1; % Domain length
U0 = 1; % Initial maximum velocity
beta = 0; % Linear damping coefficient
f = 1; % Number of oscillations of the sinusoidal initial condition inside the domain
T = 4; % Simulation time

N_max = 3; % Maximum Carleman truncation level
ode_deg = 2; % Degree of the Carleman ODE, should not be changed

nx = 16; % Spatial discretization for Euler's method
nt = 2000; % Temporal discretization for Euler's method
nx_pde = 100; % Spatial discretization for the pdepe solver
nt_pde = 40000; % Temporal discretization for the pdepe solver

%% Initialize

nu = U0*L0/Re0; % Viscosity
Tnl = L0/U0; % Nonlinear time
t_plot = Tnl/2; % Time to plot solution
% Spatial domain edges
x0 = -L0/2;
x1 = L0/2;

% Temporal domain edges
t0 = 0;
t1 = T;

% Euler's method discretization interval sizes and domains
dx = (x1-x0)/(nx-1);
dt = (t1-t0)/(nt-1);
xs = linspace(x0,x1,nx);
ts = linspace(t0,t1,nt);

% pdepe discretization interval sizes
dx_pde = (x1-x0)/(nx_pde-1); % Spatial discretization interval size for pdepe solver
dt_pde = (t1-t0)/(nt_pde-1);
xs_pde = linspace(x0,x1,nx_pde);
ts_pde = linspace(t0,t1,nt_pde);

% Initial condition
u0 = @(x) -U0*sin(2*pi*f*x/L0);
u0s = u0(xs);

% PDE, initial condition and boundary condition for pdepe solver
burger_pde = @(x,t,u,dudx) deal(1, nu*dudx-u^2/2, -beta*u);
burger_ic = u0;
burger_bc = @(xl, ul, xr, ur, t) deal(ul, 0, ur, 0);

%% Check CFL condition

% Dissipative and advective CFL numbers for Euler's method and pdepe
C1_d = U0*dt/dx;
C2_d = 2*nu*dt/dx^2;
C1_pde = U0*dt_pde/dx_pde;
C2_pde = 2*nu*dt_pde/dx_pde^2;
if C1_d > 1
    error(sprintf("C1_d = %.2f\n",C1_d));
end
if C2_d > 1
    error(sprintf("C2_d = %.2f\n",C2_d));
end
if C1_pde > 1
    error(sprintf("C1_pde = %.2f\n",C1_pde));
end
if C2_pde > 1
    error(sprintf("C2_pde = %.2f\n",C2_pde));
end

%% Discretize Burger's equation

F1 = zeros(nx,nx);
F1(1+1:nx+1:end) = nu/dx^2;
F1(1+nx:nx+1:end) = nu/dx^2;
F1(1:nx+1:end) = -2*nu/dx^2;
F1 = F1 - beta*eye(nx); % Add linear damping if present

F2 = zeros(nx,nx^2);
F2((nx^2+nx+1):(nx^2+nx+1):end) = -1/(4*dx);
F2(1+1:(nx^2+nx+1):end) = +1/(4*dx);
F2 = reshape(F2,nx,nx^2);

% Enforce the Dirichlet boundaries within the domain.
F1(1,:) = 0;
F1(end,:) = 0;
F2(1,:) = 0;
F2(end,:) = 0;

% The dicretized Burgers ODE
burger_odefun = @(t, u) F1*u + F2*kron(u,u);

%% Calculate the Carleman convergence number

lambdas = eig(F1);

% Since we included the Dirichlet boundaries explicitly, the matrix F1 has
% a 2D nullspace corresponding to eigenvectors with non-zero boundary
% values. We remove these two zero-eigenvalues on the next line of
% code. We could also have avoided this by not explicitly including the
% zeroed boundaries in the integration domain, which corresponds to reducing
% nx by 2. This would increase lambda_1 according to the formula
% 
% lambda_j = -nu*4/dx^2*sin(j*pi/(2*(nx+1-2)))^2.
%
% Instead we choose to keep the boundaries explicitly in the domain and just
% discard the zero eigenvalues. This will result in a larger R, so
% the claims in the paper are conservative relative to this. Note that the
% spectral norm of F2 is only marginally changed by the zeroing of
% boundaries.
lambdas = lambdas(lambdas ~= 0);

lambda = max(lambdas);
R = abs(norm(u0s)*norm(F2)/lambda);

%% Prepare Carleman matrix

fprintf('Preparing Carleman matrix\n');

% Calculate matrix block sizes
dNs = zeros(N_max,1);
for N = 1:N_max
    dNs(N) = (nx^(N+1)-nx)/(nx-1);
end

A = spalloc(dNs(end),dNs(end),dNs(end)*nx);
Fs = [F1 F2];
for i = 1:N
    for j = 1:min(ode_deg,N_max-i+1)
        a0 = 1+(nx^i-nx)/(nx-1);
        a1 = a0 + nx^i-1;
        b0 = 1+(nx^(j+i-1)-nx)/(nx-1);
        b1 = b0 + nx^(j+i-1)-1;

        Aij = spalloc(nx^i,nx^(i+j-1),nx^(i+j-1+1));
        f0 = 1+(nx^j-nx)/(nx-1);
        f1 = f0+nx^j-1;
        Fj = Fs(:,f0:f1);

        for p = 1:i
            Ia = kronp(sparse(eye(nx)), p-1);
            Ib = kronp(sparse(eye(nx)), i-p);
            Aij = Aij + kron(kron(Ia, Fj), Ib);
        end
        A(a0:a1,b0:b1) = Aij;
    end
end

%% Solve Carleman system

ys_c_N = zeros(N_max,nt,dNs(N_max));
for N = 1:N_max
    A_N = A(1:dNs(N),1:dNs(N));
    y0s = [];
    for i = 1:N
        y0s = [y0s, kronp(u0s,i)];
    end

    fprintf('Solving Carleman N=%d\n',N);

    ys = zeros(nt,dNs(N));
    ys(1,:) = y0s;
    for k = 1:(nt-1)
        ys(k+1,:) = ys(k,:) + dt*(A_N*ys(k,:)')';
    end
    fprintf('Done\n',N);
    ys_c_N(N,:,1:dNs(N)) = real(ys(:,:));
end
us_c_N = ys_c_N(:,:,1:nx);

%% Solve direct Euler

fprintf('Solving direct Euler\n');
us_d = zeros(nt,nx);
us_d(1,:) = u0s;
us_d(2,:) = u0s+dt*(u0s*F1' + kron(u0s,u0s)*F2');

for k = 1:(nt-1)
    us_d(k+1,:) = us_d(k,:)+dt*(F1*us_d(k,:)' + F2*kron(us_d(k,:),us_d(k,:))')';
end

%% Solve "exact" PDE

fprintf('Solving "exact" PDE\n');
us_pde = pdepe(0, burger_pde, burger_ic, burger_bc, xs_pde, ts_pde);

% Interpolate the "exact" results so we can compare with other solutions.
% First interpolate over space, and then over time.
us_pde_interp_temp = zeros(nt,nx_pde);
for i = 1:nx_pde
    us_pde_interp_temp(:,i) = interp1(ts_pde,us_pde(:,i),ts);
end
us_pde_interp = zeros(nt,nx);
for k = 1:nt
    us_pde_interp(k,:) = interp1(xs_pde,us_pde_interp_temp(k,:),xs);
end



%% Calculate errors

% We will now calculate the error between all three solutions. First find
% their differences, and then take their norms over space. The variables
% are named according to:
%
% dus_ = difference
% eps_ = l_2 error
% eps_rel_ = l_inf error
%
% _c_ = Carleman solution
% _d_ = Direct Euler solution
% _pde_ = pdepe solution
%
% _N = per Carleman level N

dus_c_d_N = zeros(N_max,nt,nx);
dus_rel_c_d_N = zeros(N_max,nt,nx);
eps_c_d_N = zeros(N_max,nt);
eps_rel_c_d_N = zeros(N_max,nt);

dus_c_pde_N = zeros(N_max,nt,nx);
dus_rel_c_pde_N = zeros(N_max,nt,nx);
eps_c_pde_N = zeros(N_max,nt);
eps_rel_c_pde_N = zeros(N_max,nt);

dus_d_pde = zeros(nt,nx);
dus_rel_d_pde = zeros(nt,nx);
eps_d_pde = zeros(1,nt);
eps_rel_d_pde = zeros(1,nt);

for N = 1:N_max
    dus_c_d_N(N,:,:) = reshape(us_c_N(N,:,:),nt,nx)-us_d(:,:);
    dus_rel_c_d = reshape(dus_c_d_N(N,:,:),nt,nx)./us_d(:,:);
    dus_rel_c_d(isnan(dus_rel_c_d)) = 0;
    dus_rel_c_d_N(N,:,:) = dus_rel_c_d;
    
    dus_c_pde_N(N,:,:) = reshape(us_c_N(N,:,:),nt,nx)-us_pde_interp(:,:);
    dus_rel_c_pde = reshape(dus_c_pde_N(N,:,:),nt,nx)./us_pde_interp(:,:);
    dus_rel_c_pde(isnan(dus_rel_c_pde)) = 0;
    dus_rel_c_pde(isinf(dus_rel_c_pde)) = 0;
    dus_rel_c_pde_N(N,:,:) = dus_rel_c_pde;
    
    dus_d_pde(:,:) = reshape(us_d(:,:),nt,nx)-us_pde_interp(:,:);
    dus_rel_d_pde = reshape(dus_d_pde(:,:),nt,nx)./us_pde_interp(:,:);
    dus_rel_d_pde(isnan(dus_rel_d_pde)) = 0;
    dus_rel_d_pde(isinf(dus_rel_d_pde)) = 0;
    
    for k = 1:nt
        eps_c_d_N(N,k) = norm(reshape(dus_c_d_N(N,k,:),nx,1));
        eps_rel_c_d_N(N,k) = norm(reshape(dus_rel_c_d_N(N,k,:),nx,1),Inf);
        
        eps_c_pde_N(N,k) = norm(reshape(dus_c_pde_N(N,k,:),nx,1));
        eps_rel_c_pde_N(N,k) = norm(reshape(dus_rel_c_pde_N(N,k,:),nx,1),Inf);
        
        eps_d_pde(k) = norm(reshape(dus_d_pde(k,:),nx,1));
        eps_rel_d_pde(k) = norm(reshape(dus_rel_d_pde(k,:),nx,1),Inf);
    end
end

%% Plot errors

% Find indices for which we will plot the solution
i_plot = find(ts>=t_plot,1);
i_plot_pde = find(ts_pde>=t_plot,1);
i_start = ceil(i_plot*3/4);

figure(1);
clf;

% Plot Initial condition
ax = subplot(2,2,1);
plot(xs_pde,us_pde(1,:),'k-','DisplayName',sprintf('Direct PDE'));
hold on;
plot(xs,us_d(1,:),'ko','DisplayName',sprintf('Direct ODE'));
for N = 1:N_max
    ax.ColorOrderIndex = N;
    plot(xs,reshape(us_c_N(N,1,:),nx,1),'*','DisplayName',sprintf('C. ODE, $N=%d$',N));
end

% Plot absolute l_2 error between direct Euler and pdepe solution
subplot(2,2,2);
semilogy(ts,eps_d_pde,'k--','DisplayName','Direct ODE');
hold on;

% Plot solution at some time
ax = subplot(2,2,3);
plot(xs_pde,us_pde(i_plot_pde,:),'k-','DisplayName',sprintf('Direct PDE'));
hold on;
plot(xs,us_d(i_plot,:),'ko--','DisplayName',sprintf('Direct ODE'));
for N = 1:N_max
    ax.ColorOrderIndex = N;
    plot(xs,reshape(ys_c_N(N,i_plot,1:nx),nx,1),'-.*','DisplayName',sprintf('C. ODE, $N=%d$',N));
end

% Plot relative l_inf error between direct Euler and pdepe solution
subplot(2,2,4);
semilogy(ts,eps_rel_d_pde,'k--','DisplayName','Direct ODE');
hold on;

for N = 1:N_max
    % Plot absolute l_2 error between Carleman and direct Euler solutions
    ax = subplot(2,2,2);
    ax.ColorOrderIndex = N;
    semilogy(ts,eps_c_d_N(N,:),'DisplayName',sprintf('C. ODE, $N=%d$',N));
    hold on;

    % Plot relative l_inf between Carleman and direct Euler solutions
    ax = subplot(2,2,4);
    ax.ColorOrderIndex = N;
    semilogy(ts,eps_rel_c_d_N(N,:),'DisplayName',sprintf('C. ODE, $N=%d$',N));
    hold on;
end

% Format initial condition plot
subplot(2,2,1);
title(sprintf('Initial condition'), 'interpreter','latex');
xlabel('$x$', 'interpreter','latex');
ylabel('$u$', 'interpreter','latex');
xlim([x0 x1]);
lgd = legend();
set(lgd,'fontsize',fontsize-4);
set(gca,'fontsize',fontsize);

% Format absolute l_2 error plot
subplot(2,2,2);
title(sprintf('Absolute error'), 'interpreter','latex');
xlabel('$t$', 'interpreter','latex');
ylabel('$\|\varepsilon_{\mathrm{abs}}\|_2$', 'interpreter','latex');
xline(t_plot,':','DisplayName','T_{nl}/2', 'HandleVisibility', 'Off');
ylim([min([min(eps_c_d_N(:,i_start:end)),eps_d_pde(i_start:end)])*0.1 max(eps_c_d_N(1,:))*10]);
lgd = legend();
set(lgd,'fontsize',fontsize-4);
set(gca,'fontsize',fontsize);

ax = gca;
xruler = ax.XRuler;
old_fmt = xruler.TickLabelFormat;
old_xticks = xruler.TickValues;
old_labels = sprintfc(old_fmt, old_xticks);
new_tick = Tnl/2;
new_label = sprintf(['%s%d%s' old_fmt],'$T_{\mathrm{nl}}/2$');
all_xticks = [old_xticks, new_tick];
all_xlabels = [old_labels, new_label];
[new_xticks, sort_order] = sort(all_xticks);
new_labels = all_xlabels(sort_order);
set(xruler, 'TickValues', new_xticks, 'TickLabels', new_labels);

% Format solution plot
subplot(2,2,3);
title('Solution at $t=T_{\mathrm{nl}}/2$','Interpreter','latex');
xlabel('$x$', 'interpreter','latex');
ylabel('$u$', 'interpreter','latex');
xlim([x0 x1]);
ylim([-U0 U0]);
lgd = legend();
set(gca,'fontsize',fontsize);
set(lgd,'fontsize',fontsize-4);

% Format relative l_inf error plot
subplot(2,2,4);
title(sprintf('Relative error'), 'interpreter','latex');
xlabel('$t$', 'interpreter','latex');
ylabel('$\|\varepsilon_{\mathrm{rel}}\|_{\infty}$', 'interpreter','latex');
xline(t_plot,':','DisplayName','$T_{\mathrm{nl}}/2$', 'HandleVisibility','off');
ylim([min([min(eps_rel_c_d_N(:,i_start:end)),eps_rel_d_pde(i_start:end)])*0.1 max(eps_rel_c_d_N(1,:))*10]);

ax = gca;
xruler = ax.XRuler;
old_fmt = xruler.TickLabelFormat;
old_xticks = xruler.TickValues;
old_labels = sprintfc(old_fmt, old_xticks);
new_tick = Tnl/2;
new_label = sprintf(['%s%d%s' old_fmt],'$T_{\mathrm{nl}}/2$');
all_xticks = [old_xticks, new_tick];
all_xlabels = [old_labels, new_label];
[new_xticks, sort_order] = sort(all_xticks);
new_labels = all_xlabels(sort_order);
set(xruler, 'TickValues', new_xticks, 'TickLabels', new_labels);
lgd = legend();
set(gca,'fontsize',fontsize);
set(lgd,'fontsize',fontsize-4);

% Finalize and save
sgtitle(sprintf('VBE solution with $\\mathrm{Re}=%.2f$, $n_x=%d$, $n_t=%d$, $\\mathrm{R}=%.2f$',Re0,nx,nt, R), 'interpreter','latex', 'fontsize', fontsize+2);
savefig(sprintf('vbe_re0_%.2f_N_%d_nx_%d_nt_%d_rev2.fig',Re0,N_max,nx,nt));