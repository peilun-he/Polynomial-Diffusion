clear all;

addpath(genpath(pwd));

kappa_chi = 1.5;
kappa_xi = 1;
mu_xi = -2;
sigma_chi = 0.5; 
sigma_xi = 0.3;
rho = -0.7; 
lambda_chi = 0.5;
lambda_xi = 0.3;
s1 = 0.03; 
alpha = 3;
beta = 60;

par_org = [kappa_chi, kappa_xi, mu_xi, sigma_chi, sigma_xi, rho, lambda_chi, lambda_xi, s1];

% Setups
n_obs = 1000;
n_contract = 13;
dt = 1/360; % delta t
seed = 1111; 
x0 = [ 0 ; mu_xi / kappa_xi ]; 
n_para = 8; % number of different parameters to be estimated 
n_grid = 2;
model = "Lin-Qua";
noise = "Gaussian";
n_coe = 0;

% Simulate data
[yt, mats, xt] = SimulateYt(par_org, x0, n_obs, n_contract, n_coe, model, noise, seed);

% Bounds and constraints 
parL = [10^(-5), 10^(-5),   -10,   0.01,   0.01,  -0.9999, -10, -10,  0]; % lower bound
parU = [      3,       3,    10,     10,     10,   0.9999,  10,  10,  1]; % upper bound
A = [-1, 1, 0, 0, 0, 0, 0, 0, 0]; % constraint: kappa >= gamma 
b = 0;
Aeq = []; % Equal constraints: Aeq*x=beq
beq = []; 
options = optimset('TolFun', 1e-06, 'TolX', 1e-06, 'MaxIter', 1000, 'MaxFunEvals', 2000);

%% Estimate state variable
[nll1, xf1, xp1] = EKF(par_org, yt, mats, dt);
[nll2, xf2, xp2] = UKF(par_org, yt, mats, dt);

p_coordinate = [0, 0, 0, 1, 0, 1]';
G = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
     0,  -kappa_chi,               0, -2*lambda_chi,     mu_xi-lambda_xi,                   0;
     0,           0,       -kappa_xi,             0,         -lambda_chi, 2*mu_xi-2*lambda_xi;
     0,           0,               0,  -2*kappa_chi,                   0,                   0;
     0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
     0,           0,               0,             0,                   0,         -2*kappa_xi];

Hx1 = [repelem(1, n_obs)', xf1, xf1(:, 1).^2, xf1(:, 1) .* xf1(:, 2), xf1(:, 2).^2];
Hx2 = [repelem(1, n_obs)', xf2, xf2(:, 1).^2, xf2(:, 1) .* xf2(:, 2), xf2(:, 2).^2];
yf1 = zeros(n_obs, n_contract);
yf2 = zeros(n_obs, n_contract);
for i = 1: n_obs
    for k = 1: n_contract
        exp_G = Decomposition_Eigen(mats(i, k)*G);
        yf1(i, k) = Hx1(i, :) * exp_G * p_coordinate;
        yf2(i, k) = Hx2(i, :) * exp_G * p_coordinate;
    end
end

rmse_yt_EKF = sqrt(mean((yt - yf1).^2));
rmse_yt_UKF = sqrt(mean((yt - yf2).^2));
rmse_xt_EKF = sqrt(mean((xt - xf1).^2));
rmse_xt_UKF = sqrt(mean((xt - xf2).^2));

figure;
subplot(2, 2, 1);
plot(1: n_obs, xt(:, 1), 'r', 1: n_obs, xf1(:, 1), 'k');
legend('Simulated Chi', 'Filtered Chi');
subplot(2, 2, 2);
plot(1: n_obs, xt(:, 2), 'r', 1: n_obs, xf1(:, 2), 'k');
legend('Simulated Xi', 'Filtered Xi');
subplot(2, 2, 3);
plot(1: n_obs, xt(:, 1).^2+xt(:, 2).^2, 'r', 1: n_obs, xf1(:, 1).^2+xf1(:, 2).^2, 'k');
legend('Simulated St', 'Estimated St');
subplot(2, 2, 4);
plot(1: n_obs, yt(:, 1), 'r', 1: n_obs, yf1(:, 1), 'k');
legend('Simulated yt', 'Estimated yt');

figure;
subplot(2, 2, 1);
plot(1: n_obs, xt(:, 1), 'r', 1: n_obs, xf2(:, 1), 'k');
legend('Simulated Chi', 'Filtered Chi');
subplot(2, 2, 2);
plot(1: n_obs, xt(:, 2), 'r', 1: n_obs, xf2(:, 2), 'k');
legend('Simulated Xi', 'Filtered Xi');
subplot(2, 2, 3);
plot(1: n_obs, xt(:, 1).^2+xt(:, 2).^2, 'r', 1: n_obs, xf2(:, 1).^2+xf2(:, 2).^2, 'k');
legend('Simulated St', 'Estimated St');
subplot(2, 2, 4);
plot(1: n_obs, yt(:, 1), 'r', 1: n_obs, yf2(:, 1), 'k');
legend('Simulated yt', 'Estimated yt');

%%
%%%%% Parameter estimation %%%%%
par0 = [2, 2, 0, 1, 1, 0.5, 0, 0, 5, 50];
[par_est, nll, exitflag] = fmincon(@EKF, par0, A, b, Aeq, beq, parL, parU, @Const_v2, options, yt, mats, dt, "Gamma");

%%
%%%%% Grid search %%%%%
tic;
mid = (parL + parU) / 2;
MP1 = (mid + parL) / 2; % midpoint of lower bound and mid
MP2 = (mid + parU) / 2; % midpoint of upper bound and mid

if n_grid == 2
    grid = [MP1; MP2]';
elseif n_grid == 3
    grid = [MP1; mid; MP2]';
else 
    disp('The number of grid points is wrong. ');
end

est = zeros(n_grid^n_para, length(par_org)+1); % estimates of parameters and NLL at each point
init = []; % initial values

for ka = grid(1, :)
    for ga = grid(2, :)
        for m = grid(3, :)
            for sc = grid(4, :)
                for sx = grid(5, :)
                    for rh = grid(6, :)
                        for lc = grid(7, :)
                            for lx = grid(8, :)
                                init = [init; ka, ga, m, sc, sx, rh, lc, lx, 0.5];
                            end
                        end
                    end
                end
            end
        end
    end
end

for i = 1: n_grid^n_para
    i
    par0 =  init(i, :);
    options = optimset('TolFun',1e-06,'TolX',1e-06,'MaxIter',1000,'MaxFunEvals',2000, "Display", "iter");
    [par, fval, exitflag] = fmincon(@EKF, par0, A, b, Aeq, beq, parL, parU, @Const_v2, options, yt, mats, dt, n_coe, model, noise);
    est(i, :) = [par, fval];
end

index = ( est(:, end) == min(est(:, end)) );
best_est = est(index, :);
best_init = init(index, :);

increment = 10^(-5);
[asyVar, message] = Sandwich(best_est(1: end-1), yt, mats, dt, increment, n_coe, model, "EKF", noise);
se = sqrt(diag(asyVar));
time = toc;

%%
[nll, ll_table, xf, xp] = EKF(best_est(1: 9), yt, mats, dt, n_coe, model, noise);

kappa_chi = best_est(1);
kappa_xi = best_est(2);
mu_xi = best_est(3);
sigma_chi = best_est(4); 
sigma_xi = best_est(5);
rho = best_est(6); 
lambda_chi = best_est(7);
lambda_xi = best_est(8);
s1 = best_est(9);

if model == "Quadratic"
    p_coordinate = [0, 0, 0, 1, 0, 1]';
elseif model == "Lin-Qua"
    p_coordinate = [0, 1, 1, 1, 0, 1]';
elseif model == "Mixed"
    p_coordinate = [0, 0, 0, 1, 2, 1]';
else
    error("Incorrect model. ");
end

G_est = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
         0,  -kappa_chi,               0, -2*lambda_chi,     mu_xi-lambda_xi,                   0;
         0,           0,       -kappa_xi,             0,         -lambda_chi, 2*mu_xi-2*lambda_xi;
         0,           0,               0,  -2*kappa_chi,                   0,                   0;
         0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
         0,           0,               0,             0,                   0,         -2*kappa_xi];

Hx = [repelem(1, n_obs)', xf, xf(:, 1).^2, xf(:, 1) .* xf(:, 2), xf(:, 2).^2];
yf = zeros(n_obs, n_contract);
for i = 1: n_obs
    for k = 1: n_contract
        exp_G = Decomposition_Eigen(mats(i, k)*G_est);
        yf(i, k) = Hx(i, :) * exp_G * p_coordinate;
    end
end

rmse_xt = sqrt(mean((xt - xf).^2));
rmse_yt = sqrt(mean((yt - yf).^2));

figure;
subplot(2, 2, 1);
plot(1: n_obs, xt(:, 1), 'r', 1: n_obs, xf(:, 1), 'k');
legend('Simulated Chi', 'Filtered Chi');
subplot(2, 2, 2);
plot(1: n_obs, xt(:, 2), 'r', 1: n_obs, xf(:, 2), 'k');
legend('Simulated Xi', 'Filtered Xi');
subplot(2, 2, 3);
plot(1: n_obs, xt(:, 1).^2+xt(:, 2).^2, 'r', 1: n_obs, xf(:, 1).^2+xf(:, 2).^2, 'k');
legend('Simulated St', 'Estimated St');
subplot(2, 2, 4);
plot(1: n_obs, yt(:, 1), 'r', 1: n_obs, yf(:, 1), 'k');
legend('Simulated yt', 'Estimated yt');

%% Matrix exponential
kappa_chi = 1.5;
kappa_xi = 1.2;
mu_xi = -2;
sigma_chi = 0.5; 
sigma_xi = 0.3;
rho = -0.7; 
s1 = 0.03;
G = [0, 0, mu_xi, sigma_chi^2, 0, sigma_xi^2, 0; 
    0, -kappa_chi, 0, 0, mu_xi, 0, 0;
    0, 0, -kappa_xi, 0, 0, 2*mu_xi, 0;
    0, 0, 0, -2*kappa_chi, 0, 0, 0;
    0, 0, 0, 0, -kappa_chi-kappa_xi, 0, 0; 
    0, 0, 0, 0, 0, -2*kappa_xi, 0;
    0,0,0,0,0,0,1];

expG_eigen = Decomposition_Eigen(G);
expG_taylor = Series_Taylor(G, 3, 10^(-10));
expG_pade = Series_Pade(G, 10, 10);
expG_ss = Series_Scaling_Squaring(G);
expG_lagrange = Polynomial_Lagrange(G);
expG_newton = Polynomial_Newton(G);
expG_vandermonde = Polynomial_Vandermonde(G);

acc_taylor = sum(sum((expG_taylor - expG_eigen).^2));
acc_pade = sum(sum((expG_pade - expG_eigen).^2));
acc_ss = sum(sum((expG_ss - expG_eigen).^2));
acc_lagrange = sum(sum((expG_lagrange - expG_eigen).^2));
acc_newton = sum(sum((expG_newton - expG_eigen).^2));
acc_vandermonde = sum(sum((expG_vandermonde - expG_eigen).^2));

sigma_xi = sigma_xi + 0.1;
G2 = [0, 0, mu_xi, sigma_chi^2, 0, sigma_xi^2; 
    0, -kappa_chi, 0, 0, mu_xi, 0;
    0, 0, -kappa_xi, 0, 0, 2*mu_xi;
    0, 0, 0, -2*kappa_chi, 0, 0;
    0, 0, 0, 0, -kappa_chi-kappa_xi, 0; 
    0, 0, 0, 0, 0, -2*kappa_xi];
%%
sta_taylor = norm(Series_Taylor(G2, 1000, 10^(-6)) - expG_taylor) / norm(expG_taylor);
sta_pade = norm(Series_Pade(G2, 10, 10) - expG_pade) / norm(expG_pade);
sta_ss = norm(Series_Scaling_Squaring(G2) - expG_ss) / norm(expG_ss);
sta_lagrange = norm(Polynomial_Lagrange(G2) - expG_lagrange) / norm(expG_lagrange);
sta_newton = norm(Polynomial_Newton(G2) - expG_newton) / norm(expG_newton);
sta_vandermonde = norm(Polynomial_Vandermonde(G2) - expG_vandermonde) / norm(expG_vandermonde);
sta_eigen = norm(Decomposition_Eigen(G2) - expG_eigen) / norm(expG_eigen);

norm_expG = [];
for t = dt: dt: 1
    norm_expG = [norm_expG, norm(Decomposition_Eigen(t*G))];
end


