clear all;

% Set search path
addpath(genpath(pwd));

% Parameters 
kappa = 1.5;
gamma = 1;
mu = -2;
sigma_chi = 0.5;
sigma_xi = 0.3;
rho = -0.7;
lambda_chi = 0.5;
lambda_xi = 0.3;
s1 = 0.03;
%par_seasonal = [0, 0.1, 0.2, 0.3, 0.2, 0.1, 0, -0.1, -0.2, -0.3, -0.2, -0.1];

% Setups 
n_obs = 1000; % number of observations 
M = 100; % number of iterations to calculate asymptotic variance
h = 0.001; % increments to calculate asymptotic variance
n_contract = 13; % number of futures contracts 

par_org = [kappa, gamma, mu, sigma_chi, sigma_xi, rho, lambda_chi, lambda_xi, repelem(s1, n_contract)]; % original parameters 

monthdays = 30; % days per month
yeardays = 360; % days per year 
dt = 1/yeardays; % delta t
seed = 1111; 
x0 = [0, mu / gamma];
start_date = datetime("2000-01-01", "InputFormat", "yyyy-MM-dd");

n_para = 8; % number of different parameters to be estimated 
n_grid = 2; % number of grid points

% Bounds and constraints 
parL = [10^(-5), 10^(-5),   -5,  0.01,  0.01,  -0.9999, -5, -5, repelem(10^(-5), n_contract)]; % lower bound
parU = [      3,       3,    4,     3,     3,   0.9999,  4,  4, repelem(1, n_contract)      ]; % upper bound
A = [-1, 1, 0, 0, 0, 0, 0, 0, repelem(0, n_contract)]; % constraint: kappa >= gamma 
b = 0;
Aeq = []; % Equal constraints: Aeq*x=beq
beq = []; 

for i = 1: n_contract-1
    Aeq = [Aeq; repelem(0, 7+i), 1, -1, repelem(0, n_contract-1-i)];
    beq = [beq; 0];
end

% Simulate futures prices and maturities 
[yt, mats, xt, ft, date, delivery_time] = SimulateYtMats(par_org, x0, n_obs, n_contract, monthdays, yeardays, "None", start_date, seed); 

%% Parameter estimation 
par0 = [2, 2, 0, 1, 1, 0.5, 0, 0, repelem(0.1, n_contract), 0, repelem(0.1, 11)]; 
options = optimset('TolFun', 1e-06, 'TolX', 1e-06, 'MaxIter', 2000, 'MaxFunEvals', 4000);
[par_est, nll, exitflag] = fmincon(@KalmanFilter, par0, A, b, Aeq, beq, parL, parU, @Const_v2, options, yt, mats, date, dt, false, "Constant");

% Asymptotic variance
method = 2;
seed = 1234;
[asyVar, message] = AsymptoticVariance(par_est, x0, yt, M, h, method, monthdays, yeardays, seed);

%% Grid search 
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
                                init = [init; ka, ga, m, sc, sx, rh, lc, lx, repelem(0.1, n_contract)];
                            end
                        end
                    end
                end
            end
        end
    end
end

parfor i = 1: n_grid^n_para
    i
    par0 =  init(i, :);
    options = optimset('TolFun',1e-06,'TolX',1e-06,'MaxIter',1000,'MaxFunEvals',2000);
    [par, fval, exitflag] = fmincon(@KalmanFilter, par0, A, b, Aeq, beq, parL, parU, @Const_v2, options, yt, mats, date, dt, false, "None");
    est(i, :) = [par, fval];
end

index = ( est(:, end) == min(est(:, end)) );
best_est = est(index, :);
best_init = init(index, :);

% Asymptotic variances
method = 2;
seed = 1234;
[asyVar, message] = AsymptoticVariance(best_est(1: end-1), x0, yt, M, h, method, monthdays, yeardays, seed);





