function [yt, mats, xt] = SimulateYt(par, x0, n_obs, n_contract, n_coe, model, noise, seed)

% Simulate data and maturities 

% Inputs: 
%   par: parameters
%   x0: initial value for state variable x_t
%   n_obs: the number of observations
%   n_contracts: the number of contracts
%`  n_coe: the number of model coefficient to be estimated 
%   model: SS2000 -> extended Schwartz & Smith (2000) model, log(S_t) = chi_t + xi_t
%          Quadratic -> S_t = chi_t^2 + xi_t^2
%          Lin-Qua -> S_t = chi_t^2 + xi_t^2 + chi_t + xi_t
%          Mixed -> S_t = chi_t^2 + xi_t^2 + 2*chi_t*xi_t
%          Full-Qua -> S_t = 1 + chi_t + xi_t + 0.5*chi_t^2 + chi_t*xi_t + 0.5*xi_t^2
%   noise: Gaussian -> Gaussian noise for both state and measurement noise
%          Gamma -> Gaussian process noise and Gamma measurement noise
%   seed: seed for random values
% Outputs: 
%   yt: logarithm of futures price for SS2000 model, actual price for Quadratic model
%   mats: maturities
%   xt: state variable  

if n_coe ~= 0
    par_coe = par(end - n_coe + 1: end); % coefficient parameters
    par = par(1: end - n_coe); % model parameters
end

kappa_chi  = par(1);
kappa_xi   = par(2);
mu_xi      = par(3);
sigma_chi  = par(4);
sigma_xi   = par(5);
rho        = par(6);
lambda_chi = par(7);
lambda_xi  = par(8); 

monthdays = 30;
yeardays = 360;
dt = 1 / yeardays; % delta_t

n_state = length(x0);

% Generate random noises for xt and yt
rng(seed); % fix random seed

if noise == "Gaussian"
    s1 = par(9);
    V = diag(repelem(s1^2, n_contract));
    W = [sigma_chi^2/(2*kappa_chi) * ( 1-exp(-2*kappa_chi*dt) ), rho*sigma_chi*sigma_xi/(kappa_chi+kappa_xi) * ( 1-exp(-(kappa_chi+kappa_xi)*dt) ); 
        rho*sigma_chi*sigma_xi/(kappa_chi+kappa_xi) * ( 1-exp(-(kappa_chi+kappa_xi)*dt) ), sigma_xi^2/(2*kappa_xi) * ( 1-exp(-2*kappa_xi*dt) )];
    noise_xt = mvnrnd([0, 0], W, n_obs);
    noise_yt = mvnrnd(repelem(0, n_contract), V, n_obs);
elseif noise == "Gamma"
    alpha = par(9);
    beta = par(10);
    W = [sigma_chi^2/(2*kappa_chi) * ( 1-exp(-2*kappa_chi*dt) ), rho*sigma_chi*sigma_xi/(kappa_chi+kappa_xi) * ( 1-exp(-(kappa_chi+kappa_xi)*dt) ); 
        rho*sigma_chi*sigma_xi/(kappa_chi+kappa_xi) * ( 1-exp(-(kappa_chi+kappa_xi)*dt) ), sigma_xi^2/(2*kappa_xi) * ( 1-exp(-2*kappa_xi*dt) )];
    noise_xt = mvnrnd([0, 0], W, n_obs);
    noise_yt = zeros(n_obs, n_contract);
    for i = 1: n_contract
        noise_yt(:, i) = gamrnd(alpha, 1/beta, n_obs, 1);
    end
    noise_yt = noise_yt - alpha/beta;
else
    error("Incorrect distribution of noises");
end

% Simulate xt
xt = zeros(n_obs+1, n_state);
xt(1, :) = x0;

if model == "SS2000"
    A = [ 0; mu_xi / kappa_xi * ( 1-exp(-kappa_xi*dt) ) ]; 
    B = [ exp(-kappa_chi*dt), 0; 0, exp(-kappa_xi*dt) ];
    
    for j = 2: n_obs+1
        xt(j, :) = A + B * xt(j-1, :)' + noise_xt(j-1, :)'; 
    end

    xt = xt(2: end, :);    
elseif model == "Quadratic"
    A = [ 0; mu_xi / kappa_xi * ( 1-exp(-kappa_xi*dt) ) ]; 
    B = [ exp(-kappa_chi*dt), 0; 0, exp(-kappa_xi*dt) ];
    if n_coe == 2
        p_coordinate = [0, 0, 0, par_coe(1), 0, par_coe(2)]';
    elseif n_coe == 0
        p_coordinate = [0, 0, 0, 1, 0, 1]';
    else
        error("Incorrect number of coefficient. ");
    end
    G = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
         0,  -kappa_chi,               0, -2*lambda_chi,     mu_xi-lambda_xi,                   0;
         0,           0,       -kappa_xi,             0,         -lambda_chi, 2*mu_xi-2*lambda_xi;
         0,           0,               0,  -2*kappa_chi,                   0,                   0;
         0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
         0,           0,               0,             0,                   0,         -2*kappa_xi];
        
    for j = 2: n_obs+1
        xt(j, :) = A + B * xt(j-1, :)' + noise_xt(j-1, :)'; 
    end
    
    xt = xt(2: end, :);
elseif model == "Lin-Qua"
    A = [ 0; mu_xi / kappa_xi * ( 1-exp(-kappa_xi*dt) ) ]; 
    B = [ exp(-kappa_chi*dt), 0; 0, exp(-kappa_xi*dt) ];
    if n_coe == 4
        p_coordinate = [0, par_coe(1), par_coe(2), par_coe(3), 0, par_coe(4)]';
    elseif n_coe == 0
        p_coordinate = [0, 1, 1, 1, 0, 1]';
    else
        error("Incorrect number of coefficient. ");
    end
    G = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
         0,  -kappa_chi,               0, -2*lambda_chi,     mu_xi-lambda_xi,                   0;
         0,           0,       -kappa_xi,             0,         -lambda_chi, 2*mu_xi-2*lambda_xi;
         0,           0,               0,  -2*kappa_chi,                   0,                   0;
         0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
         0,           0,               0,             0,                   0,         -2*kappa_xi];
        
    for j = 2: n_obs+1
        xt(j, :) = A + B * xt(j-1, :)' + noise_xt(j-1, :)'; 
    end
    
    xt = xt(2: end, :);
elseif model == "Mixed"
    A = [ 0; mu_xi / kappa_xi * ( 1-exp(-kappa_xi*dt) ) ]; 
    B = [ exp(-kappa_chi*dt), 0; 0, exp(-kappa_xi*dt) ];
    if n_coe == 3
        p_coordinate = [0, 0, 0, par_coe(1), par_coe(2), par_coe(3)]';
    elseif n_coe == 0
        p_coordinate = [0, 0, 0, 1, 2, 1]';
    else
        error("Incorrect number of coefficient. ");
    end
    G = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
         0,  -kappa_chi,               0, -2*lambda_chi,     mu_xi-lambda_xi,                   0;
         0,           0,       -kappa_xi,             0,         -lambda_chi, 2*mu_xi-2*lambda_xi;
         0,           0,               0,  -2*kappa_chi,                   0,                   0;
         0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
         0,           0,               0,             0,                   0,         -2*kappa_xi];
        
    for j = 2: n_obs+1
        xt(j, :) = A + B * xt(j-1, :)' + noise_xt(j-1, :)'; 
    end
    
    xt = xt(2: end, :);
elseif model == "Full-Qua"
    A = [ 0; mu_xi / kappa_xi * ( 1-exp(-kappa_xi*dt) ) ]; 
    B = [ exp(-kappa_chi*dt), 0; 0, exp(-kappa_xi*dt) ];
    if n_coe == 6
        p_coordinate = [par_coe(1), par_coe(2), par_coe(3), par_coe(4), par_coe(5), par_coe(6)]';
    elseif n_coe == 0
        p_coordinate = [1, 1, 1, 0.5, 1, 0.5]';
    else
        error("Incorrect number of coefficient. ");
    end
    G = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
         0,  -kappa_chi,               0, -2*lambda_chi,     mu_xi-lambda_xi,                   0;
         0,           0,       -kappa_xi,             0,         -lambda_chi, 2*mu_xi-2*lambda_xi;
         0,           0,               0,  -2*kappa_chi,                   0,                   0;
         0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
         0,           0,               0,             0,                   0,         -2*kappa_xi];
        
    for j = 2: n_obs+1
        xt(j, :) = A + B * xt(j-1, :)' + noise_xt(j-1, :)'; 
    end
    
    xt = xt(2: end, :);
else
    error("Incorrect model");
end


% Simulate yt and maturities 
T = monthdays: monthdays: n_contract*monthdays; 
T = T + 1;
mats  = zeros(n_obs, n_contract);
yt = zeros(n_obs, n_contract);

if model == "SS2000"
    for j = 1: n_obs    
        if mod(j-1, monthdays) == 0 && j ~= 1
           T = T + monthdays;
        end

        mats(j, :) = (T - j);
        mats(j, :) = mats(j, :) ./ yeardays;

        yt(j, :) = xt(j, :) * [exp(-kappa_chi*mats(j,:)); exp(-kappa_xi*mats(j, :))] + AofT(par, mats(j,:)) + noise_yt(j, :); 
    end
elseif model == "Quadratic" || model == "Lin-Qua" || model == "Mixed" || model == "Full-Qua"
    Hx = [repelem(1, n_obs)', xt(:, 1), xt(:, 2), xt(:, 1).^2, xt(:, 1) .* xt(:, 2), xt(:, 2).^2]; 
    
    for j = 1: n_obs    
        if mod(j-1, monthdays) == 0 && j ~= 1
           T = T + monthdays;
        end

        mats(j, :) = (T - j);
        mats(j, :) = mats(j, :) ./ yeardays;

        for k = 1: n_contract
            yt(j, k) = Hx(j, :) * Decomposition_Eigen(mats(j, k)*G) * p_coordinate + noise_yt(j, k);
        end
    end
else
    error("Incorrect model");
end





