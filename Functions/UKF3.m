function [nll, ll_table, table_xt_filter, table_xt_prediction] = UKF3(par, yt, mats, dt, n_coe, model, noise)

% Unscented Kalman Filter, for polynomial diffusion with degree 3. 
% Ref: Eric Wan & Rudolph van der Merwe (2000)
% Linear-quadratic state-space model: 
%   X_t = A + B X_{t-1} + C epsilon_t, CC' = W, epsilon_t ~ N(0, 1)
%   Y_t = D + E X_t + \sum_{k=1}^m e_k X_t' F_k X_t + G eta_t, GG' = V, eta_t ~ N(0, 1)

% Inputs: 
%   par: a vector of parameters
%   yt: futures prices
%   mats: time to maturities
%   dt: delta t
%`  n_coe: the number of model coefficient to be estimated 
%   model: Full3 -> S_t = 1 + chi_t + xi_t + chi_t^2 + chi_t*xi_t + xi_t^2 + chi_t^3 + chi_t^2*xi_t + chi_t*xi_t^2 + xi_t^3
%   noise: Gaussian -> Gaussian noise for both process and measurement noise
%          Gamma -> Gaussian process noise and Gamma measurement noise
% Outputs:
%   nll: negative log-likelihood
%   ll_table: a vector to store cumulative log-likelihood at each time point - used to calculate Sandwich variance
%   table_xt_filter: filtered state variable
%   table_xt_prediction: predicted state variable

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

[n_obs, n_contract] = size(yt);
n_state = 2;
table_xt_filter = zeros(n_obs, n_state); % a_t|t
table_Pt_filter = zeros(n_state, n_state, n_obs); % P_t|t
table_xt_prediction = zeros(n_obs, n_state); % a_t|t-1
table_Pt_prediction = zeros(n_state, n_state, n_obs); % P_t|t-1
nll = 0; % negative log-likelihood
ll_table = zeros(1, n_obs);

% Equation parameters
if noise == "Gaussian"
    if length(par) == 9
        V = diag( repelem(par(9).^2, n_contract) );
    elseif length(par) == 11 && n_contract == 13
        V = diag( [repelem(par(9).^2, 5), repelem(par(10).^2, 4), repelem(par(11).^2, 4)] );
    elseif length(par) == 21 && n_contract == 13
        V = diag( par(9: end).^2 );
    else
        error("Incorrect number of standard errors. ");
    end
    W = [sigma_chi^2/(2*kappa_chi) * ( 1-exp(-2*kappa_chi*dt) ), rho*sigma_chi*sigma_xi/(kappa_chi+kappa_xi) * ( 1-exp(-(kappa_chi+kappa_xi)*dt) ); 
        rho*sigma_chi*sigma_xi/(kappa_chi+kappa_xi) * ( 1-exp(-(kappa_chi+kappa_xi)*dt) ), sigma_xi^2/(2*kappa_xi) * ( 1-exp(-2*kappa_xi*dt) )];
elseif noise == "Gamma"
    s_sq = par(9);
    V = diag(repelem(s_sq, n_contract));
    W = [sigma_chi^2/(2*kappa_chi) * ( 1-exp(-2*kappa_chi*dt) ), rho*sigma_chi*sigma_xi/(kappa_chi+kappa_xi) * ( 1-exp(-(kappa_chi+kappa_xi)*dt) ); 
        rho*sigma_chi*sigma_xi/(kappa_chi+kappa_xi) * ( 1-exp(-(kappa_chi+kappa_xi)*dt) ), sigma_xi^2/(2*kappa_xi) * ( 1-exp(-2*kappa_xi*dt) )];
else
    error("Incorrect distribution of noises");
end

A = [ 0; mu_xi / kappa_xi * ( 1-exp(-kappa_xi*dt) ) ]; 
B = [ exp(-kappa_chi*dt), 0; 0, exp(-kappa_xi*dt) ];

if model == "Full3"
    if n_coe == 10
        p_coordinate = par_coe(1: 10)';
    else
        error("Incorrect number of coefficient. ");
    end
else
    error("Incorrect model. ");
end

G = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2,             0,                     0,                     0,                   0; 
     0,  -kappa_chi,               0, -2*lambda_chi,     mu_xi-lambda_xi,                   0, 3*sigma_chi^2,                     0,            sigma_xi^2,                   0;
     0,           0,       -kappa_xi,             0,         -lambda_chi, 2*mu_xi-2*lambda_xi,             0,           sigma_chi^2,                     0,        3*sigma_xi^2;
     0,           0,               0,  -2*kappa_chi,                   0,                   0, -3*lambda_chi,       mu_xi-lambda_xi,                     0,                   0;
     0,           0,               0,             0, -kappa_chi-kappa_xi,                   0,             0,         -2*lambda_chi,   2*mu_xi-2*lambda_xi,                   0; 
     0,           0,               0,             0,                   0,         -2*kappa_xi,             0,                     0,           -lambda_chi, 3*mu_xi-3*lambda_xi;
     0,           0,               0,             0,                   0,                   0,  -3*kappa_chi,                     0,                     0,                   0;
     0,           0,               0,             0,                   0,                   0,             0, -2*kappa_chi-kappa_xi,                     0,                   0;
     0,           0,               0,             0,                   0,                   0,             0,                     0, -kappa_chi-2*kappa_xi,                   0;
     0,           0,               0,             0,                   0,                   0,             0,                     0,                     0,         -3*kappa_xi;
     ];

% Initialization
xt_filter = [ 0 ; mu_xi / kappa_xi ]; % x_0|0
Pt_filter = [ sigma_chi^2 / (2*kappa_chi), sigma_chi*sigma_xi*rho / (kappa_chi + kappa_xi); 
    sigma_chi*sigma_xi*rho / (kappa_chi + kappa_xi), sigma_xi^2 / (2*kappa_xi)]; % P_0|0 

% Calculate weights of sigma points
%alpha = 10^(-3);
%beta = 2;
%kappa = 0;
alpha = 1;
beta = 0;
kappa = 0;
lambda = alpha^2 * (n_state + kappa) - n_state;
weight_mean = zeros(1, 2*n_state+1);
weight_mean(1) = lambda / (n_state + lambda);
weight_mean(2: end) = 1 / ( 2*(n_state + lambda) );
weight_cov = weight_mean;
weight_cov(1) = lambda / (n_state + lambda) + (1 - alpha^2 + beta);
%weight_mean = zeros(1, 2*n_state+1);
%weight_mean(1) = 0.5;
%weight_mean(2: end) = (1-weight_mean(1))/(2*n_state);
%weight_cov = weight_mean;

for i = 1: n_obs
    % Calculate sigma points
    SP = [xt_filter, xt_filter + chol( ( n_state + lambda ) * Pt_filter )', xt_filter - chol( ( n_state + lambda ) * Pt_filter )']; % each column forms a sigma point   

    % Prediction step
    SPx_prediction = A + B * SP; % 2*(2*n_state+1) matrix
    xt_prediction = weight_mean * SPx_prediction'; % 1*2 matrix
    Pt_prediction = W; % 2*2 matrix
    for j = 1: 2*n_state+1
        Pt_prediction = Pt_prediction + weight_cov(j) * (SPx_prediction(:, j) - xt_prediction') * (SPx_prediction(:, j) - xt_prediction')'; 
    end
    Hx = [repelem(1, 2*n_state+1); SPx_prediction(1, :); SPx_prediction(2, :); SPx_prediction(1, :).^2; SPx_prediction(1, :) .* SPx_prediction(2, :); SPx_prediction(2, :).^2; SPx_prediction(1, :).^3; SPx_prediction(1, :).^2 .* SPx_prediction(2, :); SPx_prediction(1, :) .* SPx_prediction(2, :).^2; SPx_prediction(2, :).^3]'; % H(x) = [1, chi, xi, chi^2, chi*xi, xi^2, chi^3, chi^2*xi, chi*xi^2, xi^3], (2*n_state+1)*10 matrix    
    SPy_prediction = zeros(2*n_state+1, n_contract); % (2*n_state+1)*n_contract matrix
    for k = 1: n_contract
        SPy_prediction(:, k) = Hx * Decomposition_Eigen(mats(i, k)*G) * p_coordinate; % (2*n_state+1)*n_obs matrix
    end
    yt_prediction = weight_mean * SPy_prediction; % 1*n_contract matrix
    
    % Filter step
    Pyy = V; % n_contract*n_contract matrix
    Pxy = 0; % 2*n_contract
    for j = 1: 2*n_state+1
        Pyy = Pyy + weight_cov(j) * (SPy_prediction(j, :) - yt_prediction)' * (SPy_prediction(j, :) - yt_prediction);
        Pxy = Pxy + weight_cov(j) * (SPx_prediction(:, j) - xt_prediction') * (SPy_prediction(j, :) - yt_prediction);
    end
    K = Pxy / Pyy; % n_state * n_contract matrix
    et = yt(i, :) - yt_prediction;
    xt_filter = xt_prediction' + K * et';
    Pt_filter = Pt_prediction - K * Pyy * K';
    %Pt_filter = Pt_prediction - Pxy * K' - K * Pxy' + K * Pyy * K'; % Joseph covariance update
    
    % Update tables
    table_xt_filter(i, :) = xt_filter;
    table_xt_prediction(i, :) = xt_prediction;
    table_Pt_filter(:, : ,i) = Pt_filter;
    table_Pt_prediction(:, :, i) = Pt_prediction;
    
    % eigenvalues of (Py+Py')/2 should be positive
    if  sum(sum(diag(eig((Pyy+Pyy')/2)<0)))>0
        disp('matrix is not semi positive definite (UKF)');
    end
    
    % Update likelihood
    nll = nll + 0.5*length( yt(i, ~isnan(yt(i,:))) )*log(2*pi) + 0.5*log(det(Pyy)) + 0.5*et/Pyy*et';
    ll_table(i) = - (0.5*length( yt(i, ~isnan(yt(i,:))) )*log(2*pi) + 0.5*log(det(Pyy)) + 0.5*et/Pyy*et');
end




