function [nll, ll_table, table_xt_filter, table_xt_prediction] = EKF3(par, yt, func_f, func_g, mats, n_coe, model, noise)

% Extended Kalman Filter, for polynomial diffusion with degree 3. 
% Ref: Eric Wan & Rudolph van der Merwe (2000)
% Linear-quadratic state-space model: 
%   X_t = f(X_{t-1}) + epsilon_t, epsilon_t ~ N(0, W)
%   Y_t = g(X_t) + eta_t, eta_t ~ N(0, V)

% Inputs: 
%   par: a vector of parameters
%   yt: futures prices
%   mats: time to maturities
%   func_f: function f(x), which should return two values, f(x) and f'(x)
%   func_g: function g(x), which should return two values, g(x) and g'(x)
%   n_coe: the number of model coefficient to be estimated 
%   model: model: Full3 -> S_t = 1 + chi_t + xi_t + chi_t^2 + chi_t*xi_t + xi_t^2 + chi_t^3 + chi_t^2*xi_t + chi_t*xi_t^2 + xi_t^3
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

if abs(mats(1, 1)) > 10^(-10)
    dt = mats(1, 1) - mats(2, 1);
else
    dt = mats(2, 1) - mats(3, 1);
end

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
    error("Incorrect distribution of noises. ");
end

if model == "Full3"
    if n_coe == 10
        p_coordinate = par_coe(1: 10)';
    else
        error("Incorrect number of coefficient. ");
    end
else
    error("Incorrect model. ");
end
 
% Initialization
xt_filter = [ 0 ; mu_xi / kappa_xi ]; % x_0|0
Pt_filter = [ sigma_chi^2 / (2*kappa_chi), sigma_chi*sigma_xi*rho / (kappa_chi + kappa_xi); 
    sigma_chi*sigma_xi*rho / (kappa_chi + kappa_xi), sigma_xi^2 / (2*kappa_xi)]; % P_0|0

for i = 1: n_obs
    % Prediction step
    [xt_prediction, J_state] = f(xt_filter, par); % J_state: Jacobian of f()
    Pt_prediction = J_state * Pt_filter * J_state' + W;
    [yt_prediction, J_measurement] = g(xt_prediction, par, mats(i, :), p_coordinate);
    
    % Filter step
    Pxy = Pt_prediction * J_measurement';
    Pyy = J_measurement * Pt_prediction * J_measurement' + V;
    K = Pxy / Pyy; % K = Pxy * inv(Pyy)
    et = yt(i, :) - yt_prediction;
    xt_filter = xt_prediction + K * et'; 
    Pt_filter = (eye(2) - K * J_measurement) * Pt_prediction;
    %Pt_filter = (eye(2) - K * J_measurement) * Pt_prediction * (eye(2) - K * J_measurement)' + K * V * K'; % Joseph covariance update
    
    % Update tables
    table_xt_filter(i, :) = xt_filter;
    table_xt_prediction(i, :) = xt_prediction;
    table_Pt_filter(:, : ,i) = Pt_filter;
    table_Pt_prediction(:, :, i) = Pt_prediction;
    
    % eigenvalues of (Py+Py')/2 should be positive
    if  sum(sum(diag(eig((Pyy+Pyy')/2)<0)))>0
        disp('matrix is not semi positive definite (EKF)');
    end
    
    % Update likelihood 
    nll = nll + 0.5*length( yt(i, ~isnan(yt(i,:))) )*log(2*pi) + 0.5*log(det(Pyy)) + 0.5*et/Pyy*et';
    ll_table(i) = - (0.5*length( yt(i, ~isnan(yt(i,:))) )*log(2*pi) + 0.5*log(det(Pyy)) + 0.5*et/Pyy*et');
end







