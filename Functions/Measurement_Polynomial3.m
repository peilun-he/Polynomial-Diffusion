function [y, Jy]= Measurement_Polynomial3(x, par, mats, p_coordinate)

% Polynomial measurement equation with degree 3
% Inputs: 
%   x: x_t
%   par: a vector of parameters
%   mats: maturities
%   p_coordinate: coordinate vector
% Outputs:
%   y: y_t
%   Jy: Jacobian 

kappa_chi  = par(1);
kappa_xi   = par(2);
mu_xi      = par(3);
sigma_chi  = par(4);
sigma_xi   = par(5);
rho        = par(6);
lambda_chi = par(7);
lambda_xi  = par(8);

n_contract = length(mats);

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
     
Hx = [1, x', x(1, :)^2, x(1, :) * x(2, :), x(2, :)^2, x(1, :)^3, x(1, :)^2 * x(2, :), x(1, :) * x(2, :)^2, x(2, :)^3];
exp_matG = zeros(10, 10, n_contract);
exp_matG_p = zeros(10, n_contract);
y = zeros(1, n_contract);
Jy = zeros(n_contract, 2);

for j = 1: n_contract
    exp_matG(:, :, j) = Decomposition_Eigen(mats(:, j)*G);
    exp_matG_p(:, j) = exp_matG(:, :, j) * p_coordinate;
    yt_prediction(j) = Hx * exp_matG(:, :, j) * p_coordinate;
    J(j, 1) = exp_matG_p(2, j) + [2*exp_matG_p(4, j), exp_matG_p(5, j)] * xt_prediction + xt_prediction' * [3*exp_matG_p(7, j), exp_matG_p(8, j); exp_matG_p(8, j), exp_matG_p(9, j)] * xt_prediction;
    J(j, 2) = exp_matG_p(3, j) + [exp_matG_p(5, j), 2*exp_matG_p(6, j)] * xt_prediction + xt_prediction' * [exp_matG_p(8, j), exp_matG_p(9, j); exp_matG_p(9, j), 3*exp_matG_p(10, j)] * xt_prediction;
end


