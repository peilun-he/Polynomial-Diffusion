function [y, Jy]= Measurement_Polynomial4(x, par, mats, n_coe, model)

% Polynomial measurement equation with degree 4
% Inputs: 
%   x: x_t
%   par: a vector of parameters and model coefficients
%   mats: maturities
%   n_coe: the number of model coefficients
%   model: Full4 -> S_t = 1 + chi_t + xi_t + 
%                         chi_t^2 + chi_t*xi_t + xi_t^2 + 
%                         chi_t^3 + chi_t^2*xi_t + chi_t*xi_t^2 + xi_t^3 + 
%                         chi_t^4 + chi_t^3*xi_t + chi_t^2*xi_t^2 + chi_t*xi_t^3 + xi_t^4
% Outputs:
%   y: y_t
%   Jy: Jacobian 

if n_coe ~= 0
    par_coe = par(end - n_coe + 1: end); % model coefficients
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

n_contract = size(mats, 2);
n_point = size(x, 2); % number of points

if model == "Full4"
    if n_coe == 15
        p_coordinate = par_coe';
    elseif n_coe == 0
        p_coordinate = [1, 1, 1, 0.5, 1, 0.5, 1/6, 0.5, 0.5, 1/6, 1/24, 1/6, 1/4, 1/6, 1/24]';
    else
        error("Incorrect number of coefficient. ");
    end
else
    error("Incorrect model. ");
end

G11 = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2,             0,                     0,                     0,                   0; 
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
G21 = zeros(5, 10);
G12 = [zeros(3, 5); 
       6*sigma_chi^2,                 0,            sigma_xi^2,                     0,                     0;
                   0,     3*sigma_chi^2,                     0,          3*sigma_xi^2,                     0;
                   0,                 0,           sigma_chi^2,                     0,          6*sigma_xi^2;
       -4*lambda_chi, mu_xi - lambda_xi,                     0,                     0,                     0;
                   0,     -3*lambda_chi, 2*mu_xi - 2*lambda_xi,                     0,                     0;
                   0,                 0,         -2*lambda_chi, 3*mu_xi - 3*lambda_xi,                     0;
                   0,                 0,                     0,           -lambda_chi, 4*mu_xi - 4*lambda_xi;
       ];
G22 = diag([-4*kappa_chi, -3*kappa_chi - kappa_xi, -2*kappa_chi - 2*kappa_xi, -kappa_chi - 3*kappa_xi, -4*kappa_xi]);
G = [G11, G12; G21, G22];

y = zeros(n_point, n_contract); 

chi = x(1, :);
xi = x(2, :);
Hx = [repelem(1, n_point); chi; xi; 
      chi.^2; chi .* xi; xi.^2; 
      chi.^3; chi.^2 .* xi; chi .* xi.^2; xi.^3; 
      chi.^4; chi.^3 .* xi; chi.^2 .* xi.^2; chi .* xi.^3; xi.^4]';            

if size(mats, 1) == 1 && n_point == 1
    Jy = zeros(n_contract, 2);    
    for j = 1: n_contract
        exp_matG = Decomposition_Eigen(mats(:, j)*G);
        exp_matG_p = exp_matG * p_coordinate;
        y(:, j) = Hx * exp_matG * p_coordinate;
        Jy(j, 1) = exp_matG_p(2) + [2*exp_matG_p(4), exp_matG_p(5)] * x + x' * [3*exp_matG_p(7), exp_matG_p(8); exp_matG_p(8), exp_matG_p(9)] * x; % change this line
        Jy(j, 2) = exp_matG_p(3) + [exp_matG_p(5), 2*exp_matG_p(6)] * x + x' * [exp_matG_p(8), exp_matG_p(9); exp_matG_p(9), 3*exp_matG_p(10)] * x; % change this line 
    end
elseif size(mats, 1) == 1 && n_point > 1
    Jy = 0;
    for j = 1: n_contract
        exp_matG = Decomposition_Eigen(mats(:, j)*G);
        y(:, j) = Hx * exp_matG * p_coordinate;  
    end
else
    Jy = 0;   
    for i = 1: n_point
        for j = 1: n_contract
            exp_matG = Decomposition_Eigen(mats(i, j)*G);
            y(i, j) = Hx(i, :) * exp_matG * p_coordinate;
        end
    end
end

