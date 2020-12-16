function [y, Jy]= Measurement_Polynomial3(x, par, mats, n_coe, model)

% Polynomial measurement equation with degree 2
% Inputs: 
%   x: x_t
%   par: a vector of parameters and model coefficients
%   mats: maturities
%   n_coe: the number of model coefficients
%   model: Quadratic -> S_t = chi_t^2 + xi_t^2
%          Lin-Qua -> S_t = chi_t^2 + xi_t^2 + chi_t + xi_t
%          Mixed -> S_t = chi_t^2 + xi_t^2 + 2*chi_t*xi_t
%          Full-Qua -> S_t = 1 + chi_t + xi_t + 0.5*chi_t^2 + chi_t*xi_t + 0.5*xi_t^2
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

if model == "Quadratic"
    if n_coe == 2
        p_coordinate = [0, 0, 0, par_coe(1), 0, par_coe(2)]';
    elseif n_coe == 0
        p_coordinate = [0, 0, 0, 1, 0, 1]';
    else
        error("Incorrect number of coefficient. ");
    end
elseif model == "Lin-Qua"
    if n_coe == 4
        p_coordinate = [0, par_coe(1), par_coe(2), par_coe(3), 0, par_coe(4)]';
    elseif n_coe == 0
        p_coordinate = [0, 1, 1, 1, 0, 1]';
    else
        error("Incorrect number of coefficient. ");
    end
elseif model == "Mixed"
    if n_coe == 3
        p_coordinate = [0, 0, 0, par_coe(1), par_coe(2), par_coe(3)]';
    elseif n_coe == 0
        p_coordinate = [0, 0, 0, 1, 2, 1]';
    else
        error("Incorrect number of coefficient. ");
    end
elseif model == "Full-Qua"
    if n_coe == 6
        p_coordinate = [par_coe(1), par_coe(2), par_coe(3), par_coe(4), par_coe(5), par_coe(6)]';
    elseif n_coe == 0
        p_coordinate = [1, 1, 1, 0.5, 1, 0.5]';
    else
        error("Incorrect number of coefficient. ");
    end
else
    error("Incorrect model. ");
end

G_est = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
         0,  -kappa_chi,               0, -2*lambda_chi,     mu_xi-lambda_xi,                   0;
         0,           0,       -kappa_xi,             0,         -lambda_chi, 2*mu_xi-2*lambda_xi;
         0,           0,               0,  -2*kappa_chi,                   0,                   0;
         0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
         0,           0,               0,             0,                   0,         -2*kappa_xi];

y = zeros(n_point, n_contract); 
Jy = zeros(n_contract, 2, n_point);

for i = 1: n_point
    chi = x(1, i);
    xi = x(2, i);
    Hx = [1, chi, xi, chi^2, chi * xi, xi^2];
    exp_matG = zeros(6, 6, n_contract);
    exp_matG_p = zeros(6, n_contract);
    
    C = zeros(1, n_contract);
    D = zeros(2, n_contract);
    E = zeros(2, 2, n_contract);

    for j = 1: n_contract
        exp_matG(:, :, j) = Decomposition_Eigen(mats(i, j)*G);
        exp_matG_p(:, j) = exp_matG(:, :, j) * p_coordinate;
        C(j) = exp_matG_p(1, j);
        D(:, j) = exp_matG_p(2: 3, j);
        E(:, :, j) = [exp_matG_p(4, j), 0.5*exp_matG_p(5, j); 0.5*exp_matG_p(5, j), exp_matG_p(6, j)];
        
        y(i, j) = Hx * exp_matG(:, :, j) * p_coordinate;
        Jy(j, :, i) = D(:, j) + 2 * E(:, :, j) * x(:, i);    
    end
end
