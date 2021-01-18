function [y, Jy]= Measurement_Polynomial(x, par, mats, degree, n_coe)

% Polynomial measurement equation
% Inputs: 
%   x: x_t
%   par: a vector of parameters and model coefficients
%   mats: maturities
%   degree: degree of polynomial
%   n_coe: the number of model coefficients
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

if n_coe == (degree+1)*(degree+2)/2
    p_coordinate = par_coe';
elseif n_coe == 0
    p_coordinate = Taylor_Coe(degree)';
else
    error("Incorrect number of coefficient. ");
end

G = GMatrix(par, degree);

y = zeros(n_point, n_contract); 

chi = x(1, :);
xi = x(2, :);
Hx = repelem(1, n_point);
for s = 1: degree
    for i = s: -1: 0
        j = s - i;
        Hx = [Hx; chi.^i .* xi.^j];
    end
end          






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

