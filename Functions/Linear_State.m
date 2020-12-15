function [fx, Jx] = Linear_State(x, par)

% Linear state equation, x_t = f(x_{t-1}) = A + B x_{t-1}
% Inputs: 
%   x: x_{t-1}
%   par: a vector of parameters
% Outputs: 
%   fx: x_t
%   Jx: Jacobian 

kappa_chi  = par(1);
kappa_xi   = par(2);
mu_xi      = par(3);
sigma_chi  = par(4);
sigma_xi   = par(5);
rho        = par(6);
lambda_chi = par(7);
lambda_xi  = par(8);

A = [ 0; mu_xi / kappa_xi * ( 1-exp(-kappa_xi*dt) ) ]; 
B = [ exp(-kappa_chi*dt), 0; 0, exp(-kappa_xi*dt) ];

fx = A + B * x;
Jx = B;
