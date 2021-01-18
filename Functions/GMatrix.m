function [G] = GMatrix(par, degree)

% Calculate the G matrix for polynomial model. 
% Inputs: 
%   par: parameters
%   degree: highest degree of polynomial
% Outputs: 
%   G: G matrix

kappa_chi  = par(1);
kappa_xi   = par(2);
mu_xi      = par(3);
sigma_chi  = par(4);
sigma_xi   = par(5);
rho        = par(6);
lambda_chi = par(7);
lambda_xi  = par(8);

if degree == 0
  G = 0;
elseif degree == 1
  G = [0, -lambda_chi, mu_xi - lambda_xi;
       0,  -kappa_chi,                 0;
       0,           0,         -kappa_xi];
else
  G11 = GMatrix(par, degree - 1);
  G21 = zeros(degree+1, degree*(degree+1)/2);
  i = degree: -1: 0;
  G22 = diag(-i*kappa_chi - (degree-i)*kappa_xi);
  G121 = zeros((degree-1)*(degree-2)/2, degree+1);
  G122 = zeros(degree-1, degree+1);
  G123 = zeros(degree,  degree+1);
  for j = 1: degree
    if j ~= degree
      G122(j, j) = (degree+1-j)*(degree-j)/2*sigma_chi^2;
      G122(j, j+2) = j*(j+1)/2*sigma_xi^2;
    end
    G123(j, j) = -(degree+1-j)*lambda_chi;
    G123(j, j+1) = j*(mu_xi - lambda_xi);
  end
  G12 = [G121; G122; G123];
  G = [G11, G12; G21, G22];
end


