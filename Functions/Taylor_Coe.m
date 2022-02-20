function [coe] = Taylor_Coe(order)

% Coefficients of Taylor series of f(chi, xi) = exp(chi+xi) at (0, 0)
% Inputs: 
%   Order: the order of Taylor series
% Outputs:
%   coe: a vector of coefficients

coe = 1;

if order == 0
    return
else
    for s = 1: order
        for i = s: -1: 0
            j = s - i;
            coe = [coe, 1 / (factorial(i) * factorial(j))];
        end
    end
end



