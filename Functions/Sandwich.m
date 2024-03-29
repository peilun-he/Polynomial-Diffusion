function [asyVar, message] = Sandwich(par, yt, mats, func_f, func_g, increment, dt, n_coe, model, filter, noise)

% Calculate Sandwich variances of estimates of parameters. 
% Inputs: 
%   par: a vector of estimates of parameters
%   yt: data
%   mats: time to maturities 
%   func_f: function f(x), which should take two arguments, xt and a vector of parameters, and return two values, f(x) and f'(x)
%   func_g: function g(x), which should take three arguments, xt, a vector of parameters and maturities, and return two values, g(x) and g'(x)
%   increment: the increment to calculate first / second order derivatives numerically
%`  n_coe: the number of model coefficient to be estimated 
%   model: SS2000 -> log(S_t) = chi_t + xi_t
%          Polynomial -> polynomial model, the degree should be specified by func_g
%   filter: filtering function
%   noise: Gaussian -> Gaussian noise for both process and measurement noise
%          Gamma -> Gaussian process noise and Gamma measurement noise
% Outputs:
%   asyVar: asymptotic variance
%   message: 0 -> normal
%            1 -> some variances are negative

n_par = length(par);
[n_obs, ~] = size(yt);

incre = repelem(increment, n_par); % vector of increments
incre_mat = diag(incre); % matrix of increments

ll_derivative1 = zeros(n_par, n_par, n_obs); % prodcut of first order derivative of log-likelihood 
ll_derivative2 = zeros(n_par, n_par, n_obs); % second order derivative of log-likelihood
grad = zeros(n_par, n_obs); % gradient 

for i = 1: n_par
    i
    for j = 1: n_par
        j
        if i == j
            if model == "SS2000"
                [~, ll_table1, ~, ~, ~, ~] = SKF(par + incre_mat(i, :), yt, mats, 0, dt, false, "None"); 
                [~, ll_table2, ~, ~, ~, ~] = SKF(par, yt, mats, 0, dt, false, "None");
                [~, ll_table3, ~, ~, ~, ~] = SKF(par - incre_mat(i, :), yt, mats, 0, dt, false, "None");
            elseif model == "Polynomial"
                [~, ll_table1, ~, ~] = filter(par + incre_mat(i, :), yt, mats, func_f, func_g, dt, n_coe, noise);
                [~, ll_table2, ~, ~] = filter(par, yt, mats, func_f, func_g, dt, n_coe, noise); 
                [~, ll_table3, ~, ~] = filter(par - incre_mat(i, :), yt, mats, func_f, func_g, dt, n_coe, noise);
            else 
                error("Incorrect model. ");
            end
            
            ll_derivative2(i, j, :) = (ll_table1 - 2 * ll_table2 + ll_table3) / (incre(i)^2);
        else            
            if model == "SS2000"
                [~, ll_table1, ~, ~, ~, ~] = SKF(par + incre_mat(i, :) + incre_mat(j, :), yt, mats, 0, dt, false, "None"); 
                [~, ll_table2, ~, ~, ~, ~] = SKF(par + incre_mat(i, :) - incre_mat(j, :), yt, mats, 0, dt, false, "None");
                [~, ll_table3, ~, ~, ~, ~] = SKF(par - incre_mat(i, :) + incre_mat(j, :), yt, mats, 0, dt, false, "None");
                [~, ll_table4, ~, ~, ~, ~] = SKF(par - incre_mat(i, :) - incre_mat(j, :), yt, mats, 0, dt, false, "None");
            elseif model == "Polynomial"
                [~, ll_table1, ~, ~] = filter(par + incre_mat(i, :) + incre_mat(j, :), yt, mats, func_f, func_g, dt, n_coe, noise);
                [~, ll_table2, ~, ~] = filter(par + incre_mat(i, :) - incre_mat(j, :), yt, mats, func_f, func_g, dt, n_coe, noise); 
                [~, ll_table3, ~, ~] = filter(par - incre_mat(i, :) + incre_mat(j, :), yt, mats, func_f, func_g, dt, n_coe, noise); 
                [~, ll_table4, ~, ~] = filter(par - incre_mat(i, :) - incre_mat(j, :), yt, mats, func_f, func_g, dt, n_coe, noise);
            else
                error("Incorrect model. ");
            end

            ll_derivative2(i, j, :) = (ll_table1 - ll_table2 - ll_table3 + ll_table4) / (4 * incre(i) * incre(j)); 
        end
    end
    
    if model == "SS2000"
        [~, ll_table5, ~, ~, ~, ~] = SKF(par + incre_mat(i, :), yt, mats, 0, dt, false, "None"); 
        [~, ll_table6, ~, ~, ~, ~] = SKF(par - incre_mat(i, :), yt, mats, 0, dt, false, "None");
    elseif model == "Polynomial"
        [~, ll_table5, ~, ~] = filter(par + incre_mat(i, :), yt, mats, func_f, func_g, dt, n_coe, noise);
        [~, ll_table6, ~, ~] = filter(par - incre_mat(i, :), yt, mats, func_f, func_g, dt, n_coe, noise);
    else
        error("Incorrect model. ");
    end
    
    grad(i, :) = (ll_table5 - ll_table6) / (2 * incre(i)); 
end

for i = 1: n_obs
    ll_derivative1(: ,:, i) = grad(:, i) * grad(:, i)';
end 

V = sum(ll_derivative1, 3);
J = sum(ll_derivative2, 3);

asyVar = inv(J) * V * inv(J); 

message = 0;

if any(diag(asyVar) < 0) 
    message = 1;
    [V1, D1] = eig(asyVar);
    for i = 1: n_par
        if D1(i, i) <= 0
            D1(i, i) = - D1(i, i); % replace negative variance by its absolute value
        end
    end
    asyVar = V1 * D1 / V1;
end










