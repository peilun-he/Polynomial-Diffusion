function [nll, ll_table, table_xt_filter, table_xt_prediction, table_xt_smoother, ft] = SKF(par, yt, mats, delivery_time, dt, smoothing, seasonality)

% Standard Kalman Filter & Smoother
% Model: 
%   y_t = d_t + F_t' x_t + f_t + v_t, v_t ~ N(0, V), observation equation
%   x_t = c + G x_{t-1} + w_t, w_t ~ N(0, W), state equation
%   f_t = b*t + beta*cos(2*pi*t*dt) + eta*sin(2*pi*t*dt), seasonal effect
% Inputs: 
%   par: a vector of parameters
%   yt: the logarihm of futures prices
%   T: maturities
%   delivery_time: a vector of date, which is necessary if seasonality is "Constant"
%   dt: delta t
%   smoothing: a boolean variable indicate if Kalman Smoothing is required
%   seasonality: "Constant" or "None"
% Outputs: 
%   nll: the negative log likelihood 
%   ll_table: a vector to store cumulative log-likelihood at each time point - used to calculate Sandwich variance
%   table_at_filter: a nT*2 matrix gives the filtered values of state
%       variables. 
%   table_at_prediction: a (nT+1)*2 matrix gives the predicted values of
%       state variables. 
%   table_at_smoother: a nT*2 matrix gives the smoothed values of state
%       variables. The algorithm of Kalman Smoother is given by Bierman
%       (1973) and De Jong (1989). 
%   ft: seasonal effect

% nT: number of observations
% n: number of contracts
[n_obs, n_contract] = size(yt); 

table_xt_filter = zeros(n_obs, 2); % a_t|t
table_Pt_filter = zeros(2, 2, n_obs); % P_t|t
table_xt_prediction = zeros(n_obs+1, 2); % a_t|t-1
table_Pt_prediction = zeros(2, 2, n_obs+1); % P_t|t-1
table_xt_smoother = zeros(n_obs, 2); % a_t|s, s > t
table_Pt_smoother = zeros(2, 2, n_obs); % P_t|s, s > t

nll = 0; % negative log-likelihood 
ll_table = zeros(1, n_obs); % table of log-likelihood

table_et = zeros(n_obs, n_contract); % e_t
table_invL = zeros(n_contract, n_contract, n_obs); % inverse of L_t|t-1
table_K = zeros(2, n_contract, n_obs); % Kalman gain matrix
table_D = zeros(n_obs, n_contract); % d_t
table_F = zeros(2, n_contract, n_obs); % F_t

% Seasonal component
if seasonality == "Constant"
    par_seasonal = par(end-11: end);
    ft = zeros(n_obs, n_contract);
    for i = 1: n_obs
        ft(i, :) = par_seasonal*(month(delivery_time(i)') == 1: 12)';
    end
    par = par(1: end-12);    
elseif seasonality == "None"
    ft = repelem(0, n_obs);
else
    error("The seasonal component must be 'Constant' or 'None'. ");
end

% Parameters
kappa_chi = par(1);
kappa_xi = par(2);
mu_xi = par(3);
sigma_chi = par(4);
sigma_xi = par(5);
rho = par(6);
lambda_chi = par(7);
lambda_xi = par(8);

if length(par) == 9
    V = diag( repelem(par(9).^2, n_contract) );
elseif length(par) == 11 && n_contract == 13
    V = diag( [repelem(par(9).^2, 5), repelem(par(10).^2, 4), repelem(par(11).^2, 4)] );
elseif length(par) == 21 && n_contract == 13
    V = diag( par(9: end).^2 );
else
    %error("Incorrect number of standard errors. ");
    V = diag( par(9: end).^2 );
end

% Initial a and P
%xt_filter = [ 0 ; mu_xi / kappa_xi ]; % x_0|0
xt_filter = [ 0 ; 2 ]; % x_0|0
Pt_filter = [ sigma_chi^2 / (2*kappa_chi), sigma_chi*sigma_xi*rho / (kappa_chi + kappa_xi); 
    sigma_chi*sigma_xi*rho / (kappa_chi + kappa_xi), sigma_xi^2 / (2*kappa_xi)]; % P_0|0 

% Parameters for state equation
c  = [ 0 ; mu_xi/kappa_xi*(1-exp(-kappa_xi*dt))];
G  = [exp(-kappa_chi*dt), 0; 0, exp(-kappa_xi*dt)];
sigmastate1 = (1-exp(-2*kappa_chi*dt))/(2*kappa_chi)*sigma_chi^2; 
sigmastate2 = (1-exp(-2*kappa_xi*dt))/(2*kappa_xi)*sigma_xi^2;
covstate = (1-exp(-(kappa_chi+kappa_xi)*dt))/(kappa_chi+kappa_xi)*sigma_chi*sigma_xi*rho;
W = [sigmastate1, covstate; covstate, sigmastate2]; % covariance matrix of w_t

% Kalman Filter
for i = 1:n_obs    
    D  = AofT(par, mats(i,:))' + ft(i); % d_t + f_t
    F = [exp(-kappa_chi*mats(i,:)); exp(-kappa_xi*mats(i,:))]'; % F_t
    
    % Prediction step
    xt_prediction  = c + G * xt_filter; % a_t+1|t 
    Pt_prediction = G*Pt_filter*G' + W; % P_t+1|t
    y_prediction = D + F * xt_prediction; % ytilde_t|t-1 = d_t + F_t a_t|t-1
    
    % Filter step
    et = yt(i, :) - y_prediction'; %e_t = y_t - ytilde_t|t-1
    L = F * Pt_prediction * F' + V; % Covariance matrix of et
    invL = L\eye(n_contract); % inverse of L 
    K  = Pt_prediction * F' * invL; % Kalman gain matrix: K_t

    xt_filter = xt_prediction + K*et'; % a_t
    Pt_filter = (eye(2) - K*F)*Pt_prediction; % P_t
           
    % Update tables
    table_xt_filter(i, :) = xt_filter';
    table_Pt_filter(:, :, i) = Pt_filter;
    table_xt_prediction(i+1, :) = xt_prediction';
    table_Pt_prediction(:, :, i+1) = Pt_prediction;
    table_et(i, :) = et;
    table_invL(:, :, i) = invL; 
    table_K(:, :, i) = K;
    table_D(i, :) = D';
    table_F(:, :, i) = F';
    
    % eigenvalues of (Py+Py')/2 should be positive
    if  sum(sum(diag(eig((L+L')/2)<0)))>0
        disp('matrix is not semi positive definite');
    end
    
    % Update likelihood 
    nll = nll + 0.5*length( yt(i, ~isnan(yt(i,:))) )*log(2*pi) + 0.5*log(det(L)) + 0.5*et/L*et';
    ll_table(i) = - (0.5*length( yt(i, ~isnan(yt(i,:))) )*log(2*pi) + 0.5*log(det(L)) + 0.5*et/L*et');
end

% Kalman Smoother
if smoothing
    for t = n_obs: -1: 1
        F = table_F(:, :, t)';
        D = table_D(t, :)';
        K = table_K(:, :, t);
        invL = table_invL(:, :, t);
        et = table_et(t, :)';
        
        xt_prediction = table_xt_prediction(t, :)';
        Pt_prediction = table_Pt_prediction(:, :, t);
        
        if t == n_obs
            rt = [0; 0]; 
            Rt = [0, 0; 0, 0];
        end
        
        rt = F' * invL * et + (G - G * K * F)' * rt; % 2 * 1 matrix 
        Rt = F' * invL * F + (G - G * K * F)' * Rt * (G - G * K *F); % 2 * 2 matrix
     
        xt_smoother = xt_prediction + Pt_prediction * rt; % a_t|n
        Pt_smoother = Pt_prediction - Pt_prediction * Rt * Pt_prediction; %P_t|n
        
        % Update tables
        table_xt_smoother(t, :) = xt_smoother;
        table_Pt_smoother(:, :, t) = Pt_smoother;
    end    
else
    table_xt_smoother = 0;
end



