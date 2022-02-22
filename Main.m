% Crude oil futures (or other futures) modelling by Schwartz & Smith two factor model (SS2000) and polynomial diffusion (PD) model. 
% Please change values under section "Global setups 1" and "Global setups 2" ONLY.
% Peilun He, 2022
clear;

% Add search path
addpath(genpath(pwd));

% Global setups 1
model = "SS2000"; % "SS2000" (Schwartz & Smith two-factor model) or "PD" (polynomial diffuison model)
exchange = "NY"; % the exchange where futures traded

if model == "SS2000"
    % Do NOT change this part when using SS2000 model!!!
    % These values have no effect on SS2000 model. Just to make sure all codes work well. 
    noise = "None";
    n_coe = 0; 
    filter = "None";
elseif model == "PD"
    noise = "Gaussian";
    n_coe = 0; % number of model coefficients
    degree = 2; % degree of polynomial  
    filter = @UKF3; % filter function
end

contracts_est = 1: 13; % contracts for estimation 
contracts_fore = 1: 25; % contracts for forecasting
first_est = datetime(2016, 1, 1); % first day for estimation
last_est = datetime(2019, 12, 31); % last day for estimation
first_fore = datetime(2016, 1, 1); % first day for forecasting
last_fore = datetime(2020, 12, 31); % last day for forecasting 

n_grid = 2; % number of grid points
n_para = 8; % number of parameters for grid search 
s = 1111; % seed of random number 
interpolation = false; % if using interpolated data 

% Read data
dat = readtable("NaturalGas_Formatted.csv");

last_trading_date = table2array(dat(:, 1)); % delivery days
last_trading_date = datetime(last_trading_date, 'InputFormat', 'dd/MM/yyyy');
last_trading_date = last_trading_date(~isnat(last_trading_date));

date = table2array(dat(:, 2)); % trading days
date = datetime(date, 'InputFormat', 'dd/MM/yyyy');

if exchange == "NY"
    holi = holidays(datetime(date(end), "Locale", "en_US"), datetime(last_trading_date(end), "Locale", "en_US"));
    date2 = date(end): last_trading_date(end);
    index_nt = []; % index of non-trading days
    for i = 1: length(date2)
        if any(date2(i) == holi) || weekday(date2(i)) == 7 || weekday(date2(i)) == 1
            index_nt = [index_nt, i]; % index of holidays or Saturdays or Sundays
        end
    end
    date2(index_nt) = []; % delete non-trading days
    trading_date = [date; date2']; % all trading days
else
    trading_date = date;
end

try
    price = table2array(dat(:, 3: end)); % futures price
catch
    price_cell = table2cell(dat(:, 3: end));
    price = zeros(size(price_cell));
    for i = 1: size(dat, 2)-2
        if isnumeric(price_cell{1, i})
            price(:, i) = cell2mat(price_cell(:, i));
        elseif ischar(price_cell{1, i})
            price(:, i) = str2double(price_cell(:, i));
        else
            error("Incorrect format. ");
        end
    end
end

total_days = zeros(1, length(last_trading_date)); % number of trading days between two delivery days
total_days(1) = find(trading_date == last_trading_date(1));
for i = 2: length(total_days)
    index = find(trading_date == last_trading_date(i));
    if isempty(index)
        break
    end
    total_days(i) = index - sum(total_days(1: (i-1))); 
end

[n_obs, n_contract] = size(price);
mats0 = zeros(n_obs, n_contract); % time to maturities
j = 1; % position in last_trading_date
total_days_left = cumsum(total_days(j: j+n_contract-1));
for i = 1: n_obs
    mats0(i, :) = total_days_left;
    if i == 1
        delivery_time = last_trading_date(j: j+n_contract-1)'; % delivery time
    else
        delivery_time = [delivery_time; last_trading_date(j: j+n_contract-1)'];
    end
    if total_days_left(1) == 0
        j = j + 1;
        total_days_left = cumsum(total_days(j: j+n_contract-1));
    end
    total_days_left = total_days_left - 1;
end

yeardays = sum(total_days(2: 13)); % number of days per year
monthdays = round(mean(total_days(2: 13)), 0); % number of days per month
dt = 1 / yeardays;
mats0 = mats0 / yeardays;

index_est = find(date >= first_est & date <= last_est); % indices for estimation
index_fore = find(date >= first_fore & date <= last_fore); % indices for forecasting 

if model == "SS2000"
    yt = log( price(index_est(1): index_est(end), contracts_est) ); % prices for estimation
    yt_forecasting = log( price(index_fore(1): index_fore(end), contracts_fore) ); % prices for forecasting
elseif model == "PD"
    yt = price(index_est(1): index_est(end), contracts_est);
    yt_forecasting = price(index_fore(1): index_fore(end), contracts_fore);
else
    error("Incorrect model");
end 

mats = mats0(index_est(1): index_est(end), contracts_est); % time to maturities for estimation
mats_forecasting = mats0(index_fore(1): index_fore(end), contracts_fore); % time to maturities for forecasting 

delivery_time = delivery_time(index_est(1): index_est(end), :);
[n_obs, n_contract] = size(yt);
[n_obs_forecasting, n_contract_forecasting] = size(yt_forecasting);

if sum(mats0 < 0, "all") > 0
    error("Incorrect maturities");
end

% Global setups 2
n_se = size(yt, 2); % number of standard errors to be estimated
if model == "SS2000"
    % Do NOT change this part when using SS2000 model!!!
    % These values have no effect on SS2000 model. Just to make sure all codes work well. 
    func_f = 0;
    func_g = 0;  
elseif model == "PD"
    func_f = @(xt, par) State_Linear(xt, par, dt); 
    func_g = @(xt, par, mats) Measurement_Polynomial(xt, par, mats, degree, n_coe);  
end

% Interpolation
% Formula: y3 = y1 + (T3-T1)/(T2-T1) * (y2-y1)
if interpolation
    T3 = 20: 20: 20*(n_contract - 1);
    T3 = T3 / yeardays;
    y3 = zeros(n_obs, n_contract - 1);
    mats3 = zeros(n_obs, n_contract - 1);
    for i = 1: n_obs
        y1 = yt(i, 1: end - 1);
        y2 = yt(i, 2: end);
        T1 = mats(i, 1: end - 1);
        T2 = mats(i, 2: end);
        y3(i, :) = y1 + (T3-T1) ./ (T2-T1) .* (y2-y1);
        mats3(i, :) = T3;
    end
    yt = y3;
    mats = mats3;
    n_se = n_se - 1;
end

% Bounds
parL = [10^(-5), 10^(-5),   -10,   0.01,   0.01,  -0.9999, -10, -10, repelem(10^(-5), n_se), repelem(10, n_coe)]; % lower bounds of all parameters
parU = [      3,       3,    10,     10,     10,   0.9999,  10,  10,       repelem(1, n_se), repelem(10, n_coe)]; % upper bounds of all parameters 
A = [-1, 1, 0, 0, 0, 0, 0, 0, repelem(0, n_se + n_coe)]; % unequal constraints: A*x<=b
b = 0;
%A = [-1, 1, 0, 0, 0, 0, 0, 0, repelem(0, n_se + n_coe);
%     0, -50, 1, 0, 0, 0, 0, 0, repelem(0, n_se + n_coe);
%     0, -50, -1, 0, 0, 0, 0, 0, repelem(0, n_se + n_coe)]; % A*x <= b
%b = [0; 0; 0];
Aeq = []; % equal constraints: Aeq*x=beq
beq = []; 
c = @(x) []; % non-linear constraints: c(x) <= 0
ceq = @(x) []; % non-linear equal constraints: ceq(x) = 0
nlcon = @(x, yt, mats, func_f, func_g, dt, n_coe, noise) deal(c(x), ceq(x)); % non-linear constraints

%for i = 1: n_contract-1
%    Aeq = [Aeq; repelem(0, 7+i), 1, -1, repelem(0, n_contract-1-i)];
%    beq = [beq; 0];
%end

% Grid search
mid = (parL + parU) / 2;
MP1 = (parL + mid) / 2; % midpoint of lower bound and mid
MP2 = (parU + mid) / 2; % midpoint of upper bound and mid

if n_grid == 2
    grid = [MP1; MP2]';
elseif n_grid == 3
    grid = [MP1; mid; MP2]';
else 
    disp('The number of grid points is wrong. ');
end

est = zeros(n_grid^n_para, length(parL) + 1); % estimates of parameters and NLL at each point
initial = []; % initial values

for ka = grid(1, :)
    for ga = grid(2, :)
        for m = grid(3, :)
            for sc = grid(4, :)
                for sx = grid(5, :)
                    for rh = grid(6, :)
                        for lc = grid(7, :)
                            for lx = grid(8, :)
                                initial = [initial; ka, ga, m, sc, sx, rh, lc, lx, repelem(0.1, n_se), repelem(2, n_coe)];
                            end
                        end
                    end
                end
            end
        end
    end
end

parfor i = 1: n_grid^n_para
    i
    par0 = initial(i, :);
    options = optimset('TolFun',1e-06,'TolX',1e-06,'MaxIter',10000,'MaxFunEvals',20000);
    try
        if model == "SS2000"
            [par, fval, exitflag] = fmincon(@SKF, par0, A, b, Aeq, beq, parL, parU, nlcon, options, yt, mats, delivery_time, dt, false, "None");
        elseif model == "PD"
            [par, fval, exitflag] = fmincon(filter, par0, A, b, Aeq, beq, parL, parU, nlcon, options, yt, mats, func_f, func_g, dt, n_coe, noise);     
        end
        est(i, :) = [par, fval];
    catch
        est(i, :) = zeros(1, length(parL)+1);
    end
end

error_index = est(:, end) == 0; % indices where an error is occured 
est(error_index, :) = []; % delete corresponding estimates 
initial(error_index, :) = []; % delete corresponding initial values 

index_mle = ( est(:, end) == min(est(:, end)) ); % index of MLE estimates 
best_est = est(index_mle, :); % MLE estimates
best_init = initial(index_mle, :); % initial values corresponding to MLE estimates

% Sandwich variances
increment = 10^(-5);
[asyVar, message] = Sandwich(best_est(1: end-1), yt, mats, func_f, func_g, increment, dt, n_coe, model, filter, noise);
se = sqrt(diag(asyVar));

% Forecasting error
if model == "SS2000"
    [~, ~, af, ~, ~, ~] = SKF(best_est(1: end-1), yt_forecasting(:, contracts_est), mats_forecasting(:, contracts_est), delivery_time, dt, false, "None");

    yf = zeros(n_obs_forecasting, n_contract_forecasting); % estimated prices 
    D_mat = zeros(n_obs_forecasting, n_contract_forecasting);
    F_mat = zeros(n_obs_forecasting, n_contract_forecasting, 2);

    for i = 1: n_obs_forecasting
        D = AofT(best_est(1: end-1), mats_forecasting(i, :))';
        D_mat(i, :) = D';
        F = [exp(-best_est(1) * mats_forecasting(i, :)); exp(-best_est(2) * mats_forecasting(i, :))]';
        F_mat(i, :, 1) = F(:, 1);
        F_mat(i, :, 2) = F(:, 2);
        yf(i, :) = D + F*af(i, :)';
    end

    rmse_f_in = sqrt( mean( ( exp(yf(1: n_obs, :)) - exp(yt_forecasting(1: n_obs, :)) ).^2 ) ); % in-sample RMSE
    rmse_f_out = sqrt( mean( ( exp(yf(n_obs+1: end, :)) - exp(yt_forecasting(n_obs+1: end, :)) ).^2 ) ); % out-of-sample RMSE
elseif model == "PD"
    [~, ~, xf, ~] = filter(best_est(1: end-1), yt_forecasting(:, contracts_est), mats_forecasting(:, contracts_est), func_f, func_g, dt, n_coe, noise);
    [yf, ~] = Measurement_Polynomial(xf', best_est(1: end-1), mats_forecasting, degree, n_coe);
    rmse_f_in = sqrt( mean( ( yf(1: n_obs, :) - yt_forecasting(1: n_obs, :) ).^2 ) ); % in-sample RMSE
    rmse_f_out = sqrt( mean( ( yf(n_obs+1: end, :) - yt_forecasting(n_obs+1: end, :) ).^2 ) ); % out-of-sample RMSE
end



