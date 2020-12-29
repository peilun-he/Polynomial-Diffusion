clear all;
tic;
% Add search path
addpath(genpath(pwd));

%% Read data
dat = readtable("WTI_Price_Formatted78.csv");
last_trading_date = table2array(dat(:, 1));
last_trading_date = datetime(last_trading_date, 'InputFormat', 'dd/MM/yyyy');
last_trading_date = last_trading_date(~isnat(last_trading_date));
date = table2array(dat(:, 2));
date = datetime(date, 'InputFormat', 'dd/MM/yyyy');
try
    price = table2array(dat(:, 3: end));
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

% price(sum(isnan(price), 2) >= 1, :) = []; % delete missing values 

total_days = zeros(1, length(last_trading_date)); % number of trading days between two delivery days
total_days(1) = find(date == last_trading_date(1));
for i = 2: length(total_days)
    index = find(date == last_trading_date(i));
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

yeardays = sum(total_days(2: 13));
monthdays = round(mean(total_days(2: 13)), 0);
dt = 1 / yeardays;
mats0 = mats0 / yeardays;

first = find(date >= datetime(2015, 1, 1));
last  = find(date <= datetime(2018, 12, 31));
first_forecasting = find(date >= datetime(2015, 1, 1));
last_forecasting = find(date <= datetime(2019, 12, 31)); 

contracts = 1: 13;
yt = log( price(first(1): last(end), contracts) );
mats = mats0(first(1): last(end), contracts);
yt_forecasting = log( price(first_forecasting(1): last_forecasting(end), 1: 66) );
mats_forecasting = mats0(first_forecasting(1): last_forecasting(end), 1: 66);

delivery_time = delivery_time(first(1): last(end), :);
[n_obs, n_contract] = size(yt);
[n_obs_forecasting, n_contract_forecasting] = size(yt_forecasting);

%% Grid search
n_grid = 2;
n_para = 8;
s = 1111;
n_se = 13; % number of standard errors

% Bounds
parL = [10^(-5), 10^(-5),   -10,   0.01,   0.01,  -0.9999, -10, -10, repelem(10^(-5), n_se)];
parU = [      3,       3,    10,     10,     10,   0.9999,  10,  10,       repelem(1, n_se)];
A = [-1, 1, 0, 0, 0, 0, 0, 0, repelem(0, n_se)];
b = 0;
Aeq = []; % Equal constraints: Aeq*x=beq
beq = []; 

%for i = 1: n_contract-1
%    Aeq = [Aeq; repelem(0, 7+i), 1, -1, repelem(0, n_contract-1-i)];
%    beq = [beq; 0];
%end

% Grid search
mid = (parL + parU) / 2;
MP1 = (parL + mid) / 2; % midpoint of lower bound and mid
MP2 = (parU + mid) / 2; % midpoint of upper bound and mid

if n_grid == 2
    init = [MP1; MP2]';
elseif n_grid == 3
    init = [MP1; mid; MP2]';
else 
    disp('The number of grid points is wrong. ');
end

est = zeros(n_grid^n_para, length(parL) + 1); % estimates of parameters and NLL at each point
initial = []; % initial values

for ka = init(1, :)
    for ga = init(2, :)
        for m = init(3, :)
            for sc = init(4, :)
                for sx = init(5, :)
                    for rh = init(6, :)
                        for lc = init(7, :)
                            for lx = init(8, :)
                                initial = [initial; ka, ga, m, sc, sx, rh, lc, lx, repelem(0.1, n_se)];
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
    options = optimset('TolFun',1e-06,'TolX',1e-06,'MaxIter',1000,'MaxFunEvals',2000);
    [par, fval, exitflag] = fmincon(@SKF, par0, A, b, Aeq, beq, parL, parU, @Const_v2, options, yt, mats, delivery_time, dt, false, "None");
    est(i, :) = [par, fval];
end

index = ( est(:, end) == min(est(:, end)) );
best_est = est(index, :);
best_init = initial(index, :);

% Asymptotic variance
increment = 10^(-5);
func_f = 0;
func_g = 0;
[asyVar, message] = Sandwich(best_est(1: end-1), yt, mats, func_f, func_g, increment, dt, 0, "SS2000", "None", "None");
se = sqrt(diag(asyVar));

time = toc;

%% Forecasting error
[~, ~, ~, af, ~, ~, ~] = KalmanFilter(best_est(1: end-1), yt_forecasting(:, contracts), mats_forecasting(:, contracts), delivery_time, dt, false, "None");

yf = zeros(n_obs_forecasting, n_contract_forecasting);

for i = 1: n_obs_forecasting
    D = AofT(best_est(1: end-1), mats_forecasting(i, :))';
    F = [exp(-best_est(1) * mats_forecasting(i, :)); exp(-best_est(2) * mats_forecasting(i, :))]';
    yf(i, :) = D + F*af(i, :)';
end

rmse_f_in = sqrt( mean( (exp(yf(1: n_obs, :)) - exp(yt_forecasting(1: n_obs, :))).^2 ) ); % in-sample RMSE
rmse_f_out = sqrt( mean( (exp(yf(n_obs+1: end, :)) - exp(yt_forecasting(n_obs+1: end, :))).^2 ) ); % out-of-sample RMSE


%%
date_in_sample = date(first(1): last(end));
relative_error = abs( ( exp(yf(1: n_obs, 1: n_contract)) - exp(yt) )./exp(yt) ); 
aver_re = mean(relative_error, 2);
figure;
plot(date_in_sample, aver_re);
yticklabels(["0%", "0.2%", "0.4%", "0.6%", "0.8%", "1%", "1.2%"]);
xticklabels(["01/2011", "07/2011", "01/2012", "07/2012", "01/2013", "07/2013", "01/2014", "07/2014", "01/2015"]);
xlabel("Date");
ylabel("Relative Errors");

figure;
boxplot(relative_error);
yticklabels(["0%", "0.5%", "1%", "1.5%", "2%", "2.5%", "3%", "3.5%"]);
xticklabels(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13"]);
xlabel("Contracts");
ylabel("Relative Errors");




