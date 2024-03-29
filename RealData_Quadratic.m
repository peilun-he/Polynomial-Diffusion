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
yt = price(first(1): last(end), contracts);
mats = mats0(first(1): last(end), contracts);
yt_forecasting = price(first_forecasting(1): last_forecasting(end), 1: 66);
mats_forecasting = mats0(first_forecasting(1): last_forecasting(end), 1: 66);

delivery_time = delivery_time(first(1): last(end), :);
[n_obs, n_contract] = size(yt);
[n_obs_forecasting, n_contract_forecasting] = size(yt_forecasting);

%% Clustering analysis
rng(1111);
error = zeros(1, 12);
for k = 2: 13
    [idx,C,sumd,d,midx,info] = kmedoids(yt_forecasting(:, 1: 13)', k, "Algorithm", "clara");
    error(k-1) = mean(sumd);
end

plot(2: 13, error);
xlabel("Number of clusters");
ylabel("Mean distance");

%yt_forecasting_cluster1 = yt_forecasting;
%yt_forecasting_cluster1(idx ~= 1, :) = NaN;
%yt_forecasting_cluster2 = yt_forecasting;
%yt_forecasting_cluster2(idx ~= 2, :) = NaN;

%figure;
%hold on;
%plot(date_forecasting, yt_forecasting_cluster1(:, 1), "r");
%plot(date_forecasting, yt_forecasting_cluster2(:, 2), "b");
%xlabel("Date");
%ylabel("Price");
%legend(["Cluter 1", "Cluter 2"]);
%title("k=2, K-medoids (PAM)");
%hold off;

%% Grid search
n_grid = 2;
n_para = 8;
s = 1111;
model = "Polynomial";
noise = "Gaussian";
n_se = 13; % number of standard errors
n_coe = 10; % number of model coefficients
degree = 2; % degree of polynomial

func_f = @(xt, par) State_Linear(xt, par, dt); 
func_g = @(xt, par, mats) Measurement_Polynomial(xt, par, mats, degree, n_coe);  
filter = @UKF3;

% Bounds and constraints
parL = [10^(-5), 10^(-5),   -10,   0.01,   0.01,  -0.9999, -10, -10, repelem(10^(-5), n_se), repelem(-10, n_coe)];
parU = [      3,       3,    10,     10,     10,   0.9999,  10,  10,       repelem(1, n_se),  repelem(10, n_coe)];
A = [-1, 1, 0, 0, 0, 0, 0, 0, repelem(0, n_se + n_coe);
     0, -50, 1, 0, 0, 0, 0, 0, repelem(0, n_se + n_coe);
     0, -50, -1, 0, 0, 0, 0, 0, repelem(0, n_se + n_coe)]; % A*x <= b
b = [0; 0; 0];
Aeq = []; % Equal constraints: Aeq*x=beq
beq = []; 
c = @(x) []; % non-linear constraints: c(x) <= 0
ceq = @(x) []; % non-linear equal constraints: ceq(x) = 0
nlcon = @(x, yt, mats, func_f, func_g, dt, n_coe, noise) deal(c(x), ceq(x)); % non-linear constraints

% Grid search
mid = (parL + parU) / 2;
MP1 = (mid + parL) / 2; % midpoint of lower bound and mid
MP2 = (mid + parU) / 2; % midpoint of upper bound and mid

if n_grid == 2
    grid = [MP1; MP2]';
elseif n_grid == 3
    grid = [MP1; mid; MP2]';
else 
    disp('The number of grid points is wrong. ');
end

est = zeros(n_grid^n_para, length(parL)+1); % estimates of parameters and NLL at each point
init = []; % initial values

for ka = grid(1, :)
    for ga = grid(2, :)
        for m = grid(3, :)
            for sc = grid(4, :)
                for sx = grid(5, :)
                    for rh = grid(6, :)
                        for lc = grid(7, :)
                            for lx = grid(8, :)
                                init = [init; ka, ga, m, sc, sx, rh, lc, lx, repelem(0.1, n_se), repelem(2, n_coe)];
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
    par0 =  init(i, :);
    options = optimset('TolFun',1e-06,'TolX',1e-06,'MaxIter',1000,'MaxFunEvals',2000);
    try
        [par, fval, exitflag] = fmincon(filter, par0, A, b, Aeq, beq, parL, parU, nlcon, options, yt, mats, func_f, func_g, dt, n_coe, noise);
        est(i, :) = [par, fval];
    catch
        est(i, :) = zeros(1, length(parL)+1);
    end
end

error_index = est(:, end) == 0;
est(error_index, :) = [];
init(error_index, :) = [];
index = est(:, end) == min(est(:, end));
best_est = est(index, :);
best_init = init(index, :);

% Asymptotic Variance
increment = 10^(-5);
[asyVar, message] = Sandwich(best_est(1: end-1), yt, mats, func_f, func_g, increment, dt, n_coe, model, filter, noise);
se = sqrt(diag(asyVar));

%% Forecasting error
[~, ~, xf, ~] = filter(best_est(1: end-1), yt_forecasting(:, contracts), mats_forecasting(:, contracts), func_f, func_g, dt, n_coe, noise);

[yf, ~] = Measurement_Polynomial(xf', best_est(1: end-1), mats_forecasting, degree, n_coe);

rmse_f_in = sqrt( mean( (yf(1: n_obs, :) - yt_forecasting(1: n_obs, :)).^2 ) ); % in-sample RMSE
rmse_f_out = sqrt( mean( (yf(n_obs+1: end, :) - yt_forecasting(n_obs+1: end, :)).^2 ) ); % out-of-sample RMSE

%%
date_in_sample = date(first(1): last(end));
date_forecasting = date(first_forecasting(1): last_forecasting(end));
out_contracts = 1: 66;
out_contracts(contracts) = [];
re_indate_incontract = abs( ( yf(1: n_obs, contracts) - yt_forecasting(1: n_obs, contracts) ) ./ yt_forecasting(1: n_obs, contracts) );
re_indate_outcontract = abs( ( yf(1: n_obs, out_contracts) - yt_forecasting(1: n_obs, out_contracts) ) ./ yt_forecasting(1: n_obs, out_contracts) );
re_outdate_incontract = abs( ( yf(n_obs+1: end, contracts) - yt_forecasting(n_obs+1: end, contracts) ) ./ yt_forecasting(n_obs+1: end, contracts) );
re_outdate_outcontract = abs( ( yf(n_obs+1: end, out_contracts) - yt_forecasting(n_obs+1: end, out_contracts) ) ./ yt_forecasting(n_obs+1: end, out_contracts) );
aver_re_indate_incontract = mean(re_indate_incontract, 2);
aver_re_indate_outcontract = mean(re_indate_outcontract, 2);
aver_re_outdate_incontract = mean(re_outdate_incontract, 2);
aver_re_outdate_outcontract = mean(re_outdate_outcontract, 2);

figure;
plot(date_in_sample, aver_re_indate_incontract);
ylim([0, 0.04]);
yticklabels(["0%", "0.5%", "1.0%", "1.5%", "2.0%", "2.5%", "3.0%", "3.5%", "4.0%"]);
xlabel("Date");
ylabel("Relative Errors");

figure;
boxplot(re_indate_incontract);
yticklabels(["0%", "0.5%", "1%", "1.5%", "2%", "2.5%", "3%", "3.5%", "4%", "4.5%"]);
xticklabels(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13"]);
xlabel("Contracts");
ylabel("Relative Errors");

figure;
plot(date_in_sample, yt(:, 1));
text(date_in_sample(86), yt(86, 1), "o", "Color", "r", "FontSize", 12);
xlabel("Date");
ylabel("Price");



