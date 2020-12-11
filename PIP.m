clear all;

addpath(genpath(pwd));

% Setups
n_obs = 1000;
n_contract = 13;
monthdays = 30; % days per month
yeardays = 360; % days per year 
dt = 1/yeardays; % delta t
seed = 1111; 

n_run = 1000;

nll_ekf_table = zeros(1, n_run);
nll_ekf_neg1_table = zeros(1, n_run); % lambda_chi -> -lambda_chi
nll_ekf_neg2_table = zeros(1, n_run); % lambda_xi -> -lambda_xi, mu_xi -> -mu_xi
nll_ekf_neg3_table = zeros(1, n_run); % lambda_chi -> -lambda_chi, lambda_xi -> -lambda_xi, mu_xi -> -mu_xi
nll_ekf_neg4_table = zeros(1, n_run); % rho -> -rho

nll_ukf_table = zeros(1, n_run);
nll_ukf_neg1_table = zeros(1, n_run);
nll_ukf_neg2_table = zeros(1, n_run);
nll_ukf_neg3_table = zeros(1, n_run);
nll_ukf_neg4_table = zeros(1, n_run);

rmse_yt_ekf_table = zeros(n_run, n_contract);
rmse_yt_ekf_neg1_table = zeros(n_run, n_contract);
rmse_yt_ekf_neg2_table = zeros(n_run, n_contract);
rmse_yt_ekf_neg3_table = zeros(n_run, n_contract);
rmse_yt_ekf_neg4_table = zeros(n_run, n_contract);

rmse_yt_ukf_table = zeros(n_run, n_contract);
rmse_yt_ukf_neg1_table = zeros(n_run, n_contract);
rmse_yt_ukf_neg2_table = zeros(n_run, n_contract);
rmse_yt_ukf_neg3_table = zeros(n_run, n_contract);
rmse_yt_ukf_neg4_table = zeros(n_run, n_contract);

for i = 1: n_run
    i
    rng(seed+i);
    kappa_xi = rand * 3;
    kappa_chi = rand * (3-kappa_xi) + kappa_xi; % kappa_chi > kappa_xi
    mu_xi = rand * 10 - 5;
    sigma_chi = rand * 3;
    sigma_xi = rand * 3;
    rho = rand * 2 - 1;
    lambda_chi = rand * 10 - 5;
    lambda_xi = rand * 10 - 5;
    s1 = rand * 0.1; 
    
    par_org      = [kappa_chi, kappa_xi,  mu_xi, sigma_chi, sigma_xi,  rho,  lambda_chi,  lambda_xi, s1];
    par_org_neg1 = [kappa_chi, kappa_xi,  mu_xi, sigma_chi, sigma_xi,  rho, -lambda_chi,  lambda_xi, s1];
    par_org_neg2 = [kappa_chi, kappa_xi, -mu_xi, sigma_chi, sigma_xi,  rho,  lambda_chi, -lambda_xi, s1];
    par_org_neg3 = [kappa_chi, kappa_xi, -mu_xi, sigma_chi, sigma_xi,  rho, -lambda_chi, -lambda_xi, s1];
    par_org_neg4 = [kappa_chi, kappa_xi,  mu_xi, sigma_chi, sigma_xi, -rho,  lambda_chi,  lambda_xi, s1];
    
    x0 = [ 0 ; mu_xi / kappa_xi ];     
    [yt, mats, xt] = SimYtQuadratic(par_org, x0, n_obs, n_contract, monthdays, yeardays, seed+i);
    
    [nll0_ekf, ~, xf0_ekf, ~] = EKF(par_org, yt, mats, dt);
    [nll1_ekf, ~, xf1_ekf, ~] = EKF(par_org_neg1, yt, mats, dt);
    [nll2_ekf, ~, xf2_ekf, ~] = EKF(par_org_neg2, yt, mats, dt);
    [nll3_ekf, ~, xf3_ekf, ~] = EKF(par_org_neg3, yt, mats, dt);
    [nll4_ekf, ~, xf4_ekf, ~] = EKF(par_org_neg4, yt, mats, dt);
    
    [nll0_ukf, xf0_ukf, ~] = UKF(par_org, yt, mats, dt);
    [nll1_ukf, xf1_ukf, ~] = UKF(par_org_neg1, yt, mats, dt);
    [nll2_ukf, xf2_ukf, ~] = UKF(par_org_neg2, yt, mats, dt);
    [nll3_ukf, xf3_ukf, ~] = UKF(par_org_neg3, yt, mats, dt);
    [nll4_ukf, xf4_ukf, ~] = UKF(par_org_neg4, yt, mats, dt);
    
    nll_ekf_table(i) = nll0_ekf;
    nll_ekf_neg1_table(i) = nll1_ekf;
    nll_ekf_neg2_table(i) = nll2_ekf;
    nll_ekf_neg3_table(i) = nll3_ekf;
    nll_ekf_neg4_table(i) = nll4_ekf;
    
    nll_ukf_table(i) = nll0_ukf;    
    nll_ukf_neg1_table(i) = nll1_ukf;
    nll_ukf_neg2_table(i) = nll2_ukf;
    nll_ukf_neg3_table(i) = nll3_ukf;
    nll_ukf_neg4_table(i) = nll4_ukf;
    
    p_coordinate = [0, 0, 0, 1, 0, 1]';
    G0 = [0, -lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
          0,  -kappa_chi,               0, -2*lambda_chi,     mu_xi-lambda_xi,                   0;
          0,           0,       -kappa_xi,             0,         -lambda_chi, 2*mu_xi-2*lambda_xi;
          0,           0,               0,  -2*kappa_chi,                   0,                   0;
          0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
          0,           0,               0,             0,                   0,         -2*kappa_xi];
      
    G1 = [0,  lambda_chi, mu_xi-lambda_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
          0,  -kappa_chi,               0,  2*lambda_chi,     mu_xi-lambda_xi,                   0;
          0,           0,       -kappa_xi,             0,          lambda_chi, 2*mu_xi-2*lambda_xi;
          0,           0,               0,  -2*kappa_chi,                   0,                   0;
          0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
          0,           0,               0,             0,                   0,         -2*kappa_xi];
      
    G2 = [0, -lambda_chi, lambda_xi-mu_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
          0,  -kappa_chi,               0, -2*lambda_chi,     lambda_xi-mu_xi,                   0;
          0,           0,       -kappa_xi,             0,         -lambda_chi, 2*lambda_xi-2*mu_xi;
          0,           0,               0,  -2*kappa_chi,                   0,                   0;
          0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
          0,           0,               0,             0,                   0,         -2*kappa_xi];
      
    G3 = [0,  lambda_chi, lambda_xi-mu_xi,   sigma_chi^2,                   0,          sigma_xi^2; 
          0,  -kappa_chi,               0,  2*lambda_chi,     lambda_xi-mu_xi,                   0;
          0,           0,       -kappa_xi,             0,          lambda_chi, 2*lambda_xi-2*mu_xi;
          0,           0,               0,  -2*kappa_chi,                   0,                   0;
          0,           0,               0,             0, -kappa_chi-kappa_xi,                   0; 
          0,           0,               0,             0,                   0,         -2*kappa_xi];
    G4 = G0;
    
    Hx0_ekf = [repelem(1, n_obs)', xf0_ekf, xf0_ekf(:, 1).^2, xf0_ekf(:, 1) .* xf0_ekf(:, 2), xf0_ekf(:, 2).^2];   
    Hx1_ekf = [repelem(1, n_obs)', xf1_ekf, xf1_ekf(:, 1).^2, xf1_ekf(:, 1) .* xf1_ekf(:, 2), xf1_ekf(:, 2).^2]; 
    Hx2_ekf = [repelem(1, n_obs)', xf2_ekf, xf2_ekf(:, 1).^2, xf2_ekf(:, 1) .* xf2_ekf(:, 2), xf2_ekf(:, 2).^2]; 
    Hx3_ekf = [repelem(1, n_obs)', xf3_ekf, xf3_ekf(:, 1).^2, xf3_ekf(:, 1) .* xf3_ekf(:, 2), xf3_ekf(:, 2).^2]; 
    Hx4_ekf = [repelem(1, n_obs)', xf4_ekf, xf4_ekf(:, 1).^2, xf4_ekf(:, 1) .* xf4_ekf(:, 2), xf4_ekf(:, 2).^2]; 

    Hx0_ukf = [repelem(1, n_obs)', xf0_ukf, xf0_ukf(:, 1).^2, xf0_ukf(:, 1) .* xf0_ukf(:, 2), xf0_ukf(:, 2).^2];           
    Hx1_ukf = [repelem(1, n_obs)', xf1_ukf, xf1_ukf(:, 1).^2, xf1_ukf(:, 1) .* xf1_ukf(:, 2), xf1_ukf(:, 2).^2];    
    Hx2_ukf = [repelem(1, n_obs)', xf2_ukf, xf2_ukf(:, 1).^2, xf2_ukf(:, 1) .* xf2_ukf(:, 2), xf2_ukf(:, 2).^2];    
    Hx3_ukf = [repelem(1, n_obs)', xf3_ukf, xf3_ukf(:, 1).^2, xf3_ukf(:, 1) .* xf3_ukf(:, 2), xf3_ukf(:, 2).^2];    
    Hx4_ukf = [repelem(1, n_obs)', xf4_ukf, xf4_ukf(:, 1).^2, xf4_ukf(:, 1) .* xf4_ukf(:, 2), xf4_ukf(:, 2).^2];    

    yf0_ekf = zeros(n_obs, n_contract);
    yf1_ekf = zeros(n_obs, n_contract);
    yf2_ekf = zeros(n_obs, n_contract);
    yf3_ekf = zeros(n_obs, n_contract);
    yf4_ekf = zeros(n_obs, n_contract);
    
    yf0_ukf = zeros(n_obs, n_contract);
    yf1_ukf = zeros(n_obs, n_contract);
    yf2_ukf = zeros(n_obs, n_contract);
    yf3_ukf = zeros(n_obs, n_contract);
    yf4_ukf = zeros(n_obs, n_contract);
    
    for j = 1: n_obs
        for k = 1: n_contract
            exp_G0 = Decomposition_Eigen(mats(j, k)*G0);
            exp_G1 = Decomposition_Eigen(mats(j, k)*G1);
            exp_G2 = Decomposition_Eigen(mats(j, k)*G2);
            exp_G3 = Decomposition_Eigen(mats(j, k)*G3);
            exp_G4 = Decomposition_Eigen(mats(j, k)*G4);
            
            yf0_ekf(j, k) = Hx0_ekf(j, :) * exp_G0 * p_coordinate;
            yf1_ekf(j, k) = Hx1_ekf(j, :) * exp_G1 * p_coordinate;
            yf2_ekf(j, k) = Hx2_ekf(j, :) * exp_G2 * p_coordinate;
            yf3_ekf(j, k) = Hx3_ekf(j, :) * exp_G3 * p_coordinate;
            yf4_ekf(j, k) = Hx4_ekf(j, :) * exp_G4 * p_coordinate;
            
            yf0_ukf(j, k) = Hx0_ukf(j, :) * exp_G0 * p_coordinate;
            yf1_ukf(j, k) = Hx1_ukf(j, :) * exp_G1 * p_coordinate;
            yf2_ukf(j, k) = Hx2_ukf(j, :) * exp_G2 * p_coordinate;
            yf3_ukf(j, k) = Hx3_ukf(j, :) * exp_G3 * p_coordinate;
            yf4_ukf(j, k) = Hx4_ukf(j, :) * exp_G4 * p_coordinate;
        end
    end
    rmse_yt_ekf_table(i, :)      = sqrt( mean( (yf0_ekf - yt).^2 ) );
    rmse_yt_ekf_neg1_table(i, :) = sqrt( mean( (yf1_ekf - yt).^2 ) );
    rmse_yt_ekf_neg2_table(i, :) = sqrt( mean( (yf2_ekf - yt).^2 ) );
    rmse_yt_ekf_neg3_table(i, :) = sqrt( mean( (yf3_ekf - yt).^2 ) );
    rmse_yt_ekf_neg4_table(i, :) = sqrt( mean( (yf4_ekf - yt).^2 ) );
    
    rmse_yt_ukf_table(i, :)      = sqrt( mean( (yf0_ukf - yt).^2 ) );
    rmse_yt_ukf_neg1_table(i, :) = sqrt( mean( (yf1_ukf - yt).^2 ) );
    rmse_yt_ukf_neg2_table(i, :) = sqrt( mean( (yf2_ukf - yt).^2 ) );
    rmse_yt_ukf_neg3_table(i, :) = sqrt( mean( (yf3_ukf - yt).^2 ) );
    rmse_yt_ukf_neg4_table(i, :) = sqrt( mean( (yf4_ukf - yt).^2 ) );
end

