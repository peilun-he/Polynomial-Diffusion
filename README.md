# Polynomial-Diffusion
Crude oil futures (or other futures) modelling by Schwartz & Smith two factor model (SS2000) and polynomial diffusion (PD) model. Please change values under section "Global setups 1" and "Global setups 2" ONLY. 

Author: Peilun He

Last update: 20 Feb 2022

## Descriptions of files & folders 
Main.m: the main function for SS2000 model and PD model. 

Functions: all relevant functions. 

OtherTests: some functions for other purposes, not directly relevant to SS2000 model and PD model. 

## Descriptions of global setups
model: define which model to use, "SS2000" or "PD".
  SS2000 -> Schwartz & Smith two factor model
  PD -> polynomial diffusion model

exchange: the exchange where the futures traded, "NY" or any other string. If the exchange is located in New York, e.g., New York Stock Exchange, New York Mercantile Exchange, the MATLAB built-in function "holidays" will be used to get all non-trading days and then get all time to maturities (TTM). Otherwise, users must provide TTM by themselves. Be careful about the values in "mat" or "mat0", it could contains negative values. 

noise: the distribution of noise. "Gaussian" or "Gamma" (I didn't actually test Gamma noise, so please just use "Gaussian"). **For PD model only**. 

n_coe: the number of model coefficients to be estimated. For example, for PD model with degree 2, if n_coe = 0, the spot price is defined as: 
S_t = 1 + chi_t + xi_t + 1/2 chi_t^2 + chi_t xi_t + 1/2 xi_t^2
If n_coe = 6, the spot price is defined as: 
S_t = a1 + a2 chi_t + a3 xi_t + a4 chi_t^2 + a5 chi_t xi_t + a6 xi_t^2
where a1, ..., a6 are the coefficients to be estimated. "n_coe" must be defined correctly based on the degree of PD model. 
**For PD model only**. 

degree: the degree of PD model. **For PD model only**. 

filter: the filtering method. "@UKF3" for Unscented Kalman Filter and "@EKF3" for Extended Kalman Filter. **For PD model only**. 

contracts_est: the contracts used for in-sample estimation. 

contracts_fore: the contracts used for out-of-sample forecasting. 

first_est, last_est: define the time period for in-sample estimation. 

first_fore, last_fore: define the time period for out-of-sample forecasting. 

n_gird: number of grid points. "2" or "3". 

n_para: number of parameters for grid search. Usually equal to 8, for kappa, gamma, mu_xi, sigma_chi, sigma_xi, rho, lambda_chi, lambda_xi. 

n_se: number of standard errors in measurement equation. Usually equal to the number of contracts. 

func_f: define the state equation, which should take two arguments, state variables x_t and a vector of parameters, and return two values, f(x) and f'(x). **For PD model only**. 

func_g: define the measurement equation, which should take three arguments, state variables x_t, a vector of parameters and maturities, and return two values, g(x) and g'(x). **For PD model only**. 

## References






