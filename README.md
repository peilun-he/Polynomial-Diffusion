# Polynomial-Diffusion
Crude oil futures (or other futures) modelling by Schwartz & Smith two factor model (SS2000) and polynomial diffusion (PD) model. 

### Descriptions of some global setups
model: define which model to use, "SS2000" or "PD".
  SS2000 -> Schwartz & Smith two factor model
  PD -> polynomial diffusion model

exchange: the exchange where the futures traded, "NY" or any other string. If the exchange is located in New York, e.g., New York Stock Exchange, New York Mercantile Exchange, the MATLAB built-in function "holidays" will be used to get all non-trading days and then get all time to maturities (TTM). Otherwise, users must provide TTM by themselves. Be careful about the values in "mat" or "mat0", it could contains negative values. 

noise: the distribution of noise. "Gaussian" or "Gamma" (I didn't actually test Gamma noise, so please just use "Gaussian"). # For PD model only #. 

### References
