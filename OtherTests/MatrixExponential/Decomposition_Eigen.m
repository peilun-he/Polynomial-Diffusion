function expA = Decomposition_Eigen(A)

[V, D] = eig(A);
expA = V * diag(exp(diag(D))) * inv(V);

