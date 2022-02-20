function expA = Polynomial_Lagrange(A)

lambda = eig(A);
n = size(A, 1);

expA = 0;

for j = 1: n
    prod = 1;
    for k = 1: n
        if k == j
            continue;
        end
        prod = prod * ( A - lambda(k) * eye(n) ) / ( lambda(j) - lambda(k) );
    end
    expA = expA + exp(lambda(j)) * prod;
end
