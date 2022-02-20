function expA = Polynomial_Vandermonde(A)

lambda = eig(A);
n = size(A, 1);
v = zeros(n, n);

for i = 0: n-1
    v(i+1, :) = lambda.^i;
end

inv_v = inv(v);
expA = 0;

for j = 1: n
    Aj = 0;
    for k = 1: n
        Aj = Aj + inv_v(j, k) * A^(k-1);
    end
    expA = expA + exp(lambda(j)) * Aj;
end
