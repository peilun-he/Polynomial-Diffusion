function expA = Series_Pade(A, p, q)

N = 0;
D = 0;

for i = 0: p
    N = N + factorial(p+q-i) * factorial(p) / factorial(p+q) / factorial(i) / factorial(p-i) * A^i;
end

for j = 0: q
    D = D + factorial(p+q-j) * factorial(q) / factorial(p+q) / factorial(j) / factorial(q-j) * (-A)^j;
end

expA = inv(D) * N; 