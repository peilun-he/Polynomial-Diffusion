function expA = Series_Taylor(A, maxIter, tolerence)

expA1 = zeros(size(A));
expA2 = zeros(size(A));

for i = 1: maxIter
    expA1 = expA1 + A^i / factorial(i);
    if all(abs(expA1 - expA2) < tolerence, 'all')
        break;
    else
        expA2 = expA1;
    end
end

expA = expA1;