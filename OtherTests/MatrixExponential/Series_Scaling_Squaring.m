function expA = Series_Scaling_Squaring(A)

norm_A = norm(A);
for i = 0: 100
    if norm_A / 2^i <= 1
        m = 2^i;
        power = i;
        break;
    end
end

expA = Series_Pade(A/m, 10, 10);

for i = 1: power
    expA = expA^2;
end

