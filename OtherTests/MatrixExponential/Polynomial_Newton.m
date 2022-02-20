function expA = Polynomial_Newton(A)

    n = size(A, 1);
    lambda = eig(A);
    expA = exp(lambda(1)) * eye(n);
    
    for j = 2: n
        prod = 1;
        for k = 1: j-1
            prod = prod * (A - lambda(k) * eye(n));
        end
        expA = expA + Divided_Difference(lambda(1: j)) * prod; 
    end
        
end

function dd = Divided_Difference(lambda)

    if length(lambda) == 2
        l1 = lambda(1);
        l2 = lambda(2);
        dd = ( exp(l1) - exp(l2) ) / (l1 - l2);
    else
        dd = ( Divided_Difference(lambda(1: end-1)) - Divided_Difference(lambda(2: end)) ) / ( lambda(1) - lambda(end) );    
    end
    
end