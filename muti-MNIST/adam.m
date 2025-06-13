function [deltW, m, v] = adam(deriv, epoch, m, v, sign)
beta1 = .9; beta2 = .999; epsilon = 10e-8;
g = sign .* deriv;
m = beta1*m + (1-beta1)*g; v = beta2*v + (1-beta2)*g.^2;
mbar = m / (1-beta1^epoch); vbar = v / (1-beta2^epoch);
deltW = mbar ./ (sqrt(vbar)+epsilon);
end
