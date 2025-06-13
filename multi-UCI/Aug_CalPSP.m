function [psp_m, psp_s, psp] = Aug_CalPSP(ti, delay)  % 计算出输入部分的电压�?
global tau_m tau_s Vnorm Timeline nargin
if nargin == 0
    psp_m = zeros(size(ti,1), length(Timeline));          
    psp_s = psp_m;          
    for k = 1: size(ti, 2)
        temp = Timeline - ti(:, k);
        temp(temp <= 0) = Inf;
        psp_m = psp_m + exp( -temp / tau_m );
        psp_s = psp_s + exp( -temp / tau_s );
    end
else  %  nargin =1  有延迟时
    t_len = length(Timeline);
    [row, column] = size(delay);  
    psp_m = zeros(row, t_len, column);         
    psp_s = psp_m;     
    for j = 1: column  
        temp_ti = ti + delay(:, j); 
        for k = 1: size(ti, 2) 
            temp = Timeline - temp_ti(:, k);
            temp(temp <= 0) = Inf; 
            psp_m(:, :, j) = psp_m(:, :, j) + exp(-temp/tau_m);
            psp_s(:, :, j) = psp_s(:, :, j) + exp(-temp/tau_s);
        end
    end
end
psp = Vnorm .* ( psp_m - psp_s ); 