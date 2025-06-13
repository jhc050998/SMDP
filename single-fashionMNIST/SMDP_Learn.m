function [deriv_w] = SMDP_Learn(Tthreshold, w, ts, PSP_m, PSP_s, scaleRate,cj)
global tau_m tau_s dt Vnorm
ts(ts == 0) =[];	
ts(ts == Inf) =[];
ts_j = ts(1: end-1);    % tsj：最后一项之前
column = int32(ts ./ dt);
psp_m = Vnorm .* PSP_m(:, column);	% ts时刻对应的psp_m
psp_s = Vnorm .* PSP_s(:, column);	
psp = psp_m - psp_s;   
partial1_w = psp;   % dV(tx)/dwi 
psp_sm = psp_m ./ tau_m - psp_s ./ tau_s;  % dV（tx）/dtx 的输入部分
if isempty(ts_j)  
    deriv_w = partial1_w;
    return
end
cj_temp = cj(ts_j); % 取出Cout
temp= ts - ts_j';     
temp(temp<=0) = Inf;  % 小于0的值去除 无意义
partial2 = -Tthreshold .* cj_temp'.* exp(-temp/tau_m) ./ tau_m  ;	% dV(tx)/dtsj    
partial3 = - w' * psp_sm - sum( partial2, 1);    % dV（tsj）/d（tsj） 
partial3(partial3 <= 0) = inf;  
partial3(partial3 <= 0.5) = 0.5; 
deriv_w = partial1_w(:, end) - scaleRate .* partial1_w(:, 1: end-1) * ( partial2(:, end) ./ partial3(1: end-1)' );
