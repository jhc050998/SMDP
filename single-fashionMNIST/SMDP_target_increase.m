function [ts, type] = SMDP_target_increase(n, to, Vo, Tthreshold, psp_m, psp_s, w, cj)
global Timeline tau_s tau_m

[peaks,locs] = findpeaks(Vo(n,:)); 
subThreshPeaks = peaks(peaks < Tthreshold);  
subThreshlocs = locs(peaks < Tthreshold);  
[Vmax, pos] = max( subThreshPeaks );  
if isempty(Vmax)
    psp_sm = - psp_m ./ tau_m + psp_s ./ tau_s;  
    ts_j = to(n,:);
    ts_j(ts_j == Inf) = [];  
    if ~isempty(ts_j)
        cj_temp = cj(ts_j); % 取出Cout
        temp= Timeline - ts_j';     % 所有的时刻都求
        temp(temp<=0) = Inf;  % 小于0的值去除—无意义
        part2 = -Tthreshold .* cj_temp'.* exp(-temp/tau_m) ./ tau_m  ;	% dV(tx)/dtsj   
        part3 = w' * psp_sm - sum( part2, 1);    %  dV（tsj）/d（tsj） 
        part3(ts_j) = -Inf ;  % 消除脉冲点影响
        [~ , t_err] = max(part3) ;
    else % 不存在前脉冲，无part2，直接求part3
        part3 = w' * psp_sm;    % 1 dV（tsj）/d（tsj） 
        part3(ts_j) = -Inf ;  % 消除脉冲点影响
        [~ , t_err] = max(part3) ;                
    end
else
    t_err = Timeline(subThreshlocs(pos));  % 最大亚阈值点对应时刻
end
    if ~isempty(to)
        temp = to(n,:);     
    else
        temp =[];
    end
    ts = [temp(temp < t_err) t_err];  % ts 包括了t_err前面的脉冲与t_err
    type = 1;
