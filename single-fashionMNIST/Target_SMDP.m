function [ts, type] = Target_SMDP(n, label, outNum, to, Vo, Tthreshold, psp_m, psp_s, w, cj)
global Timeline Nclass tau_s tau_m
if mod(n - label, Nclass)     % not target neuron （只要不是0正负数就会运行下列代码
    if outNum( n) == 0  % =0则达到期望，不需要修改 type=0
        ts = [];      
        type = 0;
    %---------------------减少脉冲，减掉最后一个--------------------------------%
    elseif outNum( n) > 10   % 脉冲过多，同时减少多个
        temp_to = to(n,:);
        temp_to(temp_to == Inf) = [];  
        ts = inf(2, length(temp_to));
        ts(1,:)= temp_to(:);
        ts(2,1:end-1)= temp_to(1:end-1);
        type = -1; 
    else    % 脉冲差距小，减少单个
        ts = to(n,:);
        ts(ts == Inf) = [];  
        type = -1;  
    end
%---------------------增加脉冲，增加最大亚阈值点--------------------------------%
else  %  期望此神经元发出脉冲
    if outNum(n) < 3 % 训练脉冲发出至少3个脉冲，超过则不训练
        % 找最大亚阈值点
        [peaks,locs] = findpeaks(Vo(n,:)); 
        subThreshPeaks = peaks(peaks < Tthreshold);  
        subThreshlocs = locs(peaks < Tthreshold);  
        [Vmax, pos] = max( subThreshPeaks );  
        if isempty(Vmax)    % 不存在亚阈值点时，选用最大斜率点
            psp_sm = - psp_m ./ tau_m + psp_s ./ tau_s;  
            ts_j = to(n,:);
            ts_j(ts_j == Inf) = [];  
            if ~isempty(ts_j)    
                cj_temp = cj(ts_j); % 取出Cout
                temp= Timeline - ts_j';    
                temp(temp<=0) = Inf;  % 小于0的值去除—无意义
                part2 = -Tthreshold .* cj_temp'.* exp(-temp/tau_m) ./ tau_m  ;	%  dV(tx)/dtsj   
                part3 = w' * psp_sm - sum( part2, 1);    %  dV（tsj）/d（tsj） 
                part3(ts_j) = -Inf ;  % 消除脉冲点影响
                [~ , t_err] = max(part3) ;
            else % 不存在前脉冲，无part2，直接求part3
                part3 = w' * psp_sm;    %  dV（tsj）/d（tsj） 
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
    else 
        ts = [];      
        type = 0;
    end
end