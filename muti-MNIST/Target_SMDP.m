function [ts, type] = Target_SMDP(n, label, outNum, to, Vo, Tthreshold, psp_m, psp_s, w, cj)
global tau_s tau_m Timeline Nclass
if mod(n-label, Nclass) % n神经元编号与标签不对应情况（发射应减少）
    if outNum(n) == 0 % 当前发射数为0，不能再减少
        ts = []; type = 0;
    %---------------------减少脉冲，减掉最后一个发射---------------------%
    elseif outNum(n) > 10 % 发射数大于10时，可同时减少多个
        temp_to = to(n,:); temp_to(temp_to == Inf) = []; % n神经元发射时刻 
        ts = inf(2, length(temp_to));
        ts(1,:)= temp_to(:); ts(2,1:end-1)= temp_to(1:end-1); % 多减去一个
        type = -1; 
    else % 减少单个
        ts = to(n,:); ts(ts == Inf) = []; % 此处ts即n神经元发射时刻记录 
        type = -1;  
    end
    %---------------------增加脉冲，增加最大亚阈值点---------------------%
else % n神经元编号与标签对应情况（发射应增加）
    if outNum(n) < 3 % 当前发射小于3时增加（至多发射3个）
        [peaks,locs] = findpeaks(Vo(n,:)); % 找到所有峰
        subThreshPeaks = peaks(peaks < Tthreshold); % 找到所有亚阈值峰 
        subThreshlocs = locs(peaks < Tthreshold);  
        [Vmax, pos] = max(subThreshPeaks ); % 最大亚阈值点位置及峰值  
        
        if isempty(Vmax) % 不存在亚阈值点时，选用最大斜率点
            psp_sm = - psp_m ./ tau_m + psp_s ./ tau_s;  
            ts_j = to(n,:); ts_j(ts_j == Inf) = []; % n神经元发射时刻记录 
            if ~isempty(ts_j)    
                cj_temp = cj(ts_j); % 各发射时刻发射数量
                temp = Timeline - ts_j';
                temp(temp<=0) = Inf; % 小于0的值去除—无意义
                part2 = -Tthreshold .* cj_temp' .* exp(-temp/tau_m) ./ tau_m; % dV(tx)/dtsj   
                part3 = w' * psp_sm - sum(part2, 1); % dV(tsj)/d(tsj) 
                part3(ts_j) = -Inf; % 消除脉冲点影响
                [~ , t_err] = max(part3) ;
            else % 不存在前脉冲，无part2，直接求part3
                part3 = w' * psp_sm; % dV(tsj)/d(tsj)
                part3(ts_j) = -Inf; % 消除脉冲点影响
                [~ , t_err] = max(part3) ;                
            end
        else
            t_err = Timeline(subThreshlocs(pos)); % 最大亚阈值点对应时刻
        end
        
        if ~isempty(to)
            temp = to(n,:);     
        else
            temp =[];
        end
        ts = [temp(temp < t_err) t_err];  % ts包括了t_err前面的脉冲与t_err
        type = 1;
        
    else % 发射已达到3个时不再增加
        ts = [];      
        type = 0;
    end
end