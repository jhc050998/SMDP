function [ts, type] = Target_SMDP_repeat(n, label, outNum, to, Vo, Tthreshold, psp_m, psp_s, w)
global Timeline Nclass 
if mod(n - label, Nclass)     % not target neuron （只要不是0正负数就会运行下列代码
    if outNum( n) == 0  % =0则达到期望，不需要修改 type=0
        ts = [];      
        type = 0;
    %---------------------减少脉冲，减掉最后一个--------------------------------%
    elseif outNum(n) > 10   % 脉冲过多，同时减少多个
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
        if isempty(Vmax)    % 不存在亚阈值点时，选除脉冲点外的最大值点
            V_temp = Vo(n,:);
            V_temp(V_temp < Tthreshold) = 0 ; %脉冲点为0，排除影响
            [~, pos] = max( V_temp );
            t_err = pos; 
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