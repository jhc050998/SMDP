function [ts, type] = Target_SMDP(n, label, outNum, to, Vo, Tthreshold, psp_m, psp_s, w, cj)
global tau_s tau_m Timeline Nclass
if mod(n-label, Nclass) % n��Ԫ������ǩ����Ӧ���������Ӧ���٣�
    if outNum(n) == 0 % ��ǰ������Ϊ0�������ټ���
        ts = []; type = 0;
    %---------------------�������壬�������һ������---------------------%
    elseif outNum(n) > 10 % ����������10ʱ����ͬʱ���ٶ��
        temp_to = to(n,:); temp_to(temp_to == Inf) = []; % n��Ԫ����ʱ�� 
        ts = inf(2, length(temp_to));
        ts(1,:)= temp_to(:); ts(2,1:end-1)= temp_to(1:end-1); % ���ȥһ��
        type = -1; 
    else % ���ٵ���
        ts = to(n,:); ts(ts == Inf) = []; % �˴�ts��n��Ԫ����ʱ�̼�¼ 
        type = -1;  
    end
    %---------------------�������壬�����������ֵ��---------------------%
else % n��Ԫ������ǩ��Ӧ���������Ӧ���ӣ�
    if outNum(n) < 3 % ��ǰ����С��3ʱ���ӣ����෢��3����
        [peaks,locs] = findpeaks(Vo(n,:)); % �ҵ����з�
        subThreshPeaks = peaks(peaks < Tthreshold); % �ҵ���������ֵ�� 
        subThreshlocs = locs(peaks < Tthreshold);  
        [Vmax, pos] = max(subThreshPeaks ); % �������ֵ��λ�ü���ֵ  
        
        if isempty(Vmax) % ����������ֵ��ʱ��ѡ�����б�ʵ�
            psp_sm = - psp_m ./ tau_m + psp_s ./ tau_s;  
            ts_j = to(n,:); ts_j(ts_j == Inf) = []; % n��Ԫ����ʱ�̼�¼ 
            if ~isempty(ts_j)    
                cj_temp = cj(ts_j); % ������ʱ�̷�������
                temp = Timeline - ts_j';
                temp(temp<=0) = Inf; % С��0��ֵȥ����������
                part2 = -Tthreshold .* cj_temp' .* exp(-temp/tau_m) ./ tau_m; % dV(tx)/dtsj   
                part3 = w' * psp_sm - sum(part2, 1); % dV(tsj)/d(tsj) 
                part3(ts_j) = -Inf; % ���������Ӱ��
                [~ , t_err] = max(part3) ;
            else % ������ǰ���壬��part2��ֱ����part3
                part3 = w' * psp_sm; % dV(tsj)/d(tsj)
                part3(ts_j) = -Inf; % ���������Ӱ��
                [~ , t_err] = max(part3) ;                
            end
        else
            t_err = Timeline(subThreshlocs(pos)); % �������ֵ���Ӧʱ��
        end
        
        if ~isempty(to)
            temp = to(n,:);     
        else
            temp =[];
        end
        ts = [temp(temp < t_err) t_err];  % ts������t_errǰ���������t_err
        type = 1;
        
    else % �����Ѵﵽ3��ʱ��������
        ts = [];      
        type = 0;
    end
end