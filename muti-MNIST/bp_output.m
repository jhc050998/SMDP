function [deriv_w, part1, part2, part3, part4, subpart_w] = bp_output(cj, ts, PSP_m, PSP_s, scaleRate,Tthreshold, w, th)
global tau_m tau_s Vnorm dt

tmax = tau_s * tau_m * log(tau_s/tau_m) / (tau_s -tau_m);  
[Ni, No] = size(w);
deriv_w = zeros(Ni, No);

part1 = cell(No,1); part2 = cell(No,1); part3 = cell(No,1);
subpart_w = cell(No,1); part4 = cell(Ni, No);

for k = 1: No % 循环各输出神经元  
    t = ts(k, :);
    t(t == 0) =[]; t(t == Inf) =[];
    if isempty(t)
        continue;
    end
    n = length(t); % 输出脉冲个数
    ts_j = t(1: end-1); % 除去最后一个
%     size(t)
%     size(ts_j)
    
    col = int32(t ./ dt); % 把t由时间变为序号
    psp_m = Vnorm .* PSP_m(:, col); psp_s = Vnorm .* PSP_s(:, col);
    psp = psp_m - psp_s; % 输入的部分
    psp_sm = - psp_m ./ tau_m + psp_s ./ tau_s;
    
    if n == 1 % 脉冲输出为1
        partial2 = [];
        partial3 =  w(:,k)' * psp_sm;
        partial3(partial3 <= 0) = inf; partial3(partial3 <= 0.1) = 0.1; % 防止梯度爆炸
    else % 脉冲输出多于1
        cj_temp = cj(k,ts_j); % 在ts_j各时刻发射脉冲个数
        temp = t - ts_j'; temp(temp<=0) = Inf;
%         size(temp)
        
        partial2 = -Tthreshold .* cj_temp' .* exp(-temp/tau_m) ./ tau_m; % dV(tx)/dtsj   
        part2{k} = partial2;
        
%         TT = w(:,k)' * psp_sm;
%         size(TT)
%         size(sum(partial2, 1))
        
        partial3 = w(:,k)' * psp_sm - sum(partial2, 1); % dV(tsj)/d(tsj)
%         size(partial3)
%         fprintf("bbb")
        partial3(partial3 <= 0) = inf; partial3(partial3 <= 0.1) = 0.1;
    end
    part3{k} = partial3;
    
    partial1 = psp; % dV(tsj)/dw    
    part1{k} = partial1;
    [deriv_w(:, k), subderiv_w] =  kk(Ni, n, scaleRate, partial1, partial2, partial3);
    subpart_w{k} = subderiv_w;
    
    % 计算隐含层的相关梯度 dV(t3)/dt2 
     if ~isempty(th) % th-上层输入，ti-这层输出
         for h = 1: size(w,1)
             ti = th(h, :)'; ti(ti == 0) = []; ti(ti == Inf) =[];
             if isempty(ti)
                 continue;
             end
             temp = t - ti; temp(temp <= 0) = Inf; temp(temp > tmax) = Inf;   
             %  dV(tx)/dt2
             partial4 = Vnorm * w(h, k) .* (exp(-temp/tau_m)./tau_m - exp(-temp/tau_s)./tau_s); 
             part4{h,k} = partial4;
         end
     end
end
end

% partial1:dV(tx)/d(wi);  partial2:dV(tx)/dtsj;  partial3:dV(tx)/d(tx)
function [deriv, subderiv] = kk(~, n, scaleRate, partial1, partial2, partial3)
if n == 1  
    subderiv = -1 ./ partial3 .* partial1; %  dt2/dw12
    deriv = partial1(:,end);  
else
    subderiv = -1 ./ partial3 .*  partial1; % -[d(tx)/dV(tx)] * dV(tx)/d(wi) = dt3/dV(t3) * d(t3)/d(wi) = dt3/d(wi)
    deriv = partial1(:,end) + scaleRate .* subderiv(:, 1:n-1) * partial2(:, end);  
end
end