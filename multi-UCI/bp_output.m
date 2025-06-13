function [deriv_w, deriv_d, part1_w, part1_d, part2, part3, part4, subpart_w, subpart_d] = bp_output(cj, ts, PSP_m, PSP_s, scaleRate,Tthreshold, w, delay, th)
global tau_m tau_s Vnorm dt

tmax = tau_s * tau_m * log(tau_s/tau_m) / (tau_s -tau_m);  
[Ni, No] = size(w);
deriv_w = zeros(Ni, No);
deriv_d = zeros(Ni, No);
part1_w = cell(No,1); 
part1_d = cell(No,1);
part2 = cell(No,1);
part3 = cell(No,1);
subpart_w = cell(No,1);
subpart_d = cell(No,1);
part4 = cell(Ni, No);

for k = 1: No   
    t = ts(k, :);	
    t(t == 0) =[];	
    t(t == Inf) =[];
    if isempty(t)
        continue;
    end
    n = length(t);	% 输出脉冲个数
    ts_j = t(1: end-1);     
    col = int32(t ./ dt);
    if ndims(PSP_m) == 3 %3 
        psp_m = squeeze(Vnorm .* PSP_m(:, col, k));	 % squeeze去掉只有�?维度�?  取出ts的psp
        psp_s = squeeze(Vnorm .* PSP_s(:, col, k));	
    else
        psp_m = Vnorm .* PSP_m(:, col);
        psp_s = Vnorm .* PSP_s(:, col);
    end
    psp = psp_m - psp_s;  % 输入的部�?
    psp_sm = - psp_m ./ tau_m + psp_s ./ tau_s;  
    
    if n == 1  % 脉冲输出�?1
        partial2 = [];
        partial3 =  w(:,k)' * psp_sm;    
        partial3(partial3 <= 0) = inf;    
        partial3(partial3 <= 0.1) = 0.1;
    else    % 多个脉冲输出
        cj_temp = cj(k,ts_j); % 取出Cout
        temp= t - ts_j';     
        temp(temp<=0) = Inf;  % 小于0的�?�去�? 无意�?
        partial2 = -Tthreshold .* cj_temp'.* exp(-temp/tau_m) ./ tau_m  ;	% dV(tx)/dtsj   
        part2{k} = partial2;	% dV(tx)/dtsj   
        partial3 = w(:,k)' * psp_sm - sum( partial2, 1);    % dV（tsj�?/d（tsj�? 
        partial3(partial3 <= 0) = inf;    
        partial3(partial3 <= 0.1) = 0.1; %  防止梯度爆炸
    end
    part3{k} = partial3;
    
    partial1_w = psp;   % dV（tsj�?/d wi    
    part1_w{k} = partial1_w;
    [deriv_w(:, k), subderiv_w] =  kk(Ni, n, scaleRate, partial1_w, partial2, partial3);
    subpart_w{k} = subderiv_w;
    if ~isempty(delay)  % 存在延迟�?
        partial1_d = - w(:,k) .* psp_sm;   
        part1_d{k} = partial1_d; 
        [deriv_d(:, k), subderiv_d] =  kk(Ni, n, scaleRate, partial1_d, partial2, partial3);
        subpart_d{k} = subderiv_d;
    end
      
    %%  用来计算隐藏层的相关梯度
    %%  求dV(t3)/dt2 
     if ~isempty(th)    % th：上�?层的输入  ti：这�?层的输出
         for h = 1: size(w,1)  
             if ~isempty(delay)
                 ti = th(h, :)' + delay(h, k);  
             else
                 ti = th(h, :)';
             end
             ti(ti == 0) = [];       
             ti(ti == Inf) =[];
             if isempty(ti)
                 continue;
             end
             temp = t - ti; 
             temp(temp <= 0) = Inf;    
             temp(temp > tmax) = Inf;   
             %  dV(tx)/dt2
             partial4 = Vnorm * w(h, k) .* ( exp( - temp/tau_m) ./ tau_m - exp( - temp/tau_s) ./ tau_s ); 
             part4{h,k} = partial4;  
         end
     end
     
end
end
 % partial1:dV（tx�?/d(wi)   partial2: dV(tx)/dtsj   partial3: dV（tx�?/d（tx�?
function [deriv, subderiv] = kk(Ni, n, scaleRate, partial1, partial2, partial3)
if n == 1  
    subderiv = -1 ./ partial3 .* partial1; %  dt2/dw12
    deriv = partial1(:, end);  
else
    subderiv = -1 ./ partial3   .*  partial1;  % -(d（tx�?/dV（tx�?) * dV（tx�?/d(wi) = dt3/dV（t3�?* d（t3�?/d(wi)  =dt3/d(wi)
    deriv = partial1(:,end) + scaleRate .* subderiv(:, 1:n-1) * partial2(:, end);  
end
end