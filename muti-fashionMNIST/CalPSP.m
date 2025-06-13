function [psp_m, psp_s, psp] = CalPSP(ti, delay)  % 计算出输入部分的电压值
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
else  % 有延迟时
    t_len = length(Timeline);
    [row, column] = size(delay);  % row-第一层神经元个数 column 第二层个数
    psp_m = zeros(row, t_len, column);         
    psp_s = psp_m;     
     % ----矩阵计算，之前我的代码是循环每个脉冲比较慢，这里由于编码方式只产生一个脉冲。直接矩阵乘法----
     %循环每个输出神经元
    for j = 1: column  
        temp_ti = ti + delay(:, j);  % ti为脉冲时刻，加上delay为最终脉冲时刻 范围为0~100 超过100就不存在了
        for k = 1: size(ti, 2)  % 存在多组输入，循环每个输入
            temp = Timeline - temp_ti(:, k);
            temp(temp <= 0) = Inf; % 小于0的时刻
            psp_m(:, :, j) = psp_m(:, :, j) + exp(-temp/tau_m);
            psp_s(:, :, j) = psp_s(:, :, j) + exp(-temp/tau_s);
        end
    end
end
psp = Vnorm .* ( psp_m - psp_s );  % 计算出输入部分的电压值 231x100x2 231个数据 100个时刻 2个