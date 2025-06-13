function [outNum, to, Vo, V0] = Aug_CalOutVo(theta,w, psp)
global  tau_m Timeline dt
factor = exp( - dt / tau_m );
No = size(w,2);    % w的第二个维度的大小——输出层神经元数量
outNum = zeros(No,1);
to = [];
Vo = zeros(No, length(Timeline));  
if ~isempty(psp) 
    if ndims(psp) == 3  % 有延迟 求出来是三个维度，需要改变一下顺序
        psp = permute(psp,[1 3 2]); % permute交换psp第二第三维度
        V0 = squeeze(sum(w .* psp, 1)); % sum(a,dim); a表示矩阵；dim等于1或者2，1表示每一列进行求和，2表示每一行进行求和；   squeeze：去掉1维的向量；  ——VO最终为每个神经元的每时刻电压值
    else
        V0 =  w' * psp;
    end
else
    return;
end

E_k = zeros(No, 1);
 % 循环每个时刻
for k = 1: length(Timeline)
    E_k = E_k * factor;  % 神经元后一项，每个输出随着时间衰减
    Vo(:, k) = V0(:, k) - E_k;                             
    row = find( Vo(:, k) >= theta );   % 找到大于theta的神经元
    while ~isempty(row)
        % 存在动态阈值时，根据不同阈值设置输出
        if length(theta) == No  
            E_k(row) = E_k(row) + theta(row);      
        else
            E_k(row) = E_k(row) + theta;
        end            
        temp = Inf(No, 1);
        temp(row) = Timeline(k);
        to = [to temp];  % 记录每个神经元的脉冲时刻(重复
        %% 多了这两步，检查多余脉冲
        Vo(:, k) = V0(:, k) - E_k;   % 再次计算k时刻新电压，观察是否还有脉冲                           
        row = find( Vo(:, k) >= theta );   % 再次检查是否还有脉冲
        % temp_Vo = V0(:, k) - E_k;   % 再次计算k时刻新电压，观察是否还有脉冲                           
        % row = find( temp_Vo >= theta );   % 再次检查是否还有脉冲
    end
end
if ~isempty(to)
    to = sort(to,2);  % 对数组进行排序
    to(:, all(to == Inf, 1)) = [];  %（不是消除所有inf，是消除所有都是inf的列
    outNum = sum(to ~= Inf, 2); % 记录脉冲的总数量（to保留了
end
