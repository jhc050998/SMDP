function [outNum, to1, to2, Vo, V0, cj] = Aug_CalOut(theta,w, psp)
global  tau_m Timeline dt
factor = exp( - dt / tau_m );
No = size(w,2);    % w的第二个维度的大小——输出层神经元数量
outNum = zeros(No,1);
to1 = [];
to2 = [];
cj = [];
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
    temp_cj = zeros(No , 1); % 记录每一时刻的脉冲系数
    while ~isempty(row)
        temp_cj(row) = temp_cj(row) + 1  ; % 对应的神经元输出+1
        % 存在动态阈值时，根据不同阈值设置输出
        if length(theta) == No  
            E_k(row) = E_k(row) + theta(row);      
        else
            E_k(row) = E_k(row) + theta;
        end            
        temp = Inf(No, 1);
        temp(row) = Timeline(k);
        to2 = [to2 temp];  % 记录每个神经元的脉冲时刻(重复
        % 再次计算k时刻新电压，观察是否还有脉冲
        Vo(:, k) = V0(:, k) - E_k;                             
        row = find( Vo(:, k) >= theta );   % 找到大于theta的神经元
    end
    temp = Inf(No, 1);
    temp(temp_cj>0) = Timeline(k); %  每个神经元中，脉冲系数大于0，则为有输出
    cj = [cj temp_cj]; % 记录每个脉冲时刻的脉冲系数
    to1 = [to1 temp];  % 记录每个神经元的脉冲时刻 
end
if ~isempty(to1)
    to1 = sort(to1,2);  % 对数组进行排序
    to2 = sort(to2,2);
    to1(:, all(to1 == Inf, 1)) = [];  %（不是消除所有inf，是消除所有都是inf的列
    to2(:, all(to2 == Inf, 1)) = [];  
    outNum = sum(cj,2);
end
