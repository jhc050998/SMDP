function [outNum, to1, to2, Vo, V0, cj] = Aug_CalOut(theta, w, psp)
% outNum-各神经元发射脉冲总数；cj-各时刻各神经元发射脉冲个数
% to1-发射情况记录，是否发射；to2-同时刻的发射分条记录；Vo-模拟的膜电位
global tau_m dt Timeline
factor = exp(-dt/tau_m ); % 时间衰减因子
No = size(w,2); % 输出神经元个数
outNum = zeros(No,1); % 记录各输出神经元发射的脉冲数
to1 = []; to2 = []; cj = [];
V0 = zeros(No,length(Timeline));
Vo = zeros(No,length(Timeline)); % (N,40)，处理完后反应真实膜电位

if ~isempty(psp) 
    V0 =  w' * psp; % 不计衰减和发射影响下单是脉冲到来引起的膜电位变化
else
    return;
end

E_k = zeros(No,1); % 模拟衰减，包括随时间衰减和发射脉冲引起的膜电位下降
count = 1;
for k = 1: length(Timeline) % 处理每个时刻，(40)
    temp_cj = zeros(No,1); % k时刻每神经元发射脉冲数量记录
    
    E_k = E_k * factor;
    Vo(:,k) = V0(:,k) - E_k; % 膜电位随时间衰减，Vo(:,k)是k时刻膜电位
    row = find(Vo(:,k) >= theta); % k时刻发射脉冲的神经元
%     count = 1;
    while ~isempty(row) % 处理每个发射对膜电位的影响
        temp_cj(row) = temp_cj(row)+1;
        if length(theta) == No % 存在动态阈值时，根据不同阈值设置输出
            E_k(row) = E_k(row) + theta(row);      
        else
%             E_k(row) = E_k(row) + theta;   %***
            E_k(row) = E_k(row) + Vo(row,k);
        end
        temp = Inf(No,1);
        temp(row) = Timeline(k); % k时刻脉冲发射情况（多个发射分别记录）
        to2 = [to2 temp];  % 各时刻各神经元的发射情况记录
        
        Vo(:,k) = V0(:,k) - E_k;                             
        row = find(Vo(:,k) >= theta);
        
%         count
        count = count + 1;
    end
    temp = Inf(No,1);
    temp(temp_cj > 0) = Timeline(k); % k时刻发射过脉冲的神经元记录
    to1 = [to1 temp];
    
    cj = [cj temp_cj]; % 各时刻各神经元发射脉冲数量记录
end

if ~isempty(to1)
    to1 = sort(to1,2); to2 = sort(to2,2);
    to1(:, all(to1 == Inf,1)) = []; to2(:, all(to2 == Inf,1)) = []; % 清除全inf列  
    outNum = sum(cj,2);
end