clear; 
close all;

load('./MNIST.mat'); 
% 读到内容：test_labels-10000*1,test_pattern-784*10000,
% train_labels-60000*1,train_pattern-784*60000

train_labels(train_labels == 0) = 10; % 将标签中的0都改为10 (60000,1)
test_labels(test_labels == 0) = 10;
train_pattern = 8*train_pattern; % 数据值扩大8倍 (784,60000)
test_pattern = 8*test_pattern;
% 标签是直接的数字，数据是处理过的，黑色背景值均为Inf

nTrain = 1;%length(train_labels); % 60000
nTest = length(test_labels); % 10000
% 训练与测试集样本个数

global threshold tau_m tau_s Vnorm T dt Timeline nargin groupSize scaleRate Nclass
threshold = 1; tau_m = 20; tau_s = tau_m/4; % 阈值，脉冲常数
beta = tau_m/tau_s; Vnorm = beta^(beta/(beta-1))/(beta-1); % 膜电位归一化
T = 40; dt = 1; Timeline = dt: dt: T; % 时间，(1,40)
nargin = 0; groupSize = 1; scaleRate = 0.1; Nclass = 10;

dropRate = 0.1; sigma = 0.1;
Ni = 784; Nh = 800; No = Nclass*groupSize; % 网络结构：784-800-10
maxEpoch = 1;%200; % 训练次数
lr_w_o = [.0005 .0003 .0001]; lr_w_h = lr_w_o; % 隐含层及输出层学习率
w_mh = zeros(Ni, Nh);  w_vh = w_mh; % 两层的adam参数
w_mo = zeros(Nh, No);  w_vo = w_mo;
maxTrain = zeros(length(lr_w_o),1); maxTest = zeros(length(lr_w_o),1); % 存准确率
max_w_o = cell(length(lr_w_o),1); max_w_h = cell(length(lr_w_o),1);% 存权重

for lr = 1: length(lr_w_o) % 3个不同的学习率各跑一次
    w_h = unifrnd(-0.15, 0.2, Ni, Nh); % 两层的权重初始化 (784,800),(800,10)
    w_o = unifrnd(-0.05, 0.06, Nh, No);
    for epoch = 1: maxEpoch % 1-200循环（200次训练）
        fprintf('lr = %d， epoch = %d\n', lr, epoch); % *有错
        numOrder = randperm(nTrain); % 将训练集随机打乱顺序
        for i = 1: nTrain % 1-60000循环（逐一使用训练集样本）
            if mod(i,10) == 0
                fprintf('i = %d\n', i);
            end
            ti = train_pattern(:,numOrder(i)); % (784,1)，输入单脉冲编码
            
            % 前向传播
            [psp_m_i, psp_s_i, psp_i] = Aug_CalPSP(ti); % 输入脉冲的膜电位影响
            [outNumh, th1, th2, Vh, ~, cj_h] = Aug_CalOut(threshold, w_h, psp_i);
            
            [psp_m_h, psp_s_h, psp_h] = Aug_CalPSP(th2);           
            [outNum, to1, ~, Vo, ~, cj_o] = Aug_CalOut(threshold, w_o, psp_h);
                        
            % 反向传播
            derivw_o = zeros(Nh, No); derivw_h = zeros(Ni, Nh); % 梯度初始化
            type = zeros(1, No);
            for k = 1:No % 循环各输出层神经元
                [ts, type(k)] = Target_SMDP(k, train_labels(numOrder(i)), outNum, to1, Vo, threshold, psp_m_h, psp_s_h, w_o(:,k), cj_o(k,:));
                if type(k) == 0 % 不需改变发射情况的情形
                    continue;
                end
                for ts_number = 1:size(ts,1) % ts_number>1时有同时减少多个操作
                    [derivw_o_temp, ~, part2, part3, part4] = bp_output(cj_o, ts(ts_number,:) , psp_m_h, psp_s_h, scaleRate, threshold, w_o(:,k), th1);
                    [~, ~, ~, ~, ~, subparth_w] = bp_output(cj_h, th1, psp_m_i, psp_s_i, scaleRate, threshold, w_h, []);
                    [derivw_h_temp] = bp_hidden(Ni, Nh, scaleRate, part2, part3, part4, subparth_w);
                    
                    derivw_o(:,k) = derivw_o(:,k) + derivw_o_temp; % 输出层梯度
                    derivw_h = derivw_h + type(k) .* derivw_h_temp; % 隐含层梯度
                end
            end
            
            % 基于梯度，用adam的方式调权重
            [deltWh, w_mh, w_vh] = adam(derivw_h, (epoch-1)*nTrain + i, w_mh, w_vh, 1);
            [deltWo, w_mo, w_vo] = adam(derivw_o, (epoch-1)*nTrain + i, w_mo, w_vo, type);
            w_h = w_h + lr_w_h(lr) .* deltWh;
            w_o = w_o + lr_w_o(lr) .* deltWo;
        end
        
%         size(test_pattern)
        numCorrect = Aug_testing(test_pattern(:, 1:nTest), test_labels(1:nTest), w_h, w_o);
        
    end
end











