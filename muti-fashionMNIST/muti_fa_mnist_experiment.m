%% 多层MNIST
clear 
close all
load('./FashionMNIST.mat')
train_labels(train_labels == 0) = 10; 
test_labels(test_labels == 0) = 10;
train_pattern = 8*train_pattern;
test_pattern = 8*test_pattern;
nTrain = length(train_labels);
nTest = length(test_labels);
global threshold tau_m tau_s Vnorm T dt Timeline nargin groupSize scaleRate Nclass       
groupSize = 1;  % 输出神经元组�?
nargin = 0 ;         
threshold = 1;  
tau_m = 20;          tau_s = tau_m/4;          beta = tau_m/tau_s;
Vnorm = beta^( beta /(beta-1) ) / (beta-1);
dropRate = 0.1;
scaleRate = 0.1 ;   
sigma = 0.1;
T = 40;         dt = 1;          Timeline = dt: dt: T; 
Nclass = 10;
Ni = 784;              
Nh = 800;
No = Nclass*groupSize;   
    %% 学习�?
learnRate_w_o = .00005  ;  
learnRate_w_h = learnRate_w_o;    % 中间层学习率
maxEpoch = 200;   
% adam参数
w_mh = zeros(Ni, Nh);  w_vh = w_mh;
w_mo = zeros(Nh, No);  w_vo = w_mo;
 %% 保留数据
maxTrain = zeros(length(learnRate_w_o),1); % 存准确率
maxTest = zeros(length(learnRate_w_o),1);
max_w_o = cell(length(learnRate_w_o),1); % 存权�?
max_w_h = cell(length(learnRate_w_o),1);
%%  训练 MTP
for learnRate = 1: length(learnRate_w_o) % 循环学习�?
    % 初始化权值，取出数据标签
    w_h = unifrnd(-0.15, 0.2, Ni, Nh);     
    w_o = unifrnd(-0.05, 0.06, Nh, No); 
%     w_h = normrnd(0.2, 0.2, Ni, Nh);     % 正�?�分�?
%     w_o = normrnd(0.001, 0.001, Nh, No); 
    for epoch = 1: maxEpoch
        fprintf('learnRate = %d, epoch = %d\n', learnRate, epoch); 
        numOrder = randperm(nTrain);
        for i = 1: nTrain
            if mod(i,1000) == 0
                fprintf('i = %d\n', i);
            end
            %% 加入噪声
            ti = train_pattern( : , numOrder(i)); 
            ti = ti + normrnd( 0, 0.05, size(ti) );  ti( ti < 0 ) = Inf; 
            % labels = train_label(numOrder(i));
            %% 前向传播
            [psp_m_i, psp_s_i, psp_i] = Aug_CalPSP(ti,[]);
            [outNumh, th1] = Aug_CalOutVo (threshold, w_h, psp_i); 
            dropNeuron = find(rand(size(th1,1),1) < dropRate);  % dropout�?
            if isempty(th1)
                continue;
            else
                th1(dropNeuron, :) = Inf;
            end
            [psp_m_h, psp_s_h, psp_h] = Aug_CalPSP(th1,[]);           
            [outNum, to, Vo] = Aug_CalOutVo (threshold, w_o, psp_h);  
            %% 后向梯度计算
            derivw_o = zeros(Nh, No);   % 输出层梯度（单次梯度�?
            derivw_h = zeros(Ni, Nh);   % 隐藏层梯�?
            type = zeros(1, No);
            for k = 1: No
                [ts, type(k)] = Target_SMDP(k, train_labels(numOrder(i)), outNum, to, Vo, threshold,psp_m_h, psp_s_h, w_o(:,k));
                if type(k) == 0
                    continue;
                end                  
                for ts_number = 1:size(ts,1) % 存在多次调整时，ts_number>1
                    [temp_derivw_o, ~, ~, ~, part2, part3, part4] = bp_output (ts(ts_number,:) , psp_m_h, psp_s_h, scaleRate,threshold, w_o(:,k),[], th1); 
                    [~, ~, ~, ~, ~, ~, ~, subparth_w,~] = bp_output (th1, psp_m_i, psp_s_i, scaleRate,threshold,w_h, [], []);
                    [derivw_h_temp,~] = bp_hidden (Ni, Nh, scaleRate, part2, part3, part4, subparth_w, []);
                    derivw_h = derivw_h + type(k) .* derivw_h_temp;  % 改变权�?? 与之前的改变量相�? 
                    derivw_o(:,k) = derivw_o(:,k) + temp_derivw_o;
                end
            end

            % adam
            [deltWh, w_mh, w_vh] = adam(derivw_h, (epoch-1)*nTrain + i, w_mh, w_vh, 1); % 上一句确定正负，不需要再
            [deltWo, w_mo, w_vo] = adam(derivw_o, (epoch-1)*nTrain + i, w_mo, w_vo, type);
            w_h = w_h + learnRate_w_h(learnRate) .* deltWh;
            w_o = w_o + learnRate_w_o(learnRate) .* deltWo;     

        end
        numCorrect = Aug_testing1 (train_pattern(:, 1:nTrain), train_labels(1:nTrain), w_h, w_o);
        fprintf(' trainCorrect = %d, trainAccuracy = %2.4f\n',numCorrect,numCorrect/nTrain);
        if numCorrect > maxTrain( learnRate) 
            maxTrain(learnRate) = numCorrect;   % 记录�?大训练率
        end
        numCorrect = Aug_testing1 (test_pattern(:, 1:nTest), test_labels(1:nTest), w_h, w_o);
        fprintf(' testCorrect = %d, testAccuracy = %2.4f\n',numCorrect,numCorrect/nTest);
        if numCorrect > maxTest(learnRate)
            maxTest(learnRate) = numCorrect;   % 记录�?大测试率
            max_w_o{learnRate} = w_o;  % 记录测试率最大时，权�?
            max_w_h{learnRate} = w_h;  
        end
    end   
    save muti-fa-mnist_result maxTrain maxTest max_w_h max_w_o -mat
end
