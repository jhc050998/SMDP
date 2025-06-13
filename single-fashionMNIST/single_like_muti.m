%% 单层MNIST
clear 
close all
load('./MNIST.mat')
train_labels(train_labels == 0) = 10;  % 将标签中的0改成10
test_labels(test_labels == 0) = 10;
train_pattern = 8 * train_pattern;    % 原始数据为[0，2.9], 乘以10
test_pattern = 8 * test_pattern;
nTrain = length(train_labels);
nTest = length(test_labels);
global threshold tau_m tau_s Vnorm T dt Timeline nargin groupSize scaleRate Nclass       
groupSize = 10; 
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
No = Nclass*groupSize;   
    %% 学习率
learnRate_w_o = [.0005  .00005 ,005 ];

maxEpoch = 500;   
% adam参数
w_mo = zeros(Ni, No);  w_vo = w_mo;
 %% 保留数据
maxTrain = zeros(length(learnRate_w_o));
maxTest = zeros(length(learnRate_w_o));
max_w_o = cell(length(learnRate_w_o)); % 存权值
%%  训练 MTP
for learnRate = 1: length(learnRate_w_o) 
    w_o = normrnd(0.01, 0.01, Ni, No);  
    for epoch = 1: maxEpoch
        fprintf('learnRate = %d， epoch = %d\n', learnRate, epoch); 
        order = randperm(nTrain);
        for i = 1: nTrain
            %% 噪声
            ti = train_pattern(:, order(i));
            ti = ti + normrnd( 0, 0.05, size(ti) );  ti( ti < 0 ) = 0; 
            label_k = train_labels(order(i));
            %% 前向传播
                [psp_m, psp_s, psp_i] = Aug_CalPSP(ti,[]);
                [outNum, to,~, Vo,~,cj]  = Aug_CalOut (threshold, w_o, psp_i); 

                %% 后向梯度计算
                derivw_o = zeros(Ni, No);   % 输出层梯度（单次梯度）
                type = zeros(1, No);
                for k = 1: No
                    if  mod(k - label_k, Nclass)
                        if outNum(k) == 0
                            continue;
                        else
                            type(k) = -1;
                            ts = to(k,:);
                            ts(ts == Inf) = [];  % 去掉inf项
                            if length(ts) > 10  % 一次调整多项，提高速度
                                derivw_o(:,k) = SMDP_Learn_reduce_two(threshold, w_o(:,k), ts, psp_m, psp_s, scaleRate, cj(k,:));  % 这里梯度爆炸0.5（是否改成0.1）
                            else  % 目标差距小，调整单脉冲
                                derivw_o(:,k) = SMDP_Learn(threshold, w_o(:,k), ts, psp_m, psp_s, scaleRate, cj(k,:));
                            end
                        end
                    else
                        if outNum(k) > 3    % Nd = 3
                            continue;
                        else   % Nd较小，单个修改脉冲
                            [ts,type(k)] = SMDP_target_increase(k, to, Vo, threshold, psp_m, psp_s, w_o(:,k),  cj(k,:));
                            derivw_o(:,k) = SMDP_Learn(threshold, w_o(:,k), ts, psp_m, psp_s, scaleRate, cj(k,:));
                        end
                    end
                end
                % adam
                [deltWo, w_mo, w_vo] = adam(derivw_o, (epoch-1)*nTrain + i, w_mo, w_vo, type);
                w_o = w_o + learnRate_w_o(learnRate) .* deltWo;     
                % 动态优化
                % w_o = w_o +  learnRate_w_o(learnRate) * type .* derivw_o + mu * old_deltaw;   % 每个输出神经元k的改变量都确定再改变
                % old_deltaw = learnRate_w_o(learnRate) * type .* derivw_o + mu * old_deltaw;  

        end
        numCorrect = Aug_testing (train_pattern(:, 1:nTrain), train_labels(1:nTrain), [], w_o);
        fprintf(' trainCorrect = %d, trainAccuracy = %2.4f\n',numCorrect,numCorrect/nTrain);
        if numCorrect > maxTrain( learnRate) 
            maxTrain(learnRate) = numCorrect;   % 记录最大训练率
        end
        numCorrect = Aug_testing (test_pattern(:, 1:nTest), test_labels(1:nTest), [], w_o);
        fprintf(' testCorrect = %d, testAccuracy = %2.4f\n',numCorrect,numCorrect/nTest);
        if numCorrect > maxTest(learnRate)
            maxTest(learnRate) = numCorrect;   % 记录最大测试率
            max_w_o{learnRate} = w_o;  % 记录测试率最大时，权值
        end
    save mnist_single_result maxTrain maxTest  max_w_o -mat
    end   
    
end
