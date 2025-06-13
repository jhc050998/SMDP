%% wbc 单层
clear 
close all
load('./encoded_wbc.mat')
tic
samples = round(encoded_wbc);
global threshold tau_m tau_s Vnorm T dt Timeline nargin groupSize scaleRate Nclass
scaleRate = 1 ;          groupSize = 1;  % 输出神经元组数�?�newtesting函数中使�?
nargin = 0 ;         threshold = 1;  
tau_m = 20;          tau_s = tau_m/4;          beta = tau_m/tau_s;
Vnorm = beta^( beta /(beta-1) ) / (beta-1);
T = 100;         dt = 1;          Timeline = dt: dt: T; 
Ni = 135;      No = 2;         Nclass = No;% 231-iono,80-pima,150-bupa,135-wbc,160-iris 
learnRate_w_o = [.01] ;  % [1 .01 .005 .001 ] 
maxEpoch = 1;     maxTrial = 1;   % 200,20
nTrain = floor(0.5 * length(label));   % �?半的数据作为训练�?
nTest = length(label) - nTrain; % �?半的数据为测试集
% adam参数
w_mo = zeros(Ni, No);  w_vo = w_mo;

%% 保留数据
maxTrain = zeros(maxTrial, length(learnRate_w_o));
maxTest = zeros(maxTrial, length(learnRate_w_o));
max_w_o = cell(maxTrial,length(learnRate_w_o)); % 存权�? 用cell存，而不是四维数�?
%%  训练 MTP
for learnRate = 1: length(learnRate_w_o) % 循环学习�?
    for trial = 1: maxTrial  % 循环圈数
        % 初始化权值，取出数据标签
        w_o = unifrnd(-0.05, 0.06, Ni, No); 
        numOrder = randperm(length(label));
        train_data = samples(:, numOrder(1:nTrain)); % 数据
        train_label = label(numOrder(1:nTrain));  % 标签
        test_data = samples(:, numOrder(nTrain+1:end));
        test_label = label(numOrder(nTrain+1:end));
        for epoch = 1: maxEpoch
            fprintf('learnRate = %d, tiral = %d, epoch = %d\n', learnRate, trial, epoch); 
            numOrder = randperm(nTrain);
            for i = 1: nTrain
                %% 噪声
                ti = train_data(:,numOrder(i));
                ti = ti + normrnd( 0, 0.05, size(ti) );  ti( ti < 0 ) = Inf; 
                label_k = train_label( numOrder(i) );
                % labels = train_label(numOrder(i));
                %% 前向传播
                [psp_m, psp_s, psp_i] = Aug_CalPSP(ti,[]);
                [outNum, to,~, Vo,~,cj_o]  = Aug_CalOut (threshold, w_o, psp_i); 

                %% 后向梯度计算
                derivw_o = zeros(Ni, No);   % 输出层梯度（单次梯度�?
                type = zeros(1, No);
                for k = 1: No
                    if k ~= label_k
                        if outNum(k) == 0
                            continue;
                        else
                            type(k) = -1;
                            ts = to(k,:);
                            ts(ts == Inf) = [];  % 去掉inf�?
                            if length(ts) > 10  % �?次调整多项，提高速度
                                derivw_o(:,k) = SMDP_Learn_reduce_two(threshold, w_o(:,k), ts, psp_m, psp_s, scaleRate,  cj_o(k,:) );  % 这里梯度爆炸0.5（是否改�?0.1�?
                            else  % 目标差距小，调整单脉�?
                                derivw_o(:,k) = SMDP_Learn(threshold, w_o(:,k), ts, psp_m, psp_s, scaleRate, cj_o(k,:) );
                            end
                        end
                    else
                        if outNum(k) > 3    
                            continue;
                        else   
                            [ts,type(k)] = SMDP_target_increase(k, to, Vo, threshold, psp_m, psp_s, w_o(:,k),  cj_o(k,:) );
                            derivw_o(:,k) = SMDP_Learn(threshold, w_o(:,k), ts, psp_m, psp_s, scaleRate,  cj_o(k,:) );
                        end
                    end
                end
                % adam
                [deltWo, w_mo, w_vo] = adam(derivw_o, (epoch-1)*nTrain + i, w_mo, w_vo, type);
                w_o = w_o + learnRate_w_o(learnRate) .* deltWo;     
                % 动�?�优�?
                % w_o = w_o +  learnRate_w_o(learnRate) * type .* derivw_o + mu * old_deltaw;   % 每个输出神经元k的改变量都确定再改变
                % old_deltaw = learnRate_w_o(learnRate) * type .* derivw_o + mu * old_deltaw;  

            end
            numCorrect = Aug_testing (train_data(:, 1:nTrain), train_label(1:nTrain), [], w_o );
            fprintf(' trainCorrect = %d, trainAccuracy = %2.4f\n',numCorrect,numCorrect/nTrain);
            if numCorrect > maxTrain(trial, learnRate) 
                maxTrain(trial, learnRate) = numCorrect;   % 记录�?大训练率
            end
            numCorrect = Aug_testing (test_data(:, 1:nTest), test_label(1:nTest), [], w_o);
            fprintf(' testCorrect = %d, testAccuracy = %2.4f\n',numCorrect,numCorrect/nTest);
            if numCorrect > maxTest(trial, learnRate)
                maxTest(trial, learnRate) = numCorrect;   % 记录�?大测试率
                max_w_o{trial, learnRate} = w_o;  % 记录测试率最大时，权�?
            end
        end
    end
    save single_wbc_result maxTrain maxTest  max_w_o -mat
end
toc;