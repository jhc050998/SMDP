%% iono数据�? 多层
clear 
close all
load('./encoded_iono.mat')
samples = round(encoded_iono);
global threshold tau_m tau_s Vnorm T dt Timeline nargin groupSize scaleRate Nclass
scaleRate = 1 ;          groupSize = 1;  % 输出神经元组数�?�newtesting函数中使�?
nargin = 0 ;         threshold = 1;  
tau_m = 20;          tau_s = tau_m/4;          beta = tau_m/tau_s;
Vnorm = beta^( beta /(beta-1) ) / (beta-1);
T = 120;         dt = 1;          Timeline = dt: dt: T; 
Ni = 231;     Nh = 360;    No = 2;           Nclass = No;
learnRate_w_o = [.008];  %[.008 .005 .001 .0005 .0001 .03];
learnRate_w_h = learnRate_w_o;    % 学习�?
maxEpoch = 40;     maxTrial = 1;   % 200,20
nTrain = floor(0.5 * length(label));   % �?半的数据作为训练�?
nTest = length(label) - nTrain; % �?半的数据为测试集
% adam参数
w_mh = zeros(Ni, Nh);  w_vh = w_mh;
w_mo = zeros(Nh, No);  w_vo = w_mo;
 %% 保留数据
maxTrain = zeros(maxTrial, length(learnRate_w_o));
maxTest = zeros(maxTrial, length(learnRate_w_o));
max_w_o = zeros(maxTrial,length(learnRate_w_o), Nh, No); % 存权�?
max_w_h = zeros(maxTrial,length(learnRate_w_o),Ni, Nh);
%%  训练 MTP
for learnRate = 1: length(learnRate_w_o) % 循环学习�?
    for trial = 1: maxTrial  % 循环圈数
        % 初始化权值，取出数据标签
        w_h = unifrnd(-1, 1.1, Ni, Nh);
        w_o = unifrnd(-0.05, 0.06, Nh, No); 
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
                % labels = train_label(numOrder(i));
                %% 前向传播
                [psp_m_i, psp_s_i, psp_i] = Aug_CalPSP (ti,[]);
                [outNumh, th1, th2, Vh,~, cj_h] = Aug_CalOut (threshold, w_h, psp_i); 
                if isempty(th2)
                    continue
                end
                [psp_m_h, psp_s_h, psp_h] = Aug_CalPSP (th2,[]);           
                [outNum, to,~, Vo,~,cj_o] = Aug_CalOut (threshold, w_o, psp_h); 
                %% 后向梯度计算
                derivw_o = zeros(Nh, No);   % 输出层梯度（单次梯度�?
                derivw_h = zeros(Ni, Nh);   % 隐藏层梯�?
                type = zeros(1, No);
                for k = 1: No
                    [ts, type(k)] = Target_SMDP(k, train_label(numOrder(i)), outNum, to, Vo, threshold,psp_m_h, psp_s_h, w_o(:,k), cj_o(k,:));
                    if type(k) == 0
                        continue;
                    end                  
                    for ts_number = 1:size(ts,1) % 存在多次调整时，ts_number>1
                        [temp_derivw_o, ~, ~, ~, part2, part3, part4] = bp_output (cj_o, ts(ts_number,:) , psp_m_h, psp_s_h, scaleRate,threshold, w_o(:,k),[], th1); 
                        [~, ~, ~, ~, ~, ~, ~, subparth_w,~] = bp_output (cj_h,th1, psp_m_i, psp_s_i, scaleRate,threshold,w_h, [], []);
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
            numCorrect = Aug_testing (train_data(:, 1:nTrain), train_label(1:nTrain), w_h, w_o );
            fprintf(' trainCorrect = %d, trainAccuracy = %2.4f\n',numCorrect,numCorrect/nTrain);
            if numCorrect > maxTrain(trial, learnRate) 
                maxTrain(trial, learnRate) = numCorrect;   % 记录�?大训练率
            end
            numCorrect = Aug_testing (test_data(:, 1:nTest), test_label(1:nTest), w_h, w_o);
            fprintf(' testCorrect = %d, testAccuracy = %2.4f\n',numCorrect,numCorrect/nTest);
            if numCorrect > maxTest(trial, learnRate)
                maxTest(trial, learnRate) = numCorrect;   % 记录�?大测试率
                max_w_o(trial, learnRate,:,:) = w_o;  % 记录测试率最大时，权�?
                max_w_h(trial, learnRate,:,:) = w_h;  
            end
        end
    end
    save IONO_result maxTrain maxTest max_w_h max_w_o -mat
end
