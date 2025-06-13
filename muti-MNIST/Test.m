clear; 
close all;

load('./MNIST.mat'); 
% �������ݣ�test_labels-10000*1,test_pattern-784*10000,
% train_labels-60000*1,train_pattern-784*60000

train_labels(train_labels == 0) = 10; % ����ǩ�е�0����Ϊ10 (60000,1)
test_labels(test_labels == 0) = 10;
train_pattern = 8*train_pattern; % ����ֵ����8�� (784,60000)
test_pattern = 8*test_pattern;
% ��ǩ��ֱ�ӵ����֣������Ǵ�����ģ���ɫ����ֵ��ΪInf

nTrain = 1;%length(train_labels); % 60000
nTest = length(test_labels); % 10000
% ѵ������Լ���������

global threshold tau_m tau_s Vnorm T dt Timeline nargin groupSize scaleRate Nclass
threshold = 1; tau_m = 20; tau_s = tau_m/4; % ��ֵ�����峣��
beta = tau_m/tau_s; Vnorm = beta^(beta/(beta-1))/(beta-1); % Ĥ��λ��һ��
T = 40; dt = 1; Timeline = dt: dt: T; % ʱ�䣬(1,40)
nargin = 0; groupSize = 1; scaleRate = 0.1; Nclass = 10;

dropRate = 0.1; sigma = 0.1;
Ni = 784; Nh = 800; No = Nclass*groupSize; % ����ṹ��784-800-10
maxEpoch = 1;%200; % ѵ������
lr_w_o = [.0005 .0003 .0001]; lr_w_h = lr_w_o; % �����㼰�����ѧϰ��
w_mh = zeros(Ni, Nh);  w_vh = w_mh; % �����adam����
w_mo = zeros(Nh, No);  w_vo = w_mo;
maxTrain = zeros(length(lr_w_o),1); maxTest = zeros(length(lr_w_o),1); % ��׼ȷ��
max_w_o = cell(length(lr_w_o),1); max_w_h = cell(length(lr_w_o),1);% ��Ȩ��

for lr = 1: length(lr_w_o) % 3����ͬ��ѧϰ�ʸ���һ��
    w_h = unifrnd(-0.15, 0.2, Ni, Nh); % �����Ȩ�س�ʼ�� (784,800),(800,10)
    w_o = unifrnd(-0.05, 0.06, Nh, No);
    for epoch = 1: maxEpoch % 1-200ѭ����200��ѵ����
        fprintf('lr = %d�� epoch = %d\n', lr, epoch); % *�д�
        numOrder = randperm(nTrain); % ��ѵ�����������˳��
        for i = 1: nTrain % 1-60000ѭ������һʹ��ѵ����������
            if mod(i,10) == 0
                fprintf('i = %d\n', i);
            end
            ti = train_pattern(:,numOrder(i)); % (784,1)�����뵥�������
            
            % ǰ�򴫲�
            [psp_m_i, psp_s_i, psp_i] = Aug_CalPSP(ti); % ���������Ĥ��λӰ��
            [outNumh, th1, th2, Vh, ~, cj_h] = Aug_CalOut(threshold, w_h, psp_i);
            
            [psp_m_h, psp_s_h, psp_h] = Aug_CalPSP(th2);           
            [outNum, to1, ~, Vo, ~, cj_o] = Aug_CalOut(threshold, w_o, psp_h);
                        
            % ���򴫲�
            derivw_o = zeros(Nh, No); derivw_h = zeros(Ni, Nh); % �ݶȳ�ʼ��
            type = zeros(1, No);
            for k = 1:No % ѭ�����������Ԫ
                [ts, type(k)] = Target_SMDP(k, train_labels(numOrder(i)), outNum, to1, Vo, threshold, psp_m_h, psp_s_h, w_o(:,k), cj_o(k,:));
                if type(k) == 0 % ����ı䷢�����������
                    continue;
                end
                for ts_number = 1:size(ts,1) % ts_number>1ʱ��ͬʱ���ٶ������
                    [derivw_o_temp, ~, part2, part3, part4] = bp_output(cj_o, ts(ts_number,:) , psp_m_h, psp_s_h, scaleRate, threshold, w_o(:,k), th1);
                    [~, ~, ~, ~, ~, subparth_w] = bp_output(cj_h, th1, psp_m_i, psp_s_i, scaleRate, threshold, w_h, []);
                    [derivw_h_temp] = bp_hidden(Ni, Nh, scaleRate, part2, part3, part4, subparth_w);
                    
                    derivw_o(:,k) = derivw_o(:,k) + derivw_o_temp; % ������ݶ�
                    derivw_h = derivw_h + type(k) .* derivw_h_temp; % �������ݶ�
                end
            end
            
            % �����ݶȣ���adam�ķ�ʽ��Ȩ��
            [deltWh, w_mh, w_vh] = adam(derivw_h, (epoch-1)*nTrain + i, w_mh, w_vh, 1);
            [deltWo, w_mo, w_vo] = adam(derivw_o, (epoch-1)*nTrain + i, w_mo, w_vo, type);
            w_h = w_h + lr_w_h(lr) .* deltWh;
            w_o = w_o + lr_w_o(lr) .* deltWo;
        end
        
%         size(test_pattern)
        numCorrect = Aug_testing(test_pattern(:, 1:nTest), test_labels(1:nTest), w_h, w_o);
        
    end
end











