clear; 
close all;

train_pattern = [];
train_labels = [];
load('./data_batch_1.mat');
% 读到内容：data-10000*3072, labels-10000*1,（整个训练集的5分之一）
% batch_label
train_pattern = [train_pattern data'];
train_labels = [train_labels labels'];
load('./data_batch_2.mat');
train_pattern = [train_pattern data'];
train_labels = [train_labels labels'];
load('./data_batch_3.mat');
train_pattern = [train_pattern data'];
train_labels = [train_labels labels'];
load('./data_batch_4.mat');
train_pattern = [train_pattern data'];
train_labels = [train_labels labels'];
load('./data_batch_5.mat');
train_pattern = [train_pattern data'];
train_labels = [train_labels labels'];
train_labels = train_labels';

load('./test_batch.mat'); % 与训练集的5分之一同格式
test_pattern = data';
test_labels = labels;

% size(train_pattern) % (3072, 50000)
% size(train_labels) % (50000, 1)
% size(test_pattern) % (3072, 10000)
% size(test_labels) % (10000, 1)

train_pattern = double(train_pattern);
test_pattern = double(test_pattern);

% 预处理
a = 24;

train_pattern = a * (255 - train_pattern)/255;
index1 = find(train_pattern >= 2.9*(a/3));
train_pattern(index1) = inf;
index_a = find(train_pattern <= 0.1*(a/3));
train_pattern(index_a) = 0.1*(a/3);
train_pattern = round(train_pattern * 10)/10;

test_pattern = a * (255 - test_pattern)/255;
index2 = find(test_pattern >= 2.9*(a/3));
test_pattern(index2) = inf;
index_b = find(test_pattern <= 0.1*(a/3));
test_pattern(index_b) = 0.1*(a/3);
test_pattern = round(test_pattern * 10)/10;

% 下面一句将数据存为.mat文件
% save CIFAR10_24 test_labels test_pattern train_labels train_pattern 

train_pattern(100:120,1)


% images = permute(reshape(data', [32, 32, 3, 10000]), [2, 1, 3, 4]);

% outputFolder = 'D:/Dataset/CiFar/CiFar10/png';
% outputFolder = fullfile(outputFolder, 'images');
% if ~exist(outputFolder, 'dir')
%     mkdir(outputFolder)
% end
% 
% % 保存为png文件
% for i = 1:10
%     fileName = fullfile(outputFolder, ['img_' num2str(i) '.png']);
%     imageToSave = images(:,:,:,i);
%     imageToSave = double(imageToSave)/255.0;
%     imwrite(imageToSave, fileName);
% end
% 
% %在Matlab中显示
% for i = 1:10
%     fileName = fullfile(outputFolder, ['img_' num2str(i) '.png']);
%     savedImage = imread(fileName);
%     
%     subplot(2, 5, i);
%     imshow(savedImage);
%     title(['Image ' num2str(i)]);
% end


