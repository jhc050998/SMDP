clear; 
close all;

% load('./MNIST.mat'); 
% �������ݣ�test_labels-10000*1,test_pattern-784*10000,
% train_labels-60000*1,train_pattern-784*60000

%��ȡ���ݼ��е�ͼƬ
train_pattern1 = readMNISTImages('D:/Dataset/mnist/FashionMNIST/raw/train-images-idx3-ubyte'); %60000��ѵ����������СΪ28*28*60000
test_pattern1 = readMNISTImages('D:/Dataset/mnist/FashionMNIST/raw/t10k-images-idx3-ubyte'); %10000��������������СΪ28*28*10000

%��ȡ���ݼ��еı�ǩ
train_labels1 = readMNISTLabels('D:/Dataset/mnist/FashionMNIST/raw/train-labels-idx1-ubyte'); %��ǩ0-9��60000����ǩ����СΪ60000*1
test_labels1 = readMNISTLabels('D:/Dataset/mnist/FashionMNIST/raw/t10k-labels-idx1-ubyte'); %10000����ǩ����СΪ10000*1

% Ԥ����
train_pattern1 = 3 * (255 - train_pattern1)/255;
index1 = find(train_pattern1 >= 2.9);
train_pattern1(index1) = inf;

test_pattern1 = 3 * (255 - test_pattern1)/255;
index2 = find(test_pattern1 >= 2.9);
test_pattern1(index2) = inf;

% ����һ�佫���ݴ�Ϊ.mat�ļ�
save FashionMNIST test_labels1 test_pattern1 train_labels1 train_pattern1 

% train_pattern��MNIST.mat��ԭ���ݣ�train_pattern1�Ǳ��εõ��Ķ�Ӧ����
% train_pattern1(200:240,1)
% train_pattern(200:240,1)

% ��ԭ����ȡ����������ͼ
% samples = ImageMNIST_integration(train_pattern1, train_labels1, 10);
% A = mat2gray(samples);
% figure(1)
% imshow(A, 'Border', 'tight');
% print(gcfm, '-r1000', '-djpeg', 'My_FashionMNIST.jpg')



