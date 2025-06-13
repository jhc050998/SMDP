clear; 
close all;

% load('./MNIST.mat'); 
% 读到内容：test_labels-10000*1,test_pattern-784*10000,
% train_labels-60000*1,train_pattern-784*60000

%读取数据集中的图片
train_pattern1 = readMNISTImages('D:/Dataset/mnist/FashionMNIST/raw/train-images-idx3-ubyte'); %60000个训练样本，大小为28*28*60000
test_pattern1 = readMNISTImages('D:/Dataset/mnist/FashionMNIST/raw/t10k-images-idx3-ubyte'); %10000个测试样本，大小为28*28*10000

%读取数据集中的标签
train_labels1 = readMNISTLabels('D:/Dataset/mnist/FashionMNIST/raw/train-labels-idx1-ubyte'); %标签0-9；60000个标签，大小为60000*1
test_labels1 = readMNISTLabels('D:/Dataset/mnist/FashionMNIST/raw/t10k-labels-idx1-ubyte'); %10000个标签，大小为10000*1

% 预处理
train_pattern1 = 3 * (255 - train_pattern1)/255;
index1 = find(train_pattern1 >= 2.9);
train_pattern1(index1) = inf;

test_pattern1 = 3 * (255 - test_pattern1)/255;
index2 = find(test_pattern1 >= 2.9);
test_pattern1(index2) = inf;

% 下面一句将数据存为.mat文件
save FashionMNIST test_labels1 test_pattern1 train_labels1 train_pattern1 

% train_pattern是MNIST.mat中原内容，train_pattern1是本次得到的对应部分
% train_pattern1(200:240,1)
% train_pattern(200:240,1)

% 对原数据取部分样例作图
% samples = ImageMNIST_integration(train_pattern1, train_labels1, 10);
% A = mat2gray(samples);
% figure(1)
% imshow(A, 'Border', 'tight');
% print(gcfm, '-r1000', '-djpeg', 'My_FashionMNIST.jpg')



