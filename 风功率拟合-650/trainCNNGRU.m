clear;clc;close all;
load D1 %D3 D4结果的很差
%% 数据划分
Xtrain=train(:,1:2);
Ytrain=train(:,3);
Xtest=test(:,1:2);
Ytest=test(:,3);
method=@mapminmax;
%% 打乱训练集
ave=[Xtrain,Ytrain];
a=randperm(size(ave,1))';
a1=ave(a,:);
Xtrain=a1(:,1:2);
Ytrain=a1(:,3);
%% 数据输入
[ntrain d] = size(Xtrain);
[ntest d] = size(Xtest);
train_x=Xtrain';
train_y=Ytrain';
test_x=Xtest';
test_y=Ytest';
%% 归一化
[trainx1, st1] = mapminmax(train_x);
[trainy1, st2] = mapminmax(train_y);
testx1 = mapminmax('apply',test_x,st1);
XTrain=trainx1;
YTrain=trainy1;
XTest=testx1;
x_train=reshape(XTrain,[d,1,1,ntrain]);%训练集输入
x_test=reshape(XTest,[d,1,1,ntest]);%测试集输入
y_train = YTrain;%训练集输出
y_test  = test_y;%测试集输出
%% CNN feature extraction
% Fine tuning the layers using your datasets
layers = [ 
          imageInputLayer([size(x_train,1) 1 1]);
          convolution2dLayer(1,10);%卷积层
          reluLayer();%relu激活函数
          crossChannelNormalizationLayer(1)
          maxPooling2dLayer(1,'Stride',1);%池化层
          convolution2dLayer(1,20);
          maxPooling2dLayer(1,'Stride',1);%池化层
          convolution2dLayer(1,40);
          maxPooling2dLayer(1,'Stride',1);%池化层
          dropoutLayer();
          fullyConnectedLayer(1);%全连接层
          regressionLayer];
options = trainingOptions('sgdm','MaxEpochs',20,'InitialLearnRate',0.00001); %20=最大训练次数 0.00001=学习率
net = trainNetwork(x_train,y_train',layers,options);  %创建网络
trainingFeatures=activations(net, x_train,10);
trainingFeatures=double(reshape(trainingFeatures,[size(trainingFeatures,1)*size(trainingFeatures,3),size(trainingFeatures,4)]));
testFeatures=activations(net, x_test,10);
testFeatures=double(reshape(testFeatures,[size(testFeatures,1)*size(testFeatures,3),size(testFeatures,4)]));
%% LSTM2 直接调用trainELM、testELM的function，不用粘贴后修改代码变量定义
[out,layerslstm]= trainLSTM2(trainingFeatures',y_train',testFeatures',y_test');
predict_value=method('reverse',out,st2);
error(1,:)=ERROR(y_test',predict_value)
%% 拟合散点图
figure(1)
scatter3(predict_value,test_x(1,:)',test_x(2,:)')
hold on
scatter3(y_test',test_x(1,:)',test_x(2,:)')
hold off
legend(["预测值" "实际值"])
ylabel("风速（m/s）")
xlabel("风向(°)")
zlabel("风功率（KW）")
title("风功率拟合")
%% 相对误差图
figure(2)
xdwc=abs((predict_value'-y_test)./y_test);
bar(xdwc)
ylabel("相对误差")
xlabel("时间（30秒）")
title("相对误差图")
%% 相关系数
figure(3)
xgxs=corrcoef(predict_value,y_test);
bar(xgxs(1,2))
xlabel("第一天")
title("相关系数")
function [out,layers] = trainLSTM2(X,Y,Z,Q)
XTrain = X';
YTrain = Y';
[numFeatures datalong] = size(XTrain);
numResponses = 1;
numHiddenUnits1 = 40;
maxEpochs=150;
learning_rate=0.001;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits1)
    fullyConnectedLayer(numResponses)
    regressionLayer];
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'InitialLearnRate',learning_rate, ...
    'GradientThreshold',1, ...
    'Shuffle','every-epoch', ...
    'Verbose',true,...
    'Plots','training-progress');%损失函数曲线图
net = trainNetwork(XTrain,YTrain,layers,options);
XTest = Z';
YPred = predict(net,XTest,'MiniBatchSize',1);
out = YPred';
end