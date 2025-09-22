function [Error ] = ERROR( X,Y )%X为实测值 Y为模拟值 
Error=zeros(1,4);
%% 1 RMSE 2 MAE 3 R 4 MAPE 5 NSE 
% Error(1,1)=sqrt(sum((X-Y).^2)/length(X));
Error(1,1)=sqrt(mse(X-Y));
Error(1,2)=sum(abs(X-Y))/length(X);
Error(1,3)=sum((X-mean(X)).*(Y-mean(Y)))/(sqrt(sum((X-mean(X)).^2))*sqrt(sum((Y-mean(Y)).^2)));
Error(1,4)=sum(abs((X-Y)./X))/length(X);
% Error(1,4)=1-sum((X-Y).^2)/sum((X-mean(X)).^2);

% Error(1,6)=mae(X-Y);
end

