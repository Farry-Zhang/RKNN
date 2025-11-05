E% 一维RKNN前向传播函数
function [output, hidden] = runRKNN_1d_new(net, input)
    % 输入: input = [x_inc(3); e_norm(1); z_norm(1)] - 5×1向量
    % 输出: output = [α; β] - 2×1向量
    
    % 第1层: Linear1 处理状态增量特征和误差特征
    % 输入特征: [x_inc(3); e_norm(1)] = 4维
    input_features1 = input(1:4); % 4×1
    linear1_out = net.Linear1_weight * input_features1 + net.Linear1_bias; % hidden_dim×1
    
    % 第2层: Linear2 处理测量特征
    % 输入特征: [z_norm(1)] = 1维
    input_features2 = input(5); % 标量
    linear2_out = net.Linear2_weight * input_features2 + net.Linear2_bias; % hidden_dim×1
    
    % 合并两个分支并应用Tanh激活
    combined = linear1_out + linear2_out; % hidden_dim×1
    hidden = tanh(combined); % hidden_dim×1
    
    % 第3层: Linear3 + Sigmoid激活，输出2个参数
    linear3_out = net.Linear3_weight * hidden + net.Linear3_bias; % 2×1
    output = sigmoid(linear3_out); % 2×1
    
    % 确保输出在合理范围内
    output = max(output, 0.01); % 避免过小的值
    output = min(output, 10.0); % 避免过大的值
end

function y = sigmoid(x)
    % 数值稳定的sigmoid函数
    y = 1 ./ (1 + exp(-x));
    % 防止数值问题
    y = max(y, eps);
    y = min(y, 1-eps);
end
