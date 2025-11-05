% 一维RKNN反向传播函数
function [grad_L1_w, grad_L1_b, grad_L2_w, grad_L2_b, grad_L3_w, grad_L3_b] = ...
    rknn_backward_1d_new(cache, window_size, params, net)
    
    hidden_dim = size(net.Linear1_weight, 1);
    
    % 初始化梯度
    grad_L1_w = zeros(hidden_dim, 4);
    grad_L1_b = zeros(hidden_dim, 1);
    grad_L2_w = zeros(hidden_dim, 1);
    grad_L2_b = zeros(hidden_dim, 1);
    grad_L3_w = zeros(2, hidden_dim);
    grad_L3_b = zeros(2, 1);
    
    for t = window_size:-1:1
        kalman_cache = cache(t).kalman_cache;
        
        K = kalman_cache.K;               % Kalman增益 3×1
        v = kalman_cache.v;               % 新息向量 标量
        H = kalman_cache.H;               % 观测矩阵 1×3
        P_pred = kalman_cache.P_pred;     % 预测协方差 3×3
        S = kalman_cache.S;               % 新息协方差 标量
        b = kalman_cache.b;               % 观测偏差 3×1
        
        % 计算中间量
        S_inv = 1/S;
        
        Kv_minus_b = K * v - b;  % 3×1 - 3×1 = 3×1
        
        % 计算通用表达式
        common_expr1 = (eye(3) - H' * S_inv * H * P_pred') * Kv_minus_b * v * S_inv * H;
        common_expr2 = -2 * S_inv * H * P_pred' * Kv_minus_b * v * S_inv;
        
        % α的梯度 (Q矩阵参数)
        dL_dalpha = params.Q0(1,1) * common_expr1(1,1);
        
        % β的梯度 (R矩阵参数)
        dL_dbeta = params.R0 * common_expr2;
        
        % 组合输出梯度
        dL_doutput = [dL_dalpha; dL_dbeta];  % 2×1
        
        % === 神经网络反向传播 ===
        sigmoid_output = cache(t).output;
        sigmoid_deriv = sigmoid_output .* (1 - sigmoid_output);
        dL_dLinear3_out = dL_doutput .* sigmoid_deriv;
        
        hidden_input = cache(t).hidden;
        grad_L3_w = grad_L3_w + dL_dLinear3_out * hidden_input';
        grad_L3_b = grad_L3_b + dL_dLinear3_out;
        
        dL_dhidden = net.Linear3_weight' * dL_dLinear3_out;
        tanh_output = cache(t).hidden;
        tanh_deriv = 1 - tanh_output.^2;
        dL_dcombined = dL_dhidden .* tanh_deriv;
        
        input_features1 = cache(t).input(1:4);
        input_features2 = cache(t).input(5);
        grad_L1_w = grad_L1_w + dL_dcombined * input_features1';
        grad_L1_b = grad_L1_b + dL_dcombined;
        grad_L2_w = grad_L2_w + dL_dcombined * input_features2';
        grad_L2_b = grad_L2_b + dL_dcombined;
    end
end

