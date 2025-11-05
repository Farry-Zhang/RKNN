% 一维Kalman更新函数
function [x_est, P, cache] = kalman_update_1d_new(x_prev, P_prev, rknn_output, z, param, true_pos)
    % rknn_output: [α; β] - 2×1
    % x_prev: [p; v; a] - 3×1
    % z: 标量观测值
    % P_prev: 3×3 协方差矩阵
    % true_pos: 标量真实位置

    alpha = rknn_output(1);
    beta = rknn_output(2);

    % 系统参数
    T = param.T;
    
    % 3×3 状态转移矩阵 [p v a]'
    A = [1 T 0.5*T^2;    % position
         0 1 T;          % velocity
         0 0 1];         % acceleration
    
    % 1×3 观测矩阵 (只观测位置)
    H = [1 0 0];

    % 计算自适应Q和R
    Q = alpha * param.Q0; % 3×3
    R = beta * param.R0;  % 标量

    % 预测步骤
    x_pred = A * x_prev;              % 3×1
    P_pred = A * P_prev * A' + Q;     % 3×3

    % 更新步骤
    S = H * P_pred * H' + R;          % 标量
    K = P_pred * H' / S;              % 3×1
    v = z - H * x_pred;               % 标量新息向量

    x_est = x_pred + K * v;           % 3×1
    P = (eye(3) - K * H) * P_pred;    % 3×3

    true_state = [true_pos; x_est(2:3)]; % 只有位置是真实的，速度和加速度使用估计值
    cache.b = true_state - x_pred;       % 使用真实状态与预测的差
    
    % 返回缓存供反向传播使用
    cache.v = v;                      % 标量
    cache.H = H;                      % 1×3
    cache.P_pred = P_pred;           % 3×3
    cache.S = S;                      % 标量
    cache.K = K;                     % 3×1
    cache.x_pred = x_pred;           % 3×1
    cache.z = z;                     % 标量
    cache.A = A;                     % 3×3
    cache.Q = Q;                     % 3×3
    cache.R = R;                     % 标量
end
