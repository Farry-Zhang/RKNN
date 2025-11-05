% 二维Kalman更新函数（修改版）
function [x_est, P, cache] = kalman_update_2d_new(x_prev, P_prev, rknn_output, z, param, true_pos)
    % rknn_output: [αx; αy; βx; βy] - 4×1
    % x_prev: [px; py; vx; vy; ax; ay] - 6×1
    % z: [zx; zy] - 2×1 观测值
    % P_prev: 6×6 协方差矩阵
    % true_pos: [true_px; true_py] - 2×1 真实位置

     if size(true_pos, 2) > 1
        true_pos = true_pos(:); % 确保是列向量
    end
    
    alpha_x = rknn_output(1);
    alpha_y = rknn_output(2);
    beta_x = rknn_output(3);
    beta_y = rknn_output(4);

    % 系统参数
    T = param.T;
    
    % 6×6 状态转移矩阵 [px py vx vy ax ay]'
    A = [1 0 T 0 0.5*T^2 0;      % px
         0 1 0 T 0 0.5*T^2;      % py
         0 0 1 0 T 0;            % vx
         0 0 0 1 0 T;            % vy
         0 0 0 0 1 0;            % ax
         0 0 0 0 0 1];           % ay
    
    % 2×6 观测矩阵 (只观测位置)
    H = [1 0 0 0 0 0;
         0 1 0 0 0 0];

    % 计算自适应Q和R
    Q = blkdiag(alpha_x * param.Q0_x, alpha_y * param.Q0_y); % 6×6对角块矩阵
    R = diag([beta_x * param.R0_x, beta_y * param.R0_y]);   % 2×2对角矩阵

    % 预测步骤
    x_pred = A * x_prev;              % 6×1
    P_pred = A * P_prev * A' + Q;     % 6×6

    % 更新步骤
    S = H * P_pred * H' + R;          % 2×2
    K = P_pred * H' / S;              % 6×2
    v = z - H * x_pred;               % 2×1 新息向量

    x_est = x_pred + K * v;           % 6×1
    P = (eye(6) - K * H) * P_pred;    % 6×6

    
    true_state = [true_pos; x_est(3:6)]; % 只有位置是真实的，速度和加速度使用估计值
    cache.b = true_state - x_pred;       % 使用真实状态与预测的差
    
    % 返回缓存供反向传播使用
    cache.v = v;                      % 2×1
    cache.H = H;                      % 2×6
    cache.P_pred = P_pred;           % 6×6
    cache.S = S;                      % 2×2
    cache.K = K;                      % 6×2
    cache.x_pred = x_pred;           % 6×1
    cache.z = z;                     % 2×1
    cache.A = A;                     % 6×6
    cache.Q = Q;                     % 6×6
    cache.R = R;                     % 2×2
end