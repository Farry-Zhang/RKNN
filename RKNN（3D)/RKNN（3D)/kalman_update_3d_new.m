% 三维Kalman更新函数（修改版）
function [x_est, P, cache] = kalman_update_3d_new(x_prev, P_prev, rknn_output, z, param, true_pos)
% rknn_output: [αx; αy; αz; βx; βy; βz] - 6×1
    % x_prev: [px; py; pz; vx; vy; vz; ax; ay; az] - 9×1
    % z: [zx; zy; zz] - 3×1 观测值
    % P_prev: 9×9 协方差矩阵
    % true_pos: [true_px; true_py; true_pz] - 3×1 真实位置

    if size(true_pos, 2) > 1
        true_pos = true_pos(:); % 确保是列向量
    end
    
    alpha_x = rknn_output(1);
    alpha_y = rknn_output(2);
    alpha_z = rknn_output(3);
    beta_x = rknn_output(4);
    beta_y = rknn_output(5);
    beta_z = rknn_output(6);

    % 系统参数
    T = param.T;
    
    % 9×9 状态转移矩阵 [px py pz vx vy vz ax ay az]'
    A = [1 0 0 T 0 0 0.5*T^2 0 0;      % px
         0 1 0 0 T 0 0 0.5*T^2 0;      % py
         0 0 1 0 0 T 0 0 0.5*T^2;      % pz
         0 0 0 1 0 0 T 0 0;            % vx
         0 0 0 0 1 0 0 T 0;            % vy
         0 0 0 0 0 1 0 0 T;            % vz
         0 0 0 0 0 0 1 0 0;            % ax
         0 0 0 0 0 0 0 1 0;            % ay
         0 0 0 0 0 0 0 0 1];           % az
    
    % 3×9 观测矩阵 (只观测位置)
    H = [1 0 0 0 0 0 0 0 0;
         0 1 0 0 0 0 0 0 0;
         0 0 1 0 0 0 0 0 0];

    % 计算自适应Q和R
    Q = blkdiag(alpha_x * param.Q0_x, alpha_y * param.Q0_y, alpha_z * param.Q0_z); % 9×9对角块矩阵
    R = diag([beta_x * param.R0_x, beta_y * param.R0_y, beta_z * param.R0_z]);   % 3×3对角矩阵

    % 预测步骤
    x_pred = A * x_prev;              % 9×1
    P_pred = A * P_prev * A' + Q;     % 9×9

    % 更新步骤
    S = H * P_pred * H' + R;          % 3×3
    K = P_pred * H' / S;              % 9×3
    v = z - H * x_pred;               % 3×1 新息向量

    x_est = x_pred + K * v;           % 9×1
    P = (eye(9) - K * H) * P_pred;    % 9×9

    
    true_state = [true_pos; x_est(4:9)]; % 只有位置是真实的，速度和加速度使用估计值
    cache.b = true_state - x_pred;       % 使用真实状态与预测的差
    
    % 返回缓存供反向传播使用
    cache.v = v;                      % 3×1
    cache.H = H;                      % 3×9
    cache.P_pred = P_pred;           % 9×9
    cache.S = S;                      % 3×3
    cache.K = K;                      % 9×3
    cache.x_pred = x_pred;           % 9×1
    cache.z = z;                     % 3×1
    cache.A = A;                     % 9×9
    cache.Q = Q;                     % 9×9
    cache.R = R;                     % 3×3
end