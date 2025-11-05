function [inputs, targets] = get_rknn_training_data_1d_new(num_trajectories, params)
    traj_length = 200; % 200个时间步
    
    % 输入特征维度: [x_pred_norm(3), e_norm(1), z_norm(1)] = 5维
    inputs = zeros(num_trajectories, traj_length, 5);
    targets = zeros(num_trajectories, traj_length, 1); % 1D位置目标
    
    % 系统参数
    T = params.T;
    A = [1 T 0.5*T^2;    % position
         0 1 T;          % velocity
         0 0 1];         % acceleration
    
    % 观测矩阵
    H = [1 0 0];
    
    for b = 1:num_trajectories
        % 随机生成轨迹
        [positions, velocities, accelerations] = generate_maneuvering_trajectory_1d(params);
        
        % 初始化Kalman滤波
        x_est = [positions(1); velocities(1); accelerations(1)];  % 3×1
        P = diag([10, 1, 1]); % 初始协方差
        
        % 用于差分归一化的前一预测
        x_pred_prev = zeros(3,1);
        
        % 逐步运行轨迹
        for k = 1:traj_length
            true_pos = positions(k); % 标量
            
            % 预测
            x_pred = A * x_est; % 3×1
            
            % 观测（带噪声）
            z = true_pos + sqrt(params.r0) * randn();
            
            % 预测协方差
            P_pred = A * P * A' + params.Q0; % 3×3
            S = H * P_pred * H' + params.r0; % 标量
            
            % 计算新息
            v = z - H * x_pred; % 标量
            
            % 前10步用经典Kalman更新
            if k <= 10
                K = P_pred * H' / S;                  % 3×1
                x_est = x_pred + K * v;               % 更新后状态
                P = (eye(3) - K * H) * P_pred;        % 更新后协方差
                
                % 保存本步预测，作为下一步的 x_pred_prev
                x_pred_prev = x_pred;
            else
                % 从第11步开始：构建RKNN训练输入
                % 增量归一化
                x_pred_norm = (x_pred - x_pred_prev) / params.max_velocity; % 3×1
                x_pred_prev = x_pred; % 更新前一预测
                
                % 新息归一化
                e_norm = v^2 / S; % 标量
                z_norm = v / params.max_velocity;
                
                % 存储输入特征 (5维)
                inputs(b, k, :) = [x_pred_norm; e_norm; z_norm];
                
                % 在数据生成时我们不把RKNN的输出给出，所以用预测替代更新
                x_est = x_pred;
                P = P_pred;
            end
            
            % 存储真实位置 (target)
            targets(b, k, :) = positions(k);
        end
    end
end
