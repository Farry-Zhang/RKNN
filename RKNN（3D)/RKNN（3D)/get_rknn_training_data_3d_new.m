function [inputs, targets] = get_rknn_training_data_3d_new(num_trajectories, params)
    traj_length = 200; % 200个时间步
    
    % 输入特征维度: [x_pred_norm(9), e_norm_x(1), e_norm_y(1),e_norm_z(1),  z_norm_x(1), z_norm_y(1), z_norm_z(1)] = 15维
    inputs = zeros(num_trajectories, traj_length, 15);
    targets = zeros(num_trajectories, traj_length, 3); % 3D位置目标 [x; y; z]
    
    % 系统参数
    T = params.T;
    A = [1 0 0 T 0 0 0.5*T^2 0 0;      % px
         0 1 0 0 T 0 0 0.5*T^2 0;      % py
         0 0 1 0 0 T 0 0 0.5*T^2;      % pz
         0 0 0 1 0 0 T 0 0;            % vx
         0 0 0 0 1 0 0 T 0;            % vy
         0 0 0 0 0 1 0 0 T;            % vz
         0 0 0 0 0 0 1 0 0;            % ax
         0 0 0 0 0 0 0 1 0;            % ay
         0 0 0 0 0 0 0 0 1];           % az

    % 观测矩阵
    H = [1 0 0 0 0 0 0 0 0;
         0 1 0 0 0 0 0 0 0;
         0 0 1 0 0 0 0 0 0];
    
    for b = 1:num_trajectories
        % 随机生成轨迹（需你自己有此函数）
        [positions, velocities, accelerations] = generate_maneuvering_trajectory_3d(params);
        
        % 初始化Kalman滤波
        x_est = [positions(:,1); velocities(:,1); accelerations(:,1)];  % 9×1
        P = diag([10, 10, 10, 1, 1, 1, 1, 1, 1]); % 初始协方差
        
        % 用于差分归一化的前一预测（会在k<=10阶段被设为第k的预测）
        x_pred_prev = zeros(9,1);
        
        % 逐步运行轨迹
        for k = 1:traj_length
            true_pos = positions(:,k); % 3×1
            
            % 预测
            x_pred = A * x_est; % 9×1  当前时刻的预测状态
            
            % 观测（带噪声）
            z = true_pos + [sqrt(params.r0_x); sqrt(params.r0_y); sqrt(params.r0_z)] .* randn(3,1);
            
            % 预测协方差（注意这里用正确的 blkdiag）
            P_pred = A * P * A' + blkdiag(params.Q0_x, params.Q0_y, params.Q0_z); % 9×9
            S = H * P_pred * H' + diag([params.r0_x, params.r0_y, params.r0_z]); % 3×3
            
            % 计算新息
            v = z - H * x_pred; % 3×1
            
            % 前10步用经典Kalman更新
            if k <= 10
                K = P_pred * H' / S;                  % 9×3
                x_est = x_pred + K * v;               % 更新后状态
                P = (eye(9) - K * H) * P_pred;        % 更新后协方差
                
                % 保存本步预测，作为下一步（k+1）的 x_pred_prev
                x_pred_prev = x_pred;
            else
                % 从第11步开始：构建RKNN训练输入（基于预测增量）
                % 增量归一化（注意：x_pred_prev 已由 k==10 时设好）
                x_pred_norm = (x_pred - x_pred_prev) / params.max_velocity; % 9×1
                x_pred_prev = x_pred; % 更新前一预测
                
                % 新息归一化（标量）
                S_inv = inv(S);
                e_norm_x = v(1)' * S_inv(1,1) * v(1); % 标量
                e_norm_y = v(2)' * S_inv(2,2) * v(2); % 标量
                e_norm_z = v(3)' * S_inv(3,3) * v(3); % 标量
                z_norm_x = v(1) / params.max_velocity;
                z_norm_y = v(2) / params.max_velocity;
                z_norm_z = v(3) / params.max_velocity;
                
                % 存储输入特征 (15维)
                inputs(b, k, :) = [x_pred_norm; e_norm_x; e_norm_y; e_norm_z; z_norm_x; z_norm_y; z_norm_z];
                
                % 在数据生成时我们不把RKNN的输出给出，所以用预测替代更新（保持仿真）
                x_est = x_pred;
                P = P_pred;
            end
            
            % 存储真实位置 (target)
            targets(b, k, :) = positions(:,k)';
        end
    end
end

%先经过了几次经典kalman之后得到inputs数据，然后将最后一次的input数据送入RKNN中，target还是真实位置
%使用一个固定的初始状态和协方差，然后生成真实轨迹和观测，但不进行滤波。然后，在训练时，我们从头开始运行Kalman滤波，前10步使用固定参数，第11步开始使用RKNN，并记录每一步的输入特征（使用当前步的预测和当前步的观测来计算特征）。