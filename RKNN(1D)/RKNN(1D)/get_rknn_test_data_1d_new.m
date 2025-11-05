% 一维RKNN测试数据生成函数
function [true_positions, observations] = get_rknn_test_data_1d_new(num_trajectories, params)
    traj_length = 200; % 200个时间步
    
    % 输出数据
    true_positions = zeros(num_trajectories, traj_length); % 真实位置
    observations = zeros(num_trajectories, traj_length);   % 带噪声的观测值
    
    for b = 1:num_trajectories
        % 生成真实轨迹
        [positions, ~, ~] = generate_maneuvering_trajectory_1d(params);
        
        % 存储真实位置
        true_positions(b, :) = positions;
        
        % 生成带噪声的观测值
        for k = 1:traj_length
            true_pos = positions(k);
            
            % 添加观测噪声
            noisy_observation = true_pos + sqrt(params.r0) * randn();
            observations(b, k) = noisy_observation;
        end
    end
    
    fprintf('生成了 %d 条测试轨迹，每条 %d 个时间步\n', num_trajectories, traj_length);
end