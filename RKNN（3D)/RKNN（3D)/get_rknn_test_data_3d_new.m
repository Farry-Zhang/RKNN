function [true_positions, observations] = get_rknn_test_data_3d_new(num_trajectories, params)
    traj_length = 200; % 200个时间步
    
    % 初始化输出
    true_positions = zeros(num_trajectories, traj_length, 3); % 3D真实位置 [x; y; z]
    observations = zeros(num_trajectories, traj_length, 3);   % 3D观测值 [x; y; z]  
       
   for b = 1:num_trajectories
        % 随机生成轨迹
        [positions, ~, ~] = generate_maneuvering_trajectory_3d(params);
        
        % 存储真实位置
        true_positions(b, :, :) = positions';
        
        % 生成带噪声的观测
        for k = 1:traj_length
            observations(b, k, :) = positions(:,k) + ...
                [sqrt(params.r0_x); sqrt(params.r0_y); sqrt(params.r0_z)] .* randn(3,1);
        end
    end
end


