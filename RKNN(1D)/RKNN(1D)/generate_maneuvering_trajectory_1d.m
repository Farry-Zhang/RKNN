%% ========== 文件2: generate_maneuvering_trajectory_1d.m ==========
% 一维轨迹生成函数
% 初始位置：在 [5000, 7000]m 范围内随机生成
% 初始速度：在 [-100, 100]m/s 范围内随机生成
% 机动次数：3-5 次随机机动
% 机动时机：在时间步 6-195 之间随机选择机动时刻
% 加速度变化：每次机动时，在 ±30m/s² 范围内随机生成新的加速度
function [positions, velocities, accelerations] = generate_maneuvering_trajectory_1d(params)
    num_points = 200;
    positions = zeros(1, num_points);      % 1×200
    velocities = zeros(1, num_points);     % 1×200
    accelerations = zeros(1, num_points);  % 1×200

    % 初始状态
    positions(1) = 5000 + 2000 * rand(); % 范围[5000,7000]m
    velocities(1) = -100 + 200 * rand();  % 范围[-100,100]m/s

    % 机动参数
    num_maneuvers = randi([3,5]); % 3-5次机动
    maneuver_times = sort(randperm(190, num_maneuvers) + 5); % 避免前5步机动

    current_acc = 0; % 初始加速度

    for k = 2:num_points
        % 检查是否机动时刻
        if any(k == maneuver_times)
            % 随机加速度变化 (在±30m/s²范围内)
            current_acc = -30 + 60 * rand();
        end

        % 更新速度和位置
        velocities(k) = velocities(k-1) + current_acc * params.T;
        positions(k) = positions(k-1) + velocities(k-1) * params.T + 0.5 * current_acc * params.T^2;
        accelerations(k) = current_acc;

        % 限制最大速度
        if abs(velocities(k)) > params.max_velocity
            velocities(k) = sign(velocities(k)) * params.max_velocity;
        end
    end
end