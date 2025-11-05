%% === 二维轨迹生成函数 ===
%初始位置：在 [5000, 7000]m 范围内随机生成 (x,y) 坐标
%初始速度：在 [-100, 100]m/s 范围内随机生成 (vx,vy) 分量
%机动次数：3-5 次随机机动
%机动时机：在时间步 6-195 之间随机选择机动时刻
%加速度变化：每次机动时，在 ±30m/s² 范围内随机生成新的加速度向量
%速度更新时采用匀加速运动模型
function [positions, velocities, accelerations] = generate_maneuvering_trajectory_2d(params)
    num_points = 200;
    positions = zeros(2, num_points);      % 2×200 [x; y]
    velocities = zeros(2, num_points);     % 2×200 [vx; vy]
    accelerations = zeros(2, num_points);  % 2×200 [ax; ay]

    % 初始状态 (论文表1)
    positions(:,1) = [5000 + 2000 * rand(); 5000 + 2000 * rand()]; % [x; y] 范围[5000,7000]m
    velocities(:,1) = [-100 + 200 * rand(); -100 + 200 * rand()];  % [vx; vy] 范围[-100,100]m/s

    % 机动参数 (论文表1)
    num_maneuvers = randi([3,5]); % 3-5次机动
    maneuver_times = sort(randperm(190, num_maneuvers) + 5); % 避免前5步机动

    current_acc = zeros(2,1); % 初始加速度 [ax; ay]

    for k = 2:num_points
        % 检查是否机动时刻
        if any(k == maneuver_times)
            % 随机加速度变化 (在±30m/s²范围内)
            current_acc = [-30 + 60 * rand(); -30 + 60 * rand()];
        end

        % 更新速度和位置
        velocities(:,k) = velocities(:,k-1) + current_acc * params.T;
        positions(:,k) = positions(:,k-1) + velocities(:,k-1) * params.T + 0.5 * current_acc * params.T^2;
        accelerations(:,k) = current_acc;

        % 限制最大速度 (论文表1)
        for dim = 1:2
            if abs(velocities(dim,k)) > params.max_velocity
                velocities(dim,k) = sign(velocities(dim,k)) * params.max_velocity;
            end
        end
    end
end