clear;
clc;

% 定义二维RKNN网络参数
params.input_dim = 10;    % [x_inc(6), e_norm_x(1), e_norm_y(1), z_norm_x(1), z_norm_y(1)]
params.hidden_dim = 40;   % 隐藏层维度 (增加以处理更复杂的2D问题)
params.output_dim = 4;    % 输出[αx, αy, βx, βy]四个参数
params.max_velocity = 400;

% 初始化二维Kalman滤波参数
params.T = 0.5;
params.q0_x = 9;
params.q0_y = 9;
params.r0_x = 4900;
params.r0_y = 4900;

% 使用固定的Gamma0 (每个维度独立)
params.Gamma0_x = [0.5 * params.T^2; params.T; 1]; % 对应 [px; vx; ax]
params.Gamma0_y = [0.5 * params.T^2; params.T; 1]; % 对应 [py; vy; ay]

% 构建Q0矩阵 (3×3 对角块矩阵)
params.Q0_x = params.q0_x * (params.Gamma0_x * params.Gamma0_x'); % 3×3
params.Q0_y = params.q0_y * (params.Gamma0_y * params.Gamma0_y'); % 3×3

% 构建R0矩阵
params.R0_x = params.r0_x;
params.R0_y = params.r0_y;

% 初始化网络权重和偏置 (使用Xavier初始化)
% Linear1: 处理状态和误差特征 (8输入 6+2-> hidden_dim)
fan_in_1 = 8; fan_out_1 = params.hidden_dim;
xavier_bound_1 = sqrt(6 / (fan_in_1 + fan_out_1));
net.Linear1_weight = (rand(params.hidden_dim, 8) - 0.5) * 2 * xavier_bound_1;
net.Linear1_bias = zeros(params.hidden_dim, 1);

% Linear2: 处理测量特征 (2输入 -> hidden_dim)
fan_in_2 = 2; fan_out_2 = params.hidden_dim;
xavier_bound_2 = sqrt(6 / (fan_in_2 + fan_out_2));
net.Linear2_weight = (rand(params.hidden_dim, 2) - 0.5) * 2 * xavier_bound_2;
net.Linear2_bias = zeros(params.hidden_dim, 1);

% Linear3: 输出层 (hidden_dim -> 4输出)
fan_in_3 = params.hidden_dim; fan_out_3 = 4;
xavier_bound_3 = sqrt(6 / (fan_in_3 + fan_out_3));
net.Linear3_weight = (rand(4, params.hidden_dim) - 0.5) * 2 * xavier_bound_3;
net.Linear3_bias = zeros(4, 1);

% 训练参数 - 修正初始化函数
initialize_kalman = @() deal([5000 + 2000 * rand(); ...  % px ∈ [5000,7000]
                              5000 + 2000 * rand(); ...  % py ∈ [5000,7000]
                              -100 + 200 * rand();  ...  % vx ∈ [-100,100]
                              -100 + 200 * rand();  ...  % vy ∈ [-100,100]
                              0;                     ...  % ax = 0
                              0],                    ...  % ay = 0
                             diag([10, 10, 1, 1, 1, 1]));

fprintf('开始训练二维RKNN网络...\n');
fprintf('网络结构: 输入层(10) -> 隐藏层(%d) -> 输出层(4)\n', params.hidden_dim);
fprintf('激活函数: Tanh (隐藏层), Sigmoid (输出层)\n');
fprintf('状态向量: [px, py, vx, vy, ax, ay] (6维)\n');
fprintf('自适应参数: [αx, αy, βx, βy] (4维)\n');
fprintf('========================================\n');

% 训练网络
[trained_net, training_losses] = trainRKNN_2d_new(net, params);

fprintf('========================================\n');
fprintf('开始评估训练数据性能...\n');

% 使用测试数据评估性能
num_test_trajectories = 100;
[true_positions, observations] = get_rknn_test_data_2d_new(num_test_trajectories, params);

positions_est = zeros(num_test_trajectories, 200, 2); % [batch, time, dim]
positions_true = zeros(num_test_trajectories, 200, 2);
alpha_values = zeros(num_test_trajectories, 200, 2); % αx, αy
beta_values = zeros(num_test_trajectories, 200, 2);  % βx, βy

% 系统矩阵
A = [1 0 params.T 0 0.5*params.T^2 0;
     0 1 0 params.T 0 0.5*params.T^2;
     0 0 1 0 params.T 0;
     0 0 0 1 0 params.T;
     0 0 0 0 1 0;
     0 0 0 0 0 1];
H = [1 0 0 0 0 0;
     0 1 0 0 0 0];

% 初始化Kalman滤波的协方差矩阵
P_init = diag([10, 10, 1, 1, 1, 1]);

% 全程使用RKNN进行评估
for b = 1:num_test_trajectories
    % 使用前两帧观测值初始化状态
    % 位置使用第一帧观测值
    px = observations(b, 1, 1);
    py = observations(b, 1, 2);
    
    % 速度使用(第二帧观测值-第一帧观测值)/T
    vx = (observations(b, 2, 1) - observations(b, 1, 1)) / params.T;
    vy = (observations(b, 2, 2) - observations(b, 1, 2)) / params.T;
    
    % 加速度初始化为0
    ax = 0;
    ay = 0;
    
    % 初始化状态和协方差
    x_est = [px; py; vx; vy; ax; ay]; % 6×1
    P = P_init;
    
    % 初始化上一预测状态
    x_pred_prev = x_est;

    % 初始化上一时刻量测值
    prev_z = squeeze(observations(b, 1, :)); % 第一帧观测值

    for k = 1:200
        % 获取真实位置和观测值
        true_pos = squeeze(true_positions(b, k, :)); % 2×1
        noisy_z = squeeze(observations(b, k, :));    % 2×1

        % 计算量测变化率
        if k == 1
            delta_z = zeros(2, 1); % 第一帧没有变化率
        else
            delta_z = noisy_z - prev_z;
        end
        
        % 预测步骤
        x_pred = A * x_est;
        
        % 预测协方差
        P_pred = A * P * A' + blkdiag(params.Q0_x, params.Q0_y);
        S = H * P_pred * H' + diag([params.r0_x, params.r0_y]);
        
        % 计算新息
        v = noisy_z - H * x_pred;
        
        % 构建RKNN输入特征
        % 增量归一化
        x_pred_norm = (x_pred - x_pred_prev) / params.max_velocity;
        
        % 新息归一化
        S_inv = inv(S);
        e_norm_x = v(1)' * S_inv(1,1) * v(1);
        e_norm_y = v(2)' * S_inv(2,2) * v(2);
        
        % 归一化量测误差
        z_norm_x = delta_z(1) / params.max_velocity;
        z_norm_y = delta_z(2) / params.max_velocity;
        
        % 构建网络输入
        current_input = [x_pred_norm; e_norm_x; e_norm_y; z_norm_x; z_norm_y];
        
        % 运行二维RKNN获取4个自适应参数
        [rknn_output, ~] = runRKNN_2d_new(trained_net, current_input);
        
        % 记录自适应参数
        alpha_values(b, k, :) = rknn_output(1:2); % [αx, αy]
        beta_values(b, k, :) = rknn_output(3:4);  % [βx, βy]
        
        % 使用自适应参数更新二维Kalman滤波
        [x_est, P, ~] = kalman_update_2d_new(x_est, P, rknn_output, noisy_z, params, true_pos);
        
        % 更新上一预测状态
        x_pred_prev = x_pred;
        prev_z = noisy_z;
        
        % 存储估计和真实位置
        positions_est(b, k, :) = x_est(1:2)'; % 存储估计的[px, py]
        positions_true(b, k, :) = true_pos';   % 存储真实[px, py]
    end
    
    fprintf('完成测试轨迹: %d/%d\n', b, num_test_trajectories);
end

% 计算性能指标
errors_x = abs(positions_est(:,:,1) - positions_true(:,:,1));
errors_y = abs(positions_est(:,:,2) - positions_true(:,:,2));
errors_total = sqrt((positions_est(:,:,1) - positions_true(:,:,1)).^2 + ...
                   (positions_est(:,:,2) - positions_true(:,:,2)).^2);

avg_l1_error_x = mean(errors_x(:));
avg_l1_error_y = mean(errors_y(:));
avg_euclidean_error = mean(errors_total(:));

fprintf('** 训练数据跟踪性能指标 **\n');
fprintf('X方向平均L1误差: %.4f m\n', avg_l1_error_x);
fprintf('Y方向平均L1误差: %.4f m\n', avg_l1_error_y);
fprintf('平均欧氏距离误差: %.4f m\n', avg_euclidean_error);

squared_errors = (positions_est(:,:,1) - positions_true(:,:,1)).^2 + ...
                 (positions_est(:,:,2) - positions_true(:,:,2)).^2;
rmse_per_timestep = sqrt(mean(squared_errors, 1));
avg_rmse = mean(rmse_per_timestep);
fprintf('平均RMSE: %.4f m\n', avg_rmse);

% 计算自适应参数统计
fprintf('** 自适应参数统计 **\n');
fprintf('αx值范围: [%.4f, %.4f], 平均值: %.4f\n', ...
        min(alpha_values(:,:,1),[],'all'), max(alpha_values(:,:,1),[],'all'), ...
        mean(alpha_values(:,:,1),'all'));
fprintf('αy值范围: [%.4f, %.4f], 平均值: %.4f\n', ...
        min(alpha_values(:,:,2),[],'all'), max(alpha_values(:,:,2),[],'all'), ...
        mean(alpha_values(:,:,2),'all'));
fprintf('βx值范围: [%.4f, %.4f], 平均值: %.4f\n', ...
        min(beta_values(:,:,1),[],'all'), max(beta_values(:,:,1),[],'all'), ...
        mean(beta_values(:,:,1),'all'));
fprintf('βy值范围: [%.4f, %.4f], 平均值: %.4f\n', ...
        min(beta_values(:,:,2),[],'all'), max(beta_values(:,:,2),[],'all'), ...
        mean(beta_values(:,:,2),'all'));

% 可视化结果
plot_idx = randi(num_test_trajectories);

% 二维轨迹跟踪结果
figure('Color', [1 1 1], 'Position', [100, 100, 1200, 800]);

% 2D轨迹图
subplot(2,3,1);
plot(positions_true(plot_idx, :, 1), positions_true(plot_idx, :, 2), 'b-', 'LineWidth', 2);
hold on;
plot(positions_est(plot_idx, :, 1), positions_est(plot_idx, :, 2), 'r--', 'LineWidth', 1.5);
xlabel('X位置 (m)');
ylabel('Y位置 (m)');
title(sprintf('训练轨迹 %d 二维跟踪结果', plot_idx));
legend('真实轨迹', 'RKNN估计', 'Location', 'best');
grid on;
axis equal;

% X方向跟踪
subplot(2,3,2);
plot(1:200, positions_true(plot_idx, :, 1), 'b-', 'LineWidth', 2);
hold on;
plot(1:200, positions_est(plot_idx, :, 1), 'r--', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('X位置 (m)');
title('X方向跟踪');
legend('真实位置', 'RKNN估计', 'Location', 'best');
grid on;

% Y方向跟踪
subplot(2,3,3);
plot(1:200, positions_true(plot_idx, :, 2), 'b-', 'LineWidth', 2);
hold on;
plot(1:200, positions_est(plot_idx, :, 2), 'r--', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('Y位置 (m)');
title('Y方向跟踪');
legend('真实位置', 'RKNN估计', 'Location', 'best');
grid on;

% RMSE随时间变化
subplot(2,3,4);
plot(1:200, rmse_per_timestep, 'k-', 'LineWidth', 2);
xlabel('时间步');
ylabel('RMSE (m)');
title('跟踪RMSE随时间变化');
grid on;

% 训练损失曲线
subplot(2,3,5);
plot(1:length(training_losses), training_losses, 'r-', 'LineWidth', 2);
xlabel('Epoch');
ylabel('训练损失');
title('训练损失曲线');
grid on;

% 自适应参数α的变化
subplot(2,3,6);
plot(1:200, squeeze(alpha_values(plot_idx, :, 1)), 'g-', 'LineWidth', 1.5);
hold on;
plot(1:200, squeeze(alpha_values(plot_idx, :, 2)), 'c-', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('α值');
title('α参数变化');
legend('αx', 'αy', 'Location', 'best');
grid on;

% 误差分布直方图
figure('Color', [1 1 1], 'Position', [200, 200, 800, 600]);

subplot(2,2,1);
histogram(errors_x(:), 50, 'FaceColor', 'blue', 'EdgeColor', 'black', 'FaceAlpha', 0.7);
xlabel('X方向跟踪误差 (m)');
ylabel('频次');
title('X方向误差分布');
grid on;

subplot(2,2,2);
histogram(errors_y(:), 50, 'FaceColor', 'red', 'EdgeColor', 'black', 'FaceAlpha', 0.7);
xlabel('Y方向跟踪误差 (m)');
ylabel('频次');
title('Y方向误差分布');
grid on;

subplot(2,2,3);
histogram(errors_total(:), 50, 'FaceColor', 'green', 'EdgeColor', 'black', 'FaceAlpha', 0.7);
xlabel('欧氏距离误差 (m)');
ylabel('频次');
title('总体误差分布');
grid on;

subplot(2,2,4);
scatter(errors_x(:), errors_y(:), 10, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('X方向误差 (m)');
ylabel('Y方向误差 (m)');
title('X-Y误差相关性');
grid on;

fprintf('========================================\n');
fprintf('二维训练和评估完成！\n');
fprintf('训练性能 - X方向L1: %.4f m, Y方向L1: %.4f m\n', avg_l1_error_x, avg_l1_error_y);
fprintf('训练性能 - 平均欧氏误差: %.4f m, RMSE: %.4f m\n', avg_euclidean_error, avg_rmse);
