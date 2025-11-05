clear;
clc;

% 定义一维RKNN网络参数
params.input_dim = 5;    % [x_inc(3), e_norm(1), z_norm(1)]
params.hidden_dim = 30;   % 隐藏层维度
params.output_dim = 2;    % 输出[α, β]两个参数
params.max_velocity = 400;

% 初始化一维Kalman滤波参数
params.T = 0.5;
params.q0 = 9;
params.r0 = 4900;

% 使用固定的Gamma0
params.Gamma0 = [0.5 * params.T^2; params.T; 1]; % 对应 [p; v; a]

% 构建Q0矩阵 (3×3)
params.Q0 = params.q0 * (params.Gamma0 * params.Gamma0'); % 3×3

% 构建R0矩阵
params.R0 = params.r0;

% 初始化网络权重和偏置 (使用Xavier初始化)
% Linear1: 处理状态和误差特征 (4输入 3+1-> hidden_dim)
fan_in_1 = 4; fan_out_1 = params.hidden_dim;
xavier_bound_1 = sqrt(6 / (fan_in_1 + fan_out_1));
net.Linear1_weight = (rand(params.hidden_dim, 4) - 0.5) * 2 * xavier_bound_1;
net.Linear1_bias = zeros(params.hidden_dim, 1);

% Linear2: 处理测量特征 (1输入 -> hidden_dim)
fan_in_2 = 1; fan_out_2 = params.hidden_dim;
xavier_bound_2 = sqrt(6 / (fan_in_2 + fan_out_2));
net.Linear2_weight = (rand(params.hidden_dim, 1) - 0.5) * 2 * xavier_bound_2;
net.Linear2_bias = zeros(params.hidden_dim, 1);

% Linear3: 输出层 (hidden_dim -> 2输出)
fan_in_3 = params.hidden_dim; fan_out_3 = 2;
xavier_bound_3 = sqrt(6 / (fan_in_3 + fan_out_3));
net.Linear3_weight = (rand(2, params.hidden_dim) - 0.5) * 2 * xavier_bound_3;
net.Linear3_bias = zeros(2, 1);

% 训练参数 - 修正初始化函数
initialize_kalman = @() deal([5000 + 2000 * rand(); ...  % p ∈ [5000,7000]
                              -100 + 200 * rand();  ...  % v ∈ [-100,100]
                              0],                    ...  % a = 0
                             diag([10, 1, 1]));

fprintf('开始训练一维RKNN网络...\n');
fprintf('网络结构: 输入层(5) -> 隐藏层(%d) -> 输出层(2)\n', params.hidden_dim);
fprintf('激活函数: Tanh (隐藏层), Sigmoid (输出层)\n');
fprintf('状态向量: [p, v, a] (3维)\n');
fprintf('自适应参数: [α, β] (2维)\n');
fprintf('========================================\n');

% 训练网络
[trained_net, training_losses] = trainRKNN_1d_new(net, params);

fprintf('========================================\n');
fprintf('开始评估测试数据性能...\n');

% 使用测试数据评估性能
num_test_trajectories = 100;
[true_positions, observations] = get_rknn_test_data_1d_new(num_test_trajectories, params);

positions_est = zeros(num_test_trajectories, 200); % [batch, time]
positions_true = zeros(num_test_trajectories, 200);
alpha_values = zeros(num_test_trajectories, 200); % α
beta_values = zeros(num_test_trajectories, 200);  % β

% 系统矩阵
A = [1 params.T 0.5*params.T^2;    % position
     0 1 params.T;                 % velocity
     0 0 1];                       % acceleration
H = [1 0 0];

% 初始化Kalman滤波的协方差矩阵
P_init = diag([10, 1, 1]);

% 全程使用RKNN进行评估
for b = 1:num_test_trajectories
    % 使用前两帧观测值初始化状态
    % 位置使用第一帧观测值
    p = observations(b, 1);
    
    % 速度使用(第二帧观测值-第一帧观测值)/T
    v = (observations(b, 2) - observations(b, 1)) / params.T;
    
    % 加速度初始化为0
    a = 0;
    
    % 初始化状态和协方差
    x_est = [p; v; a]; % 3×1
    P = P_init;
    
    % 初始化上一预测状态
    x_pred_prev = x_est;

    for k = 1:200
        % 获取真实位置和观测值
        true_pos = true_positions(b, k);     % 标量
        noisy_z = observations(b, k);        % 标量
        
        % 预测步骤
        x_pred = A * x_est;
        
        % 预测协方差
        P_pred = A * P * A' + params.Q0;
        S = H * P_pred * H' + params.r0;
        
        % 计算新息
        v_innov = noisy_z - H * x_pred;  % 用v_innov避免与速度v混淆
        
        % 构建RKNN输入特征
        % 增量归一化
        x_pred_norm = (x_pred - x_pred_prev) / params.max_velocity;
        
        % 新息归一化
        e_norm = v_innov^2 / S;  % 标量
        
        % 归一化量测误差
        z_norm = v_innov / params.max_velocity;
        
        % 构建网络输入
        current_input = [x_pred_norm; e_norm; z_norm]; % 5×1
        
        % 运行一维RKNN获取2个自适应参数
        [rknn_output, ~] = runRKNN_1d_new(trained_net, current_input);
        
        % 记录自适应参数
        alpha_values(b, k) = rknn_output(1); % α
        beta_values(b, k) = rknn_output(2);  % β
        
        % 使用自适应参数更新一维Kalman滤波
        [x_est, P, ~] = kalman_update_1d_new(x_est, P, rknn_output, noisy_z, params, true_pos);
        
        % 更新上一预测状态
        x_pred_prev = x_pred;
        
        % 存储估计和真实位置
        positions_est(b, k) = x_est(1);      % 存储估计的位置
        positions_true(b, k) = true_pos;     % 存储真实位置
    end
    
    fprintf('完成测试轨迹: %d/%d\n', b, num_test_trajectories);
end

% 计算性能指标
errors = abs(positions_est - positions_true);
avg_l1_error = mean(errors(:));

fprintf('** 测试数据跟踪性能指标 **\n');
fprintf('平均L1误差: %.4f m\n', avg_l1_error);

squared_errors = (positions_est - positions_true).^2;
rmse_per_timestep = sqrt(mean(squared_errors, 1));
avg_rmse = mean(rmse_per_timestep);
fprintf('平均RMSE: %.4f m\n', avg_rmse);

% 计算自适应参数统计
fprintf('** 自适应参数统计 **\n');
fprintf('α值范围: [%.4f, %.4f], 平均值: %.4f\n', ...
        min(alpha_values,[],'all'), max(alpha_values,[],'all'), ...
        mean(alpha_values,'all'));
fprintf('β值范围: [%.4f, %.4f], 平均值: %.4f\n', ...
        min(beta_values,[],'all'), max(beta_values,[],'all'), ...
        mean(beta_values,'all'));

% 可视化结果
plot_idx = randi(num_test_trajectories);

% 一维轨迹跟踪结果
figure('Color', [1 1 1], 'Position', [100, 100, 1200, 800]);

% 位置跟踪
subplot(2,3,1);
plot(1:200, positions_true(plot_idx, :), 'b-', 'LineWidth', 2);
hold on;
plot(1:200, positions_est(plot_idx, :), 'r--', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('位置 (m)');
title(sprintf('测试轨迹 %d 位置跟踪结果', plot_idx));
legend('真实位置', 'RKNN估计', 'Location', 'best');
grid on;

% 跟踪误差
subplot(2,3,2);
plot(1:200, errors(plot_idx, :), 'k-', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('跟踪误差 (m)');
title('跟踪误差随时间变化');
grid on;

% RMSE随时间变化
subplot(2,3,3);
plot(1:200, rmse_per_timestep, 'k-', 'LineWidth', 2);
xlabel('时间步');
ylabel('RMSE (m)');
title('跟踪RMSE随时间变化');
grid on;

% 训练损失曲线
subplot(2,3,4);
plot(1:length(training_losses), training_losses, 'r-', 'LineWidth', 2);
xlabel('Epoch');
ylabel('训练损失');
title('训练损失曲线');
grid on;

% 自适应参数α的变化
subplot(2,3,5);
plot(1:200, alpha_values(plot_idx, :), 'g-', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('α值');
title('α参数变化');
grid on;

% 自适应参数β的变化
subplot(2,3,6);
plot(1:200, beta_values(plot_idx, :), 'c-', 'LineWidth', 1.5);
xlabel('时间步');
ylabel('β值');
title('β参数变化');
grid on;

% 误差分布直方图
figure('Color', [1 1 1], 'Position', [200, 200, 800, 600]);

subplot(2,2,1);
histogram(errors(:), 50, 'FaceColor', 'blue', 'EdgeColor', 'black', 'FaceAlpha', 0.7);
xlabel('跟踪误差 (m)');
ylabel('频次');
title('误差分布');
grid on;

subplot(2,2,2);
plot(1:200, mean(errors, 1), 'r-', 'LineWidth', 2);
xlabel('时间步');
ylabel('平均误差 (m)');
title('平均误差随时间变化');
grid on;

subplot(2,2,3);
plot(1:200, std(errors, 1), 'g-', 'LineWidth', 2);
xlabel('时间步');
ylabel('误差标准差 (m)');
title('误差标准差随时间变化');
grid on;

subplot(2,2,4);
scatter(alpha_values(:), beta_values(:), 10, 'filled', 'MarkerFaceAlpha', 0.6);
xlabel('α值');
ylabel('β值');
title('α-β参数相关性');
grid on;

fprintf('========================================\n');
fprintf('一维训练和评估完成！\n');
fprintf('测试性能 - 平均L1误差: %.4f m, RMSE: %.4f m\n', avg_l1_error, avg_rmse);