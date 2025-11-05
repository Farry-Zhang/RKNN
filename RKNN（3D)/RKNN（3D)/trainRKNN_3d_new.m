% 三维RKNN网络训练函数
function [trained_net, training_losses] = trainRKNN_3d_new(net, param)
    % 训练参数
    num_epochs = 200;
    learning_rate = 0.005;
    window_size = 5;
    batch_size = 100;
    
    best_net = net;
    best_loss = Inf;
    training_losses = zeros(1, num_epochs); % 存储每个epoch的损失
    
    % 学习率调度
    lr_schedule = learning_rate * (1 + cos(linspace(0, pi, num_epochs)))/2;
    
    % Kalman初始化函数 - 三维9状态 [px py pz vx vy vz ax ay az]
    initialize_kalman = @() deal(zeros(9,1), diag([10, 10, 10, 1, 1, 1, 1, 1, 1]));
    
    % 定义系统矩阵
    T = param.T;
    A = [1 0 0 T 0 0 0.5*T^2 0 0;      % px
         0 1 0 0 T 0 0 0.5*T^2 0;      % py
         0 0 1 0 0 T 0 0 0.5*T^2;      % pz
         0 0 0 1 0 0 T 0 0;            % vx
         0 0 0 0 1 0 0 T 0;            % vy
         0 0 0 0 0 1 0 0 T;            % vz
         0 0 0 0 0 0 1 0 0;            % ax
         0 0 0 0 0 0 0 1 0;            % ay
         0 0 0 0 0 0 0 0 1];           % az
    
    H = [1 0 0 0 0 0 0 0 0;
         0 1 0 0 0 0 0 0 0;
         0 0 1 0 0 0 0 0 0];
    
    for epoch = 1:num_epochs
        % 获取新的训练数据（每个epoch重新生成）
        [train_inputs, train_targets] = get_rknn_training_data_3d_new(batch_size, param);
        
        epoch_loss = 0;
        
        % 初始化梯度累积器
        hidden_dim = size(net.Linear1_weight, 1);
        total_grad_L1_w = zeros(hidden_dim, 12);
        total_grad_L1_b = zeros(hidden_dim, 1);
        total_grad_L2_w = zeros(hidden_dim, 3);
        total_grad_L2_b = zeros(hidden_dim, 1);
        total_grad_L3_w = zeros(6, hidden_dim);
        total_grad_L3_b = zeros(6, 1);
        
        total_windows = 0;
        
        % 添加测量噪声
        noisy_inputs = add_measurement_noise_3d(train_inputs, sqrt(param.r0_x + param.r0_y + param.r0_z)/3);
        
        % 批次训练
        for batch = 1:batch_size
            [x_est, P] = initialize_kalman();
            
            % 预分配缓存
            trajectory_length = size(noisy_inputs, 2); % 200
            cache = struct(...
                'input', cell(1, trajectory_length), ...
                'hidden', cell(1, trajectory_length), ...
                'output', cell(1, trajectory_length), ...
                'pos_error', cell(1, trajectory_length), ...
                'kalman_cache', cell(1, trajectory_length) ...
            );

            % 初始化上一时刻量测值
            prev_z = zeros(3, 1);
            
             for k = 1:10               
                true_pos = squeeze(train_targets(batch, k, :)); % 3×1，确保是列向量
                true_pos = true_pos(:); % 强制转换为列向量，此时是3*1的列向量
                % 添加观测噪声
                noisy_z = true_pos + [sqrt(param.r0_x); sqrt(param.r0_y); sqrt(param.r0_z)] .* randn(3,1);
                
                % 使用固定参数进行Kalman更新
                fixed_output = ones(6, 1); % αx=1, αy=1, αz=1, βx=1, βy=1, βz=1
                [x_est, P, kalman_cache] = kalman_update_3d_new(x_est, P, fixed_output, noisy_z, param, true_pos); % 添加 true_pos 参数
                                
                % 计算位置误差
                pos_error = norm(true_pos - x_est(1:3)); % 欧氏距离误差L2范数
                
                % 缓存数据（前10步不用于RKNN训练，但记录误差）
                cache(k).input = zeros(15, 1); % 空输入
                cache(k).hidden = zeros(size(net.Linear1_bias)); % 空隐藏状态
                cache(k).output = fixed_output;
                cache(k).pos_error = pos_error;
                cache(k).kalman_cache = kalman_cache;

                 % 更新上一时刻量测值
                prev_z = noisy_z;
            end
            
            % 从第11步开始使用RKNN
            for k = 11:trajectory_length
                true_pos = squeeze(train_targets(batch, k, :)); % 3×1
                % 添加观测噪声
                noisy_z = true_pos + [sqrt(param.r0_x); sqrt(param.r0_y); sqrt(param.r0_z)] .* randn(3,1);
                
                 % 计算量测变化率
                delta_z = noisy_z - prev_z;

                x_pred = A * x_est;  % 9×1，提前计算，保证在 k==11 分支也存在
                               
                % 获取当前输入（前一步的预测结果）
                if k == 11
                    % 第11步使用预先生成的输入（已经是15×1）
                    current_input = squeeze(noisy_inputs(batch, k, :)); 
                else
                    % ---- 状态增量归一化 ----
                    x_pred_prev = cache(k-1).kalman_cache.x_pred;  % 取前一时刻预测（第11步时不会用到）
                    x_pred_norm = (x_pred - x_pred_prev) / param.max_velocity; % 9×1
                    
                    % 计算新息向量
                    v = noisy_z - H * x_pred; % 3×1
                    
                    % 计算新息协方差矩阵S
                    P_pred = A * P * A' + blkdiag(param.Q0_x, param.Q0_y, param.Q0_z); % 9×9
                    S = H * P_pred * H' + diag([param.r0_x, param.r0_y, param.r0_z]); % 3×3
                    
                    % 计算新息标量
                    S_inv = inv(S);
                    e_norm_x = v(1)' * S_inv(1,1) * v(1); % 标量
                    e_norm_y = v(2)' * S_inv(2,2) * v(2); % 标量
                    e_norm_z = v(3)' * S_inv(3,3) * v(3); % 标量
                    z_norm_x = delta_z(1) / param.max_velocity;
                    z_norm_y = delta_z(2) / param.max_velocity;
                    z_norm_z = delta_z(3) / param.max_velocity;
                    
                    % 构建输入特征 (15×1)
                    current_input = [x_pred_norm(:); e_norm_x(:); e_norm_y(:); e_norm_z(:); z_norm_x(:); z_norm_y(:); z_norm_z(:)];
                end
                
                % RKNN前向传播
                [output, hidden] = runRKNN_3d_new(net, current_input);
                
                % Kalman更新（kalman_update_3d_new 应该返回更新后的 kalman_cache）
                [x_est, P, kalman_cache] = kalman_update_3d_new(x_est, P, output, noisy_z, param, true_pos); % 添加 true_pos 参数
                
                % 把当前步的预测状态存入 kalman_cache 方便下次用（保证下一步能取到 cache(k).kalman_cache.x_pred）
                kalman_cache.x_pred = x_pred;
                
                % 计算位置误差
                pos_error = norm(true_pos - x_est(1:3)); % 欧氏距离误差
                
                % 缓存数据
                cache(k).input = current_input;
                cache(k).hidden = hidden;
                cache(k).output = output;
                cache(k).pos_error = pos_error;
                cache(k).kalman_cache = kalman_cache;

                 % 更新上一时刻量测值
                prev_z = noisy_z;
            end
            
            % 滑动窗口反向传播，从第16步（11+5）开始
            for k = 16:trajectory_length
                window_start = k - window_size + 1;
                window_end = k;
                
                % 确保窗口内的所有步骤都使用了RKNN（k >= 11）
                if window_start >= 11
                    % 计算窗口损失
                    window_loss = 0;
                    for t = window_start:window_end
                        window_loss = window_loss + cache(t).pos_error;
                    end
                    epoch_loss = epoch_loss + window_loss;
                    
                    % 反向传播计算梯度
                    [grad_L1_w, grad_L1_b, grad_L2_w, grad_L2_b, grad_L3_w, grad_L3_b] = ...
                        rknn_backward_3d_new(cache(window_start:window_end), window_size, param, net);
                    
                    % 累积梯度
                    total_grad_L1_w = total_grad_L1_w + grad_L1_w;
                    total_grad_L1_b = total_grad_L1_b + grad_L1_b;
                    total_grad_L2_w = total_grad_L2_w + grad_L2_w;
                    total_grad_L2_b = total_grad_L2_b + grad_L2_b;
                    total_grad_L3_w = total_grad_L3_w + grad_L3_w;
                    total_grad_L3_b = total_grad_L3_b + grad_L3_b;
                    
                    total_windows = total_windows + 1;
                end
            end
        end
        
        % 计算平均梯度
        avg_grad_L1_w = total_grad_L1_w / total_windows;
        avg_grad_L1_b = total_grad_L1_b / total_windows;
        avg_grad_L2_w = total_grad_L2_w / total_windows;
        avg_grad_L2_b = total_grad_L2_b / total_windows;
        avg_grad_L3_w = total_grad_L3_w / total_windows;
        avg_grad_L3_b = total_grad_L3_b / total_windows;
        
        % 梯度裁剪（防止梯度爆炸）
        grad_norm = sqrt(sum(avg_grad_L1_w(:).^2) + sum(avg_grad_L1_b(:).^2) + ...
                        sum(avg_grad_L2_w(:).^2) + sum(avg_grad_L2_b(:).^2) + ...
                        sum(avg_grad_L3_w(:).^2) + sum(avg_grad_L3_b(:).^2));
        
        max_grad_norm = 5.0; % 梯度裁剪阈值
        if grad_norm > max_grad_norm
            clip_factor = max_grad_norm / grad_norm;
            avg_grad_L1_w = avg_grad_L1_w * clip_factor;
            avg_grad_L1_b = avg_grad_L1_b * clip_factor;
            avg_grad_L2_w = avg_grad_L2_w * clip_factor;
            avg_grad_L2_b = avg_grad_L2_b * clip_factor;
            avg_grad_L3_w = avg_grad_L3_w * clip_factor;
            avg_grad_L3_b = avg_grad_L3_b * clip_factor;
        end
        
        % 更新网络参数
        net.Linear1_weight = net.Linear1_weight - lr_schedule(epoch) * avg_grad_L1_w;
        net.Linear1_bias = net.Linear1_bias - lr_schedule(epoch) * avg_grad_L1_b;
        net.Linear2_weight = net.Linear2_weight - lr_schedule(epoch) * avg_grad_L2_w;
        net.Linear2_bias = net.Linear2_bias - lr_schedule(epoch) * avg_grad_L2_b;
        net.Linear3_weight = net.Linear3_weight - lr_schedule(epoch) * avg_grad_L3_w;
        net.Linear3_bias = net.Linear3_bias - lr_schedule(epoch) * avg_grad_L3_b;
        
        % 计算平均损失
        avg_epoch_loss = epoch_loss / total_windows;
        training_losses(epoch) = avg_epoch_loss;
        
        % 保存最佳网络
        if avg_epoch_loss < best_loss
            best_net = net;
            best_loss = avg_epoch_loss;
        end
        
        % 输出训练信息
        if mod(epoch, 10) == 0 || epoch == 1
            fprintf('Epoch %d/%d | Loss: %.4f | Grad Norm: %.4f | LR: %.6f\n', ...
                    epoch, num_epochs, avg_epoch_loss, grad_norm, lr_schedule(epoch));
        end
    end
    
    trained_net = best_net;
    fprintf('三维训练完成! 最佳损失: %.4f\n', best_loss);
end
