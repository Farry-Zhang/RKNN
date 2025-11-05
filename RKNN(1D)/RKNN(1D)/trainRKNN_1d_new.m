% 一维RKNN网络训练函数
function [trained_net, training_losses] = trainRKNN_1d_new(net, param)
    % 训练参数
    num_epochs = 200;
    learning_rate = 0.005;
    window_size = 5;
    batch_size = 100;
    
    best_net = net;
    best_loss = Inf;
    training_losses = zeros(1, num_epochs);
    
    % 学习率调度
    lr_schedule = learning_rate * (1 + cos(linspace(0, pi, num_epochs)))/2;
    
    % Kalman初始化函数 - 一维3状态 [p v a]
    initialize_kalman = @() deal(zeros(3,1), diag([10, 1, 1]));
    
    % 定义系统矩阵
    T = param.T;
    A = [1 T 0.5*T^2;    % position
         0 1 T;          % velocity
         0 0 1];         % acceleration
    
    H = [1 0 0];
    
    for epoch = 1:num_epochs
        % 获取新的训练数据（每个epoch重新生成）
        [train_inputs, train_targets] = get_rknn_training_data_1d_new(batch_size, param);
        
        epoch_loss = 0;
        
        % 初始化梯度累积器
        hidden_dim = size(net.Linear1_weight, 1);
        total_grad_L1_w = zeros(hidden_dim, 4);
        total_grad_L1_b = zeros(hidden_dim, 1);
        total_grad_L2_w = zeros(hidden_dim, 1);
        total_grad_L2_b = zeros(hidden_dim, 1);
        total_grad_L3_w = zeros(2, hidden_dim);
        total_grad_L3_b = zeros(2, 1);
        
        total_windows = 0;
        
        % 添加测量噪声
        noisy_inputs = add_measurement_noise_1d(train_inputs, sqrt(param.r0));
        
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
            
            % 前10步使用经典Kalman滤波
            for k = 1:10               
                true_pos = squeeze(train_targets(batch, k, :)); % 标量
                % 添加观测噪声
                noisy_z = true_pos + sqrt(param.r0) * randn();
                
                % 使用固定参数进行Kalman更新
                fixed_output = ones(2, 1); % α=1, β=1
                [x_est, P, kalman_cache] = kalman_update_1d_new(x_est, P, fixed_output, noisy_z, param, true_pos);
                
                % 计算位置误差
                pos_error = abs(true_pos - x_est(1)); % 绝对误差
                
                % 缓存数据（前10步不用于RKNN训练，但记录误差）
                cache(k).input = zeros(5, 1); % 空输入
                cache(k).hidden = zeros(size(net.Linear1_bias)); % 空隐藏状态
                cache(k).output = fixed_output;
                cache(k).pos_error = pos_error;
                cache(k).kalman_cache = kalman_cache;
            end
            
            % 从第11步开始使用RKNN
            for k = 11:trajectory_length
                true_pos = squeeze(train_targets(batch, k, :)); % 标量
                % 添加观测噪声
                noisy_z = true_pos + sqrt(param.r0) * randn();
                
                % 关键修正：先计算 x_pred，确保后面可用
                x_pred = A * x_est;  % 3×1，提前计算，保证在 k==11 分支也存在
                
                % 获取当前输入（前一步的预测结果）
                if k == 11
                    % 第11步使用预先生成的输入（已经是5×1）
                    current_input = squeeze(noisy_inputs(batch, k, :)); 
                else
                    % 状态增量归一化
                    x_pred_prev = cache(k-1).kalman_cache.x_pred;  % 取前一时刻预测
                    x_pred_norm = (x_pred - x_pred_prev) / param.max_velocity; % 3×1
                    
                    % 计算新息向量
                    v = noisy_z - H * x_pred; % 标量
                    
                    % 计算新息协方差矩阵S
                    P_pred = A * P * A' + param.Q0; % 3×3
                    S = H * P_pred * H' + param.r0; % 标量
                    
                    % 计算新息标量
                    e_norm = v^2 / S; % 标量
                    z_norm = v / param.max_velocity;
                    
                    % 构建输入特征 (5×1)
                    current_input = [x_pred_norm(:); e_norm; z_norm];
                end
                
                % RKNN前向传播
                [output, hidden] = runRKNN_1d_new(net, current_input);
                
                % Kalman更新
                [x_est, P, kalman_cache] = kalman_update_1d_new(x_est, P, output, noisy_z, param, true_pos);
                
                % 把当前步的预测状态存入 kalman_cache 方便下次用
                kalman_cache.x_pred = x_pred;
                
                % 计算位置误差
                pos_error = abs(true_pos - x_est(1)); % 绝对误差
                
                % 缓存数据
                cache(k).input = current_input;
                cache(k).hidden = hidden;
                cache(k).output = output;
                cache(k).pos_error = pos_error;
                cache(k).kalman_cache = kalman_cache;
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
                        rknn_backward_1d_new(cache(window_start:window_end), window_size, param, net);
                    
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
    fprintf('一维训练完成! 最佳损失: %.4f\n', best_loss);
end
