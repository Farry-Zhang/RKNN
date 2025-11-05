import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# 前向传播函数
def runRKNN(net, input_vec):
    # 提取网络参数
    Linear1_w = net['Linear1_weight']  # 20x4
    Linear1_b = net['Linear1_bias']    # 20x1
    Linear2_w = net['Linear2_weight']  # 20x1
    Linear2_b = net['Linear2_bias']    # 20x1
    Linear3_w = net['Linear3_weight']  # 2x20
    Linear3_b = net['Linear3_bias']    # 2x1
    
    # 拆分输入
    x_normal = input_vec[:3]    # 状态增量
    e_normal = input_vec[3]     # 新息范数
    z_normal = input_vec[4]     # 量测增量
    
    # 第一路径处理 (公式25)
    in1 = np.hstack((x_normal, e_normal))
    in2 = [[x] for x in in1]
    out1 = Linear1_w @ in2 + Linear1_b  #in1有问题
    
    # 第二路径处理 (公式in1 26)
    out2 = Linear2_w * z_normal + Linear2_b
    
    # 合并路径并应用Tanh (隐藏层输出)
    hidden = np.tanh(out1 + out2)
    
    # 输出层处理 (公式27)
    out3 = Linear3_w @ hidden + Linear3_b
    output = 1 / (1 + np.exp(-out3))  # Sigmoid激活
    
    return output, hidden

# Kalman更新函数
def kalman_update(x_prev, P_prev, rknn_output, z, params):
    T = params['T']
    Q0 = params['Q0']
    R0 = params['R0']
    H = np.array([[1, 0, 0]])  # 观测矩阵
    I = np.eye(3)              # 单位矩阵
    
    # 解析RKNN输出
    alpha = rknn_output[0]
    beta = rknn_output[1]
    
    # 状态转移矩阵
    F = np.array([
        [1, T, 0.5*T**2],
        [0, 1, T],
        [0, 0, 1]
    ])
    
    # 动态噪声矩阵
    Q = alpha * Q0
    R = beta * R0
    
    # 预测步骤
    x_pred = F @ x_prev
    P_pred = F @ P_prev @ F.T + Q
    
    # 计算残差 (新息)
    v = z - H @ x_pred
    
    # Kalman增益
    S = H @ P_pred @ H.T + R
    K = (P_pred @ H.T)/S
    
    # 状态更新
    x_est = x_pred + K @ v
    P = (I - K @ H) @ P_pred
    
    # 计算b(k+1)
    b = -(x_pred[0] - x_est[0])
    
    # 缓存中间变量
    cache = {
        'x_prev': x_prev, 'x_pred': x_pred, 'P_pred': P_pred,
        'K': K, 'S': S, 'v': v, 'H': H, 'F': F,
        'Q': Q, 'R': R, 'b': b
    }
    
    return x_est, P, cache

# 反向传播函数
def rknn_backward(cache, window_size, params):
    # 初始化梯度
    grad_L1_w = np.zeros_like(cache[0]['Linear1_weight']).astype(np.float64)
    grad_L1_b = np.zeros_like(cache[0]['Linear1_bias']).astype(np.float64)
    grad_L2_w = np.zeros_like(cache[0]['Linear2_weight']).astype(np.float64)
    grad_L2_b = np.zeros_like(cache[0]['Linear2_bias']).astype(np.float64)
    grad_L3_w = np.zeros_like(cache[0]['Linear3_weight']).astype(np.float64)
    grad_L3_b = np.zeros_like(cache[0]['Linear3_bias']).astype(np.float64)
    
    for t in range(window_size-1, -1, -1):
        kalman_cache = cache[t]['kalman_cache']
        v = kalman_cache['v']
        H = kalman_cache['H']
        P_pred = kalman_cache['P_pred']
        S = kalman_cache['S']
        b = kalman_cache['b']
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
    # 奇异矩阵处理
        S_inv = np.linalg.pinv(S)  # 或添加正则化
        # 计算dL/dalpha
        term1 = np.eye(H.shape[0]) - H.T @ S_inv.T @ H @ P_pred
        term2 = P_pred @ H.T @ S.T @ v - b
        term3 = v.T @ S_inv.T @ H
        try1 = term1 @ term2 @ term3
        dL_dalpha = params['Q0'][0,0] * try1
        
        # 计算dL/dbeta
        term1_beta = -2 * S_inv.T @ H @ P_pred
        term2_beta = P_pred @ H.T @ S.T @ v - b
        term3_beta = v.T @ S_inv.T 
        try2 = term1_beta @ term2_beta @ term3_beta
        dL_dbeta = -params['R0'] *try2
        
        # 组合输出梯度
        dL_doutput = [[dL_dalpha], [dL_dbeta]]
        
        # Sigmoid导数
        dsigmoid = cache[t]['output'] * (1 - cache[t]['output'])
        d_output = dL_doutput * dsigmoid
        #d_output should be 2*1

        # 输出层反向传播
        grad_L3_w += d_output * cache[t]['hidden'].T #'hidden' is 20*1,turn to 1*20. 
        grad_L3_b += d_output
        
        # 隐藏层梯度
        d_hidden = cache[t]['Linear3_weight'].T @ d_output
        d_hidden = d_hidden * (1 - cache[t]['hidden']**2)
        
        # 第二路径梯度
        grad_L2_w += np.outer(d_hidden, np.array([cache[t]['input'][4]]))
        grad_L2_b +=  d_hidden
        
        # 第一路径梯度
        in1 = np.hstack((cache[t]['input'][:3], cache[t]['input'][3]))
        grad_L1_w += np.outer(d_hidden, in1)
        grad_L1_b +=  d_hidden
    
    # 平均梯度
    grad_L1_w /= window_size
    grad_L1_b /= window_size
    grad_L2_w /= window_size
    grad_L2_b /= window_size
    grad_L3_w /= window_size
    grad_L3_b /= window_size
    
    return grad_L1_w, grad_L1_b, grad_L2_w, grad_L2_b, grad_L3_w, grad_L3_b

# 轨迹生成函数
def generate_maneuvering_trajectory(params):
    num_points = 200
    positions = np.zeros(num_points)
    velocities = np.zeros(num_points)
    accelerations = np.zeros(num_points)
    
    # 初始状态
    positions[0] = 5000 + 2000 * np.random.rand()
    velocities[0] = -100 + 200 * np.random.rand()
    
    # 机动参数
    num_maneuvers = np.random.randint(3, 6)
    maneuver_times = np.sort(np.random.choice(range(5, 195), num_maneuvers, replace=False))
    
    current_acc = 0
    
    for k in range(1, num_points):
        # 检查是否机动时刻
        if k in maneuver_times:
            current_acc = -30 + 60 * np.random.rand()
        
        # 更新状态
        velocities[k] = velocities[k-1] + current_acc * params['T']
        positions[k] = positions[k-1] + velocities[k-1]*params['T'] + 0.5*current_acc*(params['T']**2)
        accelerations[k] = current_acc
        
        # 速度限制
        if abs(velocities[k]) > params['max_velocity']:
            velocities[k] = np.sign(velocities[k]) * params['max_velocity']
    
    return positions, velocities, accelerations

# 训练数据生成
def get_rknn_training_data(num_trajectories, params):
    traj_length = 100
    sampling_rate = 2
    num_points = traj_length * sampling_rate
    inputs = np.zeros((num_trajectories, num_points, 5))
    targets = np.zeros((num_trajectories, num_points, 1))
    
    for b in range(num_trajectories):
        positions, velocities, accelerations = generate_maneuvering_trajectory(params)
        
        for k in range(num_points):
            if k > 0:
                state_inc = np.array([
                    positions[k] - positions[k-1],
                    velocities[k] - velocities[k-1],
                    accelerations[k] - accelerations[k-1]
                ])
                e_norm = abs(positions[k] - positions[k-1]) / params['max_velocity']
                z_norm = (positions[k] - positions[k-1]) / params['max_velocity']
            else:
                state_inc = np.zeros(3)
                e_norm = 0
                z_norm = 0
            
            state_inc_norm = state_inc / params['max_velocity']
            inputs[b, k] = np.hstack((state_inc_norm, e_norm, z_norm))
            targets[b, k, 0] = positions[k]
    
    return inputs, targets

# 测试数据生成
def get_rknn_test_data(num_trajectories, params):
    num_points = 200
    inputs = np.zeros((num_trajectories, num_points, 5))
    targets = np.zeros((num_trajectories, num_points, 1))
    
    for b in range(num_trajectories):
        positions, _, _ = generate_maneuvering_trajectory(params)
        
        for k in range(num_points):
            if k > 0:
                state_inc = np.array([
                    positions[k] - positions[k-1],
                    positions[k] - positions[k-1],  # 简化处理
                    positions[k] - positions[k-1]   # 简化处理
                ])
                e_norm = abs(positions[k] - positions[k-1]) / params['max_velocity']
                z_norm = (positions[k] - positions[k-1]) / params['max_velocity']
            else:
                state_inc = np.zeros(3)
                e_norm = 0
                z_norm = 0
            
            state_inc_norm = state_inc / params['max_velocity']
            inputs[b, k] = np.hstack((state_inc_norm, e_norm, z_norm))
            targets[b, k, 0] = positions[k]
    
    return inputs, targets

# 添加量测噪声
def add_measurement_noise(inputs, noise_var):
    noise = np.sqrt(noise_var) * np.random.randn(*inputs.shape)
    return inputs + noise

# 网络训练函数
def trainRKNN(best_net, param):
    num_epochs = 500
    learning_rate = 0.005
    window_size = 5
    batch_size = 100
    best_loss = float('inf')
    
    # 学习率调度 (余弦退火)
    lr_schedule = learning_rate * (1 + np.cos(np.linspace(0, np.pi, num_epochs))) / 2
    
    # 获取训练数据
    train_inputs, train_targets = get_rknn_training_data(batch_size, param)
    
    # Kalman初始化函数
    def initialize_kalman():
        return np.zeros(3), np.diag([1000, 100, 10])
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        noisy_inputs = add_measurement_noise(train_inputs, param['r0'])
        
        for batch in range(batch_size):
            x_est, P = initialize_kalman()
            cache = [{} for _ in range(noisy_inputs.shape[1])]
            
            for k in range(noisy_inputs.shape[1]):
                true_pos = train_targets[batch, k, 0]
                noisy_z = true_pos + np.sqrt(param['r0']) * np.random.randn()
                current_input = noisy_inputs[batch, k]
                
                # 前向传播
                output, hidden = runRKNN(net, current_input)
                
                # Kalman更新
                x_est, P, kalman_cache = kalman_update(x_est, P, output, noisy_z, param)
                pos_error = train_targets[batch, k, 0] - x_est[0]
                
                # 存储缓存
                cache[k] = {
                    'input': current_input,
                    'hidden': hidden,
                    'output': output,
                    'pos_error': pos_error,
                    'kalman_cache': kalman_cache,
                    'Linear1_weight': net['Linear1_weight'],
                    'Linear1_bias': net['Linear1_bias'],
                    'Linear2_weight': net['Linear2_weight'],
                    'Linear2_bias': net['Linear2_bias'],
                    'Linear3_weight': net['Linear3_weight'],
                    'Linear3_bias': net['Linear3_bias']
                }
                
                # 时间窗反向传播
                if k >= window_size:
                    window_start = k - window_size + 1
                    window_end = k
                    window_loss_val = sum(
                        abs(cache[t]['pos_error']) for t in range(window_start, window_end+1))
                    epoch_loss += window_loss_val
                    
                    # 反向传播
                    grads = rknn_backward(
                        cache[window_start:window_end+1], window_size, param)
                    
                    # 更新权重
                    net['Linear1_weight'] -= lr_schedule[epoch] * grads[0]
                    net['Linear1_bias'] -= lr_schedule[epoch] * grads[1]
                    net['Linear2_weight'] -= lr_schedule[epoch] * grads[2]
                    net['Linear2_bias'] -= lr_schedule[epoch] * grads[3]
                    net['Linear3_weight'] -= lr_schedule[epoch] * grads[4]
                    net['Linear3_bias'] -= lr_schedule[epoch] * grads[5]
        
        # 计算平均损失
        avg_epoch_loss = epoch_loss / (batch_size * noisy_inputs.shape[1])
        
        # 更新最佳网络
        if avg_epoch_loss < best_loss:
            best_net = net.copy()
            best_loss = avg_epoch_loss
        
        print(f'Epoch {epoch+1}/{num_epochs} | Loss: {best_loss:.4f} | LR: {lr_schedule[epoch]:.6f}')
    
    return best_net, None

# 主程序
if __name__ == "__main__":
    # 定义参数
    params = {
        'input_dim': 5,
        'hidden_dim': 20,
        'output_dim': 2,
        'max_velocity': 400,
        'T': 0.5,
        'q0': 9,
        'r0': 4900
    }
    
    # 计算噪声矩阵
    Gamma0 = np.array([[0.5 * params['T']**2], [params['T']], [1]])
    params['Q0'] = params['q0'] * (Gamma0 @ Gamma0.T)
    params['R0'] = params['r0']
    
    # 初始化网络
    net = {
        'Linear1_weight': np.random.randn(20, 4) * 0.1,
        'Linear1_bias': [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],
        'Linear2_weight': np.random.randn(20, 1) * 0.1,
        'Linear2_bias': [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]],
        'Linear3_weight': np.random.randn(2, 20) * 0.1,
        'Linear3_bias': [[0],[0]]
    }
    
    # 训练网络
    trained_net, _ = trainRKNN(net, params)
    
    # 测试网络
    test_len = 10  # 减少测试轨迹数量以加快速度
    test_inputs, test_targets = get_rknn_test_data(test_len, params)
    positions_est = np.zeros((test_len, 200))
    positions_true = np.zeros((test_len, 200))
    
    # Kalman初始化函数
    def initialize_kalman():
        return np.zeros(3), np.diag([1000, 100, 10])
    
    for b in range(test_len):
        x_est, P = initialize_kalman()
        for k in range(200):
            true_pos = test_targets[b, k, 0]
            noisy_z = true_pos + np.sqrt(params['r0']) * np.random.randn()
            current_input = test_inputs[b, k]
            
            # 前向传播
            rknn_output, _ = runRKNN(trained_net, current_input)
            
            # Kalman更新
            x_est, P, _ = kalman_update(x_est, P, rknn_output, noisy_z, params)
            
            positions_est[b, k] = x_est[0]
            positions_true[b, k] = test_targets[b, k, 0]
    
    # 计算性能指标
    errors = np.abs(positions_est - positions_true)
    avg_l1_error = np.mean(errors)
    print(f'** 平均L1跟踪误差: {avg_l1_error:.4f} m **')
    
    squared_errors = (positions_est - positions_true)**2
    rmse_per_timestep = np.sqrt(np.mean(squared_errors, axis=0))
    avg_rmse = np.mean(rmse_per_timestep)
    print(f'** 平均RMSE: {avg_rmse:.4f} m **')
    
    # 可视化结果
    plot_idx = np.random.randint(test_len)
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(200), positions_true[plot_idx], 'b-', linewidth=2, label='真实位置')
    plt.plot(range(200), positions_est[plot_idx], 'r--', linewidth=1.5, label='RKNN估计')
    plt.xlabel('时间步')
    plt.ylabel('位置 (m)')
    plt.title(f'轨迹 {plot_idx} 跟踪结果')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(200), rmse_per_timestep, 'k-', linewidth=2)
    plt.xlabel('时间步')
    plt.ylabel('RMSE (m)')
    plt.title('跟踪RMSE随时间变化')
    plt.grid(True)
    plt.show()