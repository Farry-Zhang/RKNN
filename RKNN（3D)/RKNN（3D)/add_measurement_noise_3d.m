%  三维噪声添加函数 === (论文3.2节)
function noisy_inputs = add_measurement_noise_3d(inputs, noise_var)
    % inputs: [batch_size, time_steps, 15] - 输入特征
    % noise_var: 噪声方差
    
    % 添加量测噪声到所有输入特征
    noise = sqrt(noise_var) * randn(size(inputs));
    noisy_inputs = inputs + noise;
end