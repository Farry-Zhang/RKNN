'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1️⃣ 生成二维机动目标轨迹
# ============================================================
def generate_maneuvering_trajectory_2d(seq_len=200, dt=1.0, n_maneuvers=3):
    x = np.zeros((seq_len, 6))
    x[0, :] = [0, 0, np.random.uniform(10, 20),
               np.random.uniform(5, 10),
               np.random.uniform(-1, 1),
               np.random.uniform(-1, 1)]
    maneuver_points = np.sort(np.random.choice(np.arange(20, seq_len - 20), n_maneuvers, replace=False))
    for t in range(1, seq_len):
        A = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        x[t] = A @ x[t-1]
        if t in maneuver_points:
            x[t,4:] += np.random.uniform(-2,2, size=2)
    return x

# ============================================================
# 2️⃣ Kalman滤波器
# ============================================================
def kalman_filter_predict_update(x, P, z, A, H, Q, R):
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(len(x)) - K @ H) @ P_pred
    return x_new, P_new, x_pred, y

# ============================================================
# 3️⃣ RKNN 神经网络
# ============================================================
class RKNN2D(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=40, output_dim=4):
        super(RKNN2D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ============================================================
# 4️⃣ 构建训练样本（使用合理标签）
# ============================================================
def build_training_set(n_samples=20, seq_len=200):
    data_x, data_y = [], []
    dt = 1.0
    A = np.array([
        [1, 0, dt, 0, 0.5*dt**2, 0],
        [0, 1, 0, dt, 0, 0.5*dt**2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ])
    H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
    
    for _ in range(n_samples):
        x_true = generate_maneuvering_trajectory_2d(seq_len)
        R_meas = np.diag([10,10])
        z = x_true[:,:2] + np.random.multivariate_normal([0,0], R_meas, seq_len)
        Q0 = np.diag([0.01,0.01,0.01,0.01,0.001,0.001])
        P = np.eye(6)
        x = x_true[0]

        for t in range(1, seq_len):
            # 先使用固定Q0, R_meas进行Kalman预测
            x, P, x_pred, e = kalman_filter_predict_update(x, P, z[t], A, H, Q0, R_meas)

            dx_pred = (x_pred - x_true[t-1]) / (np.abs(x_true[t-1]) + 1e-3)
            e_norm = e / (np.linalg.norm(e) + 1e-6)
            z_norm = (z[t] - z[t-1]) / (np.linalg.norm(z[t-1]) + 1e-6)
            feature = np.concatenate([dx_pred, e_norm, z_norm])
            data_x.append(feature)

            # --------- 合理目标标签：最小化下一步位置误差 ---------
            # 简单搜索方法：在一定范围内找使误差最小的 α,β
            best_loss = float('inf')
            best_params = None
            for alpha_x in np.linspace(0,0.5,5):
                for alpha_y in np.linspace(0,0.5,5):
                    for beta_x in np.linspace(0,1,5):
                        for beta_y in np.linspace(0,1,5):
                            Q_try = np.diag([alpha_x, alpha_y,0.01,0.01,0.001,0.001])
                            R_try = np.diag([beta_x*10,beta_y*10])
                            x_try, _, _, _ = kalman_filter_predict_update(x, P, z[t], A, H, Q_try, R_try)
                            loss = np.linalg.norm(x_try[:2]-x_true[t,:2])
                            if loss < best_loss:
                                best_loss = loss
                                best_params = [alpha_x, alpha_y, beta_x, beta_y]
            data_y.append(best_params)

    inputs = torch.tensor(np.array(data_x), dtype=torch.float32)
    targets = torch.tensor(np.array(data_y), dtype=torch.float32)
    dataset = [(inputs[i].unsqueeze(0), targets[i].unsqueeze(0)) for i in range(len(inputs))]
    return dataset

# ============================================================
# 5️⃣ 训练函数
# ============================================================
def train_rknn(model, train_data, epochs=5, lr=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (inputs, targets) in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data):.6f}")
    print("✅ 训练完成！")
    return model

# ============================================================
# 6️⃣ 测试与可视化
# ============================================================
def test_rknn(model):
    model.eval()
    x_true = generate_maneuvering_trajectory_2d(seq_len=200)
    dt = 1.0
    A = np.array([
        [1,0,dt,0,0.5*dt**2,0],
        [0,1,0,dt,0,0.5*dt**2],
        [0,0,1,0,dt,0],
        [0,0,0,1,0,dt],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ])
    H = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0]])
    Q0 = np.diag([0.01,0.01,0.01,0.01,0.001,0.001])
    R0 = np.diag([10,10])
    z = x_true[:,:2] + np.random.multivariate_normal([0,0], R0, len(x_true))
    x = x_true[0]
    P = np.eye(6)
    est_traj = []
    alpha_beta_list = []

    for t in range(1, len(x_true)):
        dx_pred = (A@x - x)/(np.abs(x)+1e-3)
        e = z[t]-H@(A@x)
        e_norm = e / (np.linalg.norm(e)+1e-6)
        z_norm = (z[t]-z[t-1])/(np.linalg.norm(z[t-1])+1e-6)
        feature = np.concatenate([dx_pred,e_norm,z_norm])
        feature = torch.tensor(feature, dtype=torch.float32, device=device).unsqueeze(0)
        ab = model(feature).detach().cpu().numpy().squeeze()

        αx, αy, βx, βy = ab
        alpha_beta_list.append(ab)
        Q = np.diag([αx, αy,0.01,0.01,0.001,0.001])
        R = np.diag([βx*10,βy*10])
        x, P, x_pred, e = kalman_filter_predict_update(x,P,z[t],A,H,Q,R)
        est_traj.append(x[:2])

    est_traj = np.array(est_traj)
    alpha_beta_list = np.array(alpha_beta_list)

    # ----------- 轨迹可视化 ----------- #
    plt.figure(figsize=(8,6))
    plt.plot(x_true[:,0], x_true[:,1], 'k-', label="True Trajectory")
    plt.plot(z[:,0], z[:,1], 'r.', alpha=0.3, label="Measurements")
    plt.plot(est_traj[:,0], est_traj[:,1], 'b--', label="RKNN Estimate")
    plt.legend()
    plt.title("2D RKNN Kalman Tracking")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()
# ----------- 各参数单独画图 ----------- #
    param_names = ["αx", "αy", "βx", "βy"]
    for i in range(4):
        plt.figure(figsize=(6, 4))
        plt.plot(alpha_beta_list[:, i], label=f"{param_names[i]} Value")
        plt.title(f"Adaptive Parameter: {param_names[i]}")
        plt.xlabel("Time step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    # ----------- RMSE 计算 ----------- #
    overall_rmse = np.sqrt(np.mean((x_true[1:len(est_traj)+1,:2]-est_traj)**2))
    print(f"Overall 2D Trajectory RMSE = {overall_rmse:.6f}")

# ============================================================
# 7️⃣ 主程序入口
# ============================================================
if __name__ == "__main__":
    print("🔧 构建训练数据...")
    train_data = build_training_set(n_samples=5, seq_len=100)  # 样本数和序列长度可调

    print("🧠 初始化模型...")
    model = RKNN2D().to(device)

    print("🚀 开始训练...")
    model = train_rknn(model, train_data, epochs=5, lr=1e-3)

    print("🔍 测试与可视化...")
    test_rknn(model)

'''


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1️⃣ 生成二维机动轨迹 =================
def generate_maneuvering_trajectory_2d(seq_len=200, dt=1.0, n_maneuvers=3):
    x = np.zeros((seq_len, 6))
    x[0, :] = [0, 0, np.random.uniform(10, 20),
               np.random.uniform(5, 10),
               np.random.uniform(-1, 1),
               np.random.uniform(-1, 1)]
    maneuver_points = np.sort(np.random.choice(np.arange(20, seq_len - 20), n_maneuvers, replace=False))
    for t in range(1, seq_len):
        A = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        x[t] = A @ x[t-1]
        if t in maneuver_points:
            x[t,4:] += np.random.uniform(-2,2, size=2)
    return x

# ================= 2️⃣ Kalman滤波器 =================
def kalman_filter_predict_update(x, P, z, A, H, Q, R):
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_new = x_pred + K @ y
    P_new = (np.eye(len(x)) - K @ H) @ P_pred
    return x_new, P_new, x_pred, y

# ================= 3️⃣ RKNN 网络 =================
class RKNN2D(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=40, output_dim=4):
        super(RKNN2D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ================= 4️⃣ 构建训练数据 =================
def build_training_data(n_samples=5, seq_len=100):
    data_x = []
    for _ in range(n_samples):
        x_true = generate_maneuvering_trajectory_2d(seq_len)
        data_x.append(x_true)
    return data_x

# ================= 5️⃣ 端到端训练函数 =================
def train_rknn_end2end(model, trajectories, epochs=5, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dt = 1.0
    A = torch.tensor([
        [1,0,dt,0,0.5*dt**2,0],
        [0,1,0,dt,0,0.5*dt**2],
        [0,0,1,0,dt,0],
        [0,0,0,1,0,dt],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ], dtype=torch.float32, device=device)
    H = torch.tensor([[1,0,0,0,0,0],[0,1,0,0,0,0]], dtype=torch.float32, device=device)

    model.train()
    mse_list = []  # 📈 保存每个epoch的平均MSE

    for epoch in range(epochs):
        total_loss = 0
        for x_true_np in tqdm(trajectories, desc=f"Epoch {epoch+1}/{epochs}"):
            x_true = torch.tensor(x_true_np, dtype=torch.float32, device=device)
            seq_len = x_true.shape[0]

            x = x_true[0].clone()
            P = torch.eye(6, device=device)
            z = x_true[:,:2] + torch.randn(seq_len,2,device=device)*3.0  # 测量噪声
            est_traj = []

            for t in range(1, seq_len):
                dx_pred = (A @ x - x)/(torch.abs(x)+1e-3)
                e = z[t]-H@(A@x)
                e_norm = e/(torch.norm(e)+1e-6)
                z_norm = (z[t]-z[t-1])/(torch.norm(z[t-1])+1e-6)
                feature = torch.cat([dx_pred, e_norm, z_norm]).unsqueeze(0)
                ab = model(feature)[0]

                alpha_x, alpha_y, beta_x, beta_y = ab
                Q = torch.diag(torch.stack([alpha_x, alpha_y, torch.tensor(0.01, device=device),
                                            torch.tensor(0.01, device=device),
                                            torch.tensor(0.001, device=device),
                                            torch.tensor(0.001, device=device)]))
                R = torch.diag(torch.stack([beta_x*10, beta_y*10]))

                x_pred = A @ x
                P_pred = A @ P @ A.T + Q
                y = z[t]-H@x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T @ torch.linalg.inv(S)
                x = x_pred + K @ y
                P = (torch.eye(6, device=device)-K@H)@P_pred
                est_traj.append(x[:2])

            est_traj = torch.stack(est_traj)
            loss = loss_fn(est_traj, x_true[1:,:2])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trajectories)
        mse_list.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

    # 📉 绘制MSE曲线
    plt.figure(figsize=(6,4))
    plt.plot(range(1, epochs+1), mse_list, marker='o', color='b')
    plt.title("Training MSE Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

    print("✅ 训练完成！")
    return model

# ================= 6️⃣ 测试与可视化 =================
def test_rknn(model):
    model.eval()
    x_true_np = generate_maneuvering_trajectory_2d(seq_len=200)
    x_true = torch.tensor(x_true_np, dtype=torch.float32, device=device)
    dt = 1.0
    A = torch.tensor([
        [1,0,dt,0,0.5*dt**2,0],
        [0,1,0,dt,0,0.5*dt**2],
        [0,0,1,0,dt,0],
        [0,0,0,1,0,dt],
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ], dtype=torch.float32, device=device)
    H = torch.tensor([[1,0,0,0,0,0],[0,1,0,0,0,0]], dtype=torch.float32, device=device)

    x = x_true[0].clone()
    P = torch.eye(6, device=device)
    z = x_true[:,:2] + torch.randn_like(x_true[:,:2])*3.0
    est_traj = []

    for t in range(1, len(x_true)):
        dx_pred = (A@x - x)/(torch.abs(x)+1e-3)
        e = z[t]-H@(A@x)
        e_norm = e/(torch.norm(e)+1e-6)
        z_norm = (z[t]-z[t-1])/(torch.norm(z[t-1])+1e-6)
        feature = torch.cat([dx_pred, e_norm, z_norm]).unsqueeze(0)
        ab = model(feature)[0].detach()
        alpha_x, alpha_y, beta_x, beta_y = ab

        Q = torch.diag(torch.tensor([alpha_x, alpha_y, 0.01,0.01,0.001,0.001], device=device))
        R = torch.diag(torch.tensor([beta_x*10, beta_y*10], device=device))
        x_pred = A@x
        P_pred = A@P@A.T + Q
        y = z[t]-H@x_pred
        S = H@P_pred@H.T + R
        K = P_pred@H.T@torch.linalg.inv(S)
        x = x_pred + K@y
        P = (torch.eye(6, device=device)-K@H)@P_pred
        est_traj.append(x[:2])

    est_traj = torch.stack(est_traj).cpu().numpy()
    plt.figure(figsize=(8,6))
    plt.plot(x_true_np[:,0], x_true_np[:,1], 'k-', label="True Trajectory")
    plt.plot(z.cpu()[:,0], z.cpu()[:,1], 'r.', alpha=0.3, label="Measurements")
    plt.plot(est_traj[:,0], est_traj[:,1], 'b--', label="RKNN Estimate")
    plt.legend()
    plt.grid(True)
    plt.show()

# ================= 7️⃣ 主程序 =================
if __name__ == "__main__":
    print("🔧 构建训练数据...")
    train_data = build_training_data(n_samples=5, seq_len=100)
    print("🧠 初始化模型...")
    model = RKNN2D().to(device)
    print("🚀 开始端到端训练...")
    model = train_rknn_end2end(model, train_data, epochs=10, lr=1e-3)
    print("🔍 测试与可视化...")
    test_rknn(model)
