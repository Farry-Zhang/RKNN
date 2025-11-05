import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ------------------ Model definition ------------------
class RKNNModule(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=20, output_dim=2):
        super().__init__()
        # First path: 4 -> hidden_dim
        self.path1 = nn.Linear(4, hidden_dim)
        # Second path: 1 -> hidden_dim
        self.path2 = nn.Linear(1, hidden_dim)
        # Output layer: hidden_dim -> 2
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: tensor shape (..., 5)
        x_normal = x[..., :3]                 # (...,3)
        e_normal = x[..., 3].unsqueeze(-1)    # (...,1)
        z_normal = x[..., 4].unsqueeze(-1)    # (...,1)

        in1 = torch.cat([x_normal, e_normal], dim=-1)  # (...,4)
        out1 = self.path1(in1)                          # (...,hidden)
        out2 = self.path2(z_normal)                     # (...,hidden)

        hidden = torch.tanh(out1 + out2)                # (...,hidden)
        out3 = self.out(hidden)                         # (...,2)
        output = torch.sigmoid(out3)                    # (...,2) in (0,1)
        return output, hidden

# ------------------ Kalman update implemented with torch ------------------

def kalman_update_torch(x_prev, P_prev, rknn_output, z, params):
    # x_prev: (3,) tensor
    # P_prev: (3,3) tensor
    # rknn_output: (2,) tensor with alpha, beta in (0,1)
    T = params['T']
    Q0 = params['Q0']
    R0 = params['R0']

    H = torch.tensor([[1., 0., 0.]], dtype=x_prev.dtype, device=x_prev.device)  # (1,3)
    I = torch.eye(3, dtype=x_prev.dtype, device=x_prev.device)

    alpha = rknn_output[0]
    beta = rknn_output[1]

    F = torch.tensor([
        [1., T, 0.5 * T**2],
        [0., 1., T],
        [0., 0., 1.]
    ], dtype=x_prev.dtype, device=x_prev.device)

    Q = alpha * Q0
    R = beta * R0

    # Prediction
    x_pred = F @ x_prev
    P_pred = F @ P_prev @ F.t() + Q

    # Innovation
    v = z - (H @ x_pred).squeeze()

    S = (H @ P_pred @ H.t()).squeeze() + R

    # Kalman gain K: (3,1)
    K = (P_pred @ H.t()) / S

    # Update
    x_est = x_pred + (K * v).squeeze(-1)
    P = (I - K @ H) @ P_pred

    b = -(x_pred[0] - x_est[0])

    cache = {
        'x_prev': x_prev.detach().cpu().numpy(),
        'x_pred': x_pred.detach().cpu().numpy(),
        'P_pred': P_pred.detach().cpu().numpy(),
        'K': K.detach().cpu().numpy(),
        'S': S.detach().cpu().numpy(),
        'v': v.detach().cpu().numpy(),
        'H': H.detach().cpu().numpy(),
        'F': F.detach().cpu().numpy(),
        'Q': Q.detach().cpu().numpy(),
        'R': R.detach().cpu().numpy(),
        'b': b.detach().cpu().numpy()
    }

    return x_est, P, cache

# ------------------ Data generation (keeps numpy for randomness but returns torch) ------------------

def generate_maneuvering_trajectory(params, num_points=200):
    positions = np.zeros(num_points)
    velocities = np.zeros(num_points)
    accelerations = np.zeros(num_points)

    positions[0] = 5000 + 2000 * np.random.rand()
    velocities[0] = -100 + 200 * np.random.rand()

    num_maneuvers = np.random.randint(3, 6)
    maneuver_times = np.sort(np.random.choice(range(5, num_points-5), num_maneuvers, replace=False))

    current_acc = 0.0

    for k in range(1, num_points):
        if k in maneuver_times:
            current_acc = -30 + 60 * np.random.rand()
        velocities[k] = velocities[k-1] + current_acc * params['T']
        positions[k] = positions[k-1] + velocities[k-1]*params['T'] + 0.5*current_acc*(params['T']**2)
        accelerations[k] = current_acc
        if abs(velocities[k]) > params['max_velocity']:
            velocities[k] = np.sign(velocities[k]) * params['max_velocity']

    return positions, velocities, accelerations

class RKDataset(Dataset):
    def __init__(self, num_trajectories, traj_length, params):
        self.inputs = []
        self.targets = []
        for _ in range(num_trajectories):
            positions, velocities, accelerations = generate_maneuvering_trajectory(params, num_points=traj_length)
            for k in range(traj_length):
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
                    e_norm = 0.0
                    z_norm = 0.0
                state_inc_norm = state_inc / params['max_velocity']
                inp = np.hstack((state_inc_norm, e_norm, z_norm)).astype(np.float32)
                tgt = np.array([positions[k]]).astype(np.float32)
                self.inputs.append(inp)
                self.targets.append(tgt)
        self.inputs = torch.from_numpy(np.stack(self.inputs))
        self.targets = torch.from_numpy(np.stack(self.targets))

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# ------------------ Training routine ------------------

def train_pytorch_rknn(params, device='cpu'):
    torch.manual_seed(0)

    model = RKNNModule(input_dim=5, hidden_dim=params['hidden_dim'], output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['num_epochs'])

    dataset = RKDataset(num_trajectories=params['train_trajectories'], traj_length=params['traj_length'], params=params)
    dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=True)

    best_loss = float('inf')
    for epoch in range(params['num_epochs']):
        epoch_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device).squeeze(-1)

            # Add measurement noise to z (last element of input is z_norm). We perturb the input's z_norm
            noisy_inputs = batch_inputs.clone()
            noise = torch.randn_like(noisy_inputs[..., 4]) * np.sqrt(params['r0']) / params['max_velocity']
            noisy_inputs[..., 4] += noise

            # Initialize Kalman for each sample in batch
            batch_size = noisy_inputs.shape[0]
            x_est = torch.zeros(batch_size, 3, device=device)
            P = torch.stack([torch.diag(torch.tensor([1000., 100., 10.], device=device)) for _ in range(batch_size)])

            optimizer.zero_grad()

            losses = []
            # process sequence as independent samples (flattened trajectories)
            # each dataset item is a single timestep; for simplicity we treat each item independently
            # so Kalman here acts on single-step measurements
            rknn_out, _ = model(noisy_inputs)
            # rknn_out shape: (batch_size, 2)
            for i in range(batch_size):
                # treat as one-step Kalman update
                xi = x_est[i]
                Pi = P[i]
                outi = rknn_out[i]
                zi = noisy_inputs[i, 4] * params['max_velocity']  # denormalize measurement
                zi = torch.tensor(zi, device=device)
                x_new, P_new, _ = kalman_update_torch(xi, Pi, outi, zi, params)
                # Loss on position
                loss_i = (x_new[0] - batch_targets[i])**2
                losses.append(loss_i)

            loss = torch.mean(torch.stack(losses))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_inputs.shape[0]

        scheduler.step()
        avg_loss = epoch_loss / len(dataset)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch+1}/{params['num_epochs']} - AvgLoss: {avg_loss:.6f} - LR: {scheduler.get_last_lr()[0]:.6f}")

    model.load_state_dict(best_state)
    return model

# ------------------ Evaluation ------------------

def evaluate_model(model, params, device='cpu'):
    model.eval()
    test_positions_est = []
    test_positions_true = []

    for _ in range(params['test_trajectories']):
        positions, velocities, accelerations = generate_maneuvering_trajectory(params, num_points=200)
        x_est = torch.zeros(3, device=device)
        P = torch.diag(torch.tensor([1000., 100., 10.], device=device))
        est_traj = []
        true_traj = []
        for k in range(200):
            if k > 0:
                state_inc = np.array([
                    positions[k] - positions[k-1],
                    positions[k] - positions[k-1],
                    positions[k] - positions[k-1]
                ])
                e_norm = abs(positions[k] - positions[k-1]) / params['max_velocity']
                z_norm = (positions[k] - positions[k-1]) / params['max_velocity']
            else:
                state_inc = np.zeros(3)
                e_norm = 0.0
                z_norm = 0.0
            inp = np.hstack((state_inc/params['max_velocity'], e_norm, z_norm)).astype(np.float32)
            inp_t = torch.from_numpy(inp).to(device)
            with torch.no_grad():
                rknn_out, _ = model(inp_t)
                noisy_z = positions[k] + np.sqrt(params['r0'])*np.random.randn()
                noisy_z_t = torch.tensor(noisy_z, device=device, dtype=inp_t.dtype)
                x_est, P, _ = kalman_update_torch(x_est, P, rknn_out.squeeze(), noisy_z_t, params)
            est_traj.append(x_est[0].item())
            true_traj.append(positions[k])
        test_positions_est.append(est_traj)
        test_positions_true.append(true_traj)

    est = np.array(test_positions_est)
    true = np.array(test_positions_true)
    errors = np.abs(est - true)
    avg_l1 = np.mean(errors)
    rmse_per_t = np.sqrt(np.mean((est - true)**2, axis=0))
    avg_rmse = np.mean(rmse_per_t)
    print(f'Avg L1 error: {avg_l1:.4f} m')
    print(f'Avg RMSE: {avg_rmse:.4f} m')

    # plot one trajectory
    idx = np.random.randint(len(est))
    plt.figure(figsize=(10,5))
    plt.plot(true[idx], label='True')
    plt.plot(est[idx], label='Estimated')
    plt.legend()
    plt.title('trajectory')
    plt.show()

# ------------------ Run script ------------------
if __name__ == '__main__':
    params = {
        'input_dim': 5,
        'hidden_dim': 20,
        'output_dim': 2,
        'max_velocity': 400.0,
        'T': 0.5,
        'q0': 9.0,
        'r0': 4900.0,
        'train_trajectories': 200,
        'traj_length': 100,
        'batch_size': 256,
        'num_epochs': 10,
        'learning_rate': 0.001,
        'test_trajectories': 10
    }

    # compute Q0 and R0 as tensors
    Gamma0 = np.array([[0.5 * params['T']**2], [params['T']], [1.0]])
    Q0 = params['q0'] * (Gamma0 @ Gamma0.T)
    params['Q0'] = torch.from_numpy(Q0).float()
    params['R0'] = torch.tensor(params['r0']).float()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train_pytorch_rknn(params, device=device)
    evaluate_model(model, params, device=device)
