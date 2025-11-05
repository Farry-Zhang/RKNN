function model= gen_model_gen
model.numSensor = 3;
% basic parameters
model.x_dim= 6;   %dimension of state vector
model.z_dim_A= 4;   %dimension of observation vector
model.z_dim_P= 2;   %dimension of observation vector
model.z_dim_E= 2;   %dimension of observation vector

model.wA_dim= 4;   %雷达dimension of observation noise
model.wP_dim= 2;   %红外dimension of observation noise
model.wE_dim= 2;   %红外dimension of observation noise

% dynamical model parameters (CA model)
model.T= 1; 
model.A0= [ 1 model.T ; 0 1];  
model.F= [ model.A0 zeros(2,4); zeros(2,2) model.A0 zeros(2,2);zeros(2,4) model.A0];

model.B0 = [ (model.T^2)/2; model.T];
%%%%x维度
model.B1 = [ model.B0 zeros(2,1) zeros(2,1)]; 
model.sigma_vx =1;
model.Q1 = (model.sigma_vx)^2*model.B1*model.B1';
%%%%y维度
model.B2 = [ zeros(2,1) model.B0 zeros(2,1)];
model.sigma_vy =1;
model.Q2 = (model.sigma_vy)^2*model.B2*model.B2';
%%%%z维度
model.B3 = [ zeros(2,1) zeros(2,1) model.B0];
model.sigma_vz =0.1; 
model.Q3 = (model.sigma_vz)^2*model.B3*model.B3';

model.Q = blkdiag(model.Q1,model.Q2,model.Q3);

model.B = [ model.B0 zeros(2,1) zeros(2,1); zeros(2,1) model.B0 zeros(2,1);zeros(2,1) zeros(2,1) model.B0];
model.sigma_v = [model.sigma_vx;model.sigma_vy;model.sigma_vz];


model.D_A= diag([10; 0.2; 0.2; 3]); 
model.R_A= model.D_A*model.D_A';              %雷达observation noise covariance

model.D_P= diag([0.01; 0.01]);
model.R_P= model.D_P*model.D_P';              %红外observation noise covariance

model.D_E= diag([0.1; 0.1]);
model.R_E= model.D_P*model.D_P';              %侦察observation noise covariance

% detection parameters
model.P_D= [0.98 0.98 0.98];   %probability of detection in measurements
model.Q_D= 1-model.P_D; %probability of missed detection in measurements


% clutter parameters
model.lambda_c= [20 1 1];                             %poisson average rate of uniform clutter (per scan)
model.range_c= [ 20000 85000;-60000 60000;4000 5000];          %uniform clutter on r/theta

