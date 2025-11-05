function X= gen_newstate_target(model,Xd,V)

%nonlinear state space equation (CT model)
if ~isnumeric(V)
    if strcmp(V,'noise')
        V1 = model.sigma_vx*model.B1*randn(size(model.B1,2),size(Xd,2));
        V2 = model.sigma_vy*model.B2*randn(size(model.B2,2),size(Xd,2));
        V3 = model.sigma_vz*model.B3*randn(size(model.B3,2),size(Xd,2));
        V = [V1;V2;V3];
    elseif strcmp(V,'noiseless')
        V= zeros(size(model.B,1),size(Xd,2));
    end
end

if isempty(Xd)
    X= [];
else %modify below here for user specified transition model
    X= model.F*Xd+ V;
end

end
