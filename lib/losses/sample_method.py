import torch
import numpy as np

def distr_loss(V_pred, V_tr):
    # V_pred: [bs, 8, 12, 5]
    # V_tr: [bs, 8, 12, 2]
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_tr.shape
    normx = V_tr[:,:,:,0]- V_pred[:,:,:,0]
    normy = V_tr[:,:,:,1]- V_pred[:,:,:,1]

    sx = torch.exp(V_pred[:,:,:,2]) #sx
    sy = torch.exp(V_pred[:,:,:,3]) #sy
    corr = torch.tanh(V_pred[:,:,:,4]) #corr

    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho)) # 0~1
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom
    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result

def multipred_loss(pred_traj, target):
    K = pred_traj.shape[3]
    obs_len = pred_traj.shape[1]
    target = target.unsqueeze(3).repeat(1, 1, 1, K, 1)
    total_loss = []
    for enc_step in range(obs_len):
        traj_rmse = torch.sqrt(
            torch.sum((pred_traj[:, enc_step, :, :, :] - target[:, enc_step, :, :, :]) ** 2, dim=-1)).sum(dim=1)
        best_idx = torch.argmin(traj_rmse, dim=1)
        loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
        total_loss.append(loss_traj)

    return sum(total_loss) / len(total_loss)