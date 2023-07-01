import sys
import os
import os.path as osp
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributions.multivariate_normal as torchdist

from lib.utils.eval_utils import eval_ethucy_cvae, eval_ethucy_cvae_output_mintraj
from lib.losses import distr_loss, multipred_loss, cvae_multi

# The code of NPSN part refers to https://github.com/inhwanbae/NPSN
def box_muller_transform(x: torch.FloatTensor):
    r"""Box-Muller transform"""
    shape = x.shape
    x = x.view(shape[:-1] + (-1, 2))
    z = torch.zeros_like(x, device=x.device)
    z[..., 0] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).cos()
    z[..., 1] = (-2 * x[..., 0].log()).sqrt() * (2 * np.pi * x[..., 1]).sin()
    return z.view(shape)

def generate_statistics_matrices(V):
    r"""generate mean and covariance matrices from the network output.
    V: [bs, obs_len, 12, 5]
    """

    mu = V[:, :, :, 0:2]
    sx = V[:, :, :, 2].exp()
    sy = V[:, :, :, 3].exp()
    corr = V[:, :, :, 4].tanh()

    cov = torch.zeros(V.size(0), V.size(1), V.size(2), 2, 2, device=V.device)
    cov[:, :, :, 0, 0] = sx * sx
    cov[:, :, :, 0, 1] = corr * sx * sy # XY的协方差 = 相关系数 * X标准差 * Y标准差
    cov[:, :, :, 1, 0] = corr * sx * sy
    cov[:, :, :, 1, 1] = sy * sy

    return mu, cov

def sample_from_distr(V_pred, N):
    # input - distr: predicted future traj distribution at every time, [bs, 8, 12, 5]
    # return - samples: sampled traj points, [bs, 8, 12, N, 2]
    sx = torch.exp(V_pred[:, :, :, 2])  # sx
    sy = torch.exp(V_pred[:, :, :, 3])  # sy
    corr = torch.tanh(V_pred[:, :, :, 4])  # corr

    cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], V_pred.shape[2], 2, 2).to(V_pred.device)
    cov[:, :, :, 0, 0] = sx * sx
    cov[:, :, :, 0, 1] = corr * sx * sy
    cov[:, :, :, 1, 0] = corr * sx * sy
    cov[:, :, :, 1, 1] = sy * sy
    mean = V_pred[:, :, :, 0:2]

    mvnormal = torchdist.MultivariateNormal(mean, cov)
    all_samples = []
    for _ in range(N):
        sample_ = mvnormal.sample()
        all_samples.append(sample_)
    all_samples = torch.stack(all_samples)
    all_samples = all_samples.permute(1, 2, 3, 0, 4) # should be [bs, 8, 12, 20, 2]

    return all_samples

def mc_sample(mu, cov, n_sample):
    r_sample = torch.randn((n_sample,) + mu.shape, dtype=mu.dtype, device=mu.device)
    sample = mu + (torch.cholesky(cov) @ r_sample.unsqueeze(dim=-1)).squeeze(dim=-1)
    return sample


def qmc_sample(mu, cov, n_sample, rng):
    qr_seq = torch.stack([box_muller_transform(rng.draw(n_sample)) for _ in range(mu.size(0))], dim=1).unsqueeze(dim=2).type_as(mu) # rng.draw从sobel序列中抽点
    qr_seq = qr_seq.unsqueeze(2).repeat(1, 1, mu.size(1), 1, 1)
    sample = mu + (torch.cholesky(cov) @ qr_seq.unsqueeze(dim=-1)).squeeze(dim=-1)
    return sample


def purposive_sample(mu, cov, n_sample, loc_sample):
    # loc: [bs, 20, 2]
    loc_norm = box_muller_transform(loc_sample).permute(1, 0, 2).unsqueeze(2).expand((n_sample,) + mu.shape) # 得到均值为0，方差为1的高斯分布
    sample = mu + (torch.cholesky(cov) @ loc_norm.unsqueeze(dim=-1)).squeeze(dim=-1)
    # Y=KX+u  KK'=cov torch.cholesky(cov)=K 其中K为下三角矩阵, therefore Y=torch.cholesky(cov)*X + u
    return sample


def train(epoch, model, model_npsn, train_gen, criterion, optimizer, lr_scheduler, device, method):
    model.train() # Sets the module in training mode.
    num_batches = len(train_gen)
    epoch_per_Iter = 1.0 / num_batches # for lr_scheduler
    epoch_plus_CurIter = epoch
    count = 0
    total_goal_loss = 0
    total_distributed_loss = 0
    total_npsn_loss = 0
    total_multi_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    loader.set_description("train")
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_bbox_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            with_neighbor = model.module.with_neighbors if model.__class__.__name__ == 'DistributedDataParallel' else model.with_neighbors
            neighbors_data = data['neighbors_data'].to(device) if with_neighbor else None
            neighbors_idx_start = data['neighbors_idx_start'].to(device) if with_neighbor else None
            neighbors_idx_end = data['neighbors_idx_end'].to(device) if with_neighbor else None

            # target_bbox_st = data['target_y_st'].to(device)

            if method == 'npsn':
                # debug
                with torch.no_grad():
                    all_goal_traj, all_dec_distr, enc_hidden = model(input_traj, first_history_index[0])
                # all_goal_traj, all_dec_distr, enc_hidden = model(input_traj, first_history_index[0])
                mu, cov = generate_statistics_matrices(all_dec_distr)
                loc = model_npsn(enc_hidden, all_goal_traj[:, -1, :, :])
                loss_dist, loss_disc = model_npsn.get_loss(loc, mu[:, -1, ...], cov[:, -1, ...],
                                                           target_traj[:, -1, :, :])
                npsn_loss = loss_dist * 1.0 + loss_disc * 0.01
                # if npsn_loss.item() == float('inf'):
                #     loc = model_npsn(enc_hidden, all_goal_traj[:, -1, :, :])
                train_loss = npsn_loss
                goal_loss = torch.zeros([1]).to(device)
                distributed_loss = torch.zeros([1]).to(device)
            elif method in ['npsn_orig_ablation', 'npsn_ablation']:
                if first_history_index[0] != 0:
                    continue
                with torch.no_grad():
                    all_goal_traj, all_dec_distr, enc_hidden = model(input_traj, first_history_index[0])
                # all_goal_traj, all_dec_distr, enc_hidden = model(input_traj, first_history_index[0])
                mu, cov = generate_statistics_matrices(all_dec_distr)
                loc_list = []
                for i in range(batch_size):
                    ego_v = input_traj[i, :, 2:4] # 8,2 use the velocity as NPSN
                    nb_start_id = neighbors_idx_start[i]
                    nv_end_id = neighbors_idx_end[i]
                    neighbor_v = neighbors_data[nb_start_id:nv_end_id, :, 2:4] # (num_ped, 8, 2)
                    V_obs = torch.cat((ego_v.unsqueeze(0),neighbor_v), dim=0)
                    V_obs = V_obs.permute(0, 2, 1).unsqueeze(0) # (1, 8, 1+num_peds, 2)
                    if method == 'npsn_orig_ablation':
                        loc_ = model_npsn(V_obs) # should be (20, 2)
                    elif method == 'npsn_ablation':
                        loc_ = model_npsn(V_obs, enc_hidden=enc_hidden[i].unsqueeze(0), goal_traj=all_goal_traj[i, -1, :, :].unsqueeze(0))
                    loc_list.append(loc_)
                loc = torch.stack(loc_list, dim=0) # should be (bs, 20, 2)
                loss_dist, loss_disc = model_npsn.get_loss(loc, mu[:, -1, ...], cov[:, -1, ...],
                                                           target_traj[:, -1, :, :])
                npsn_loss = loss_dist * 1.0 + loss_disc * 0.01
                # if npsn_loss.item() == float('inf'):
                #     loc = model_npsn(enc_hidden, all_goal_traj[:, -1, :, :])
                train_loss = npsn_loss
                goal_loss = torch.zeros([1]).to(device)
                distributed_loss = torch.zeros([1]).to(device)
            elif method == 'multi_predictor':
                all_goal_traj, all_dec_traj, _ = model(input_traj, first_history_index[0])
                goal_loss = criterion(all_goal_traj[:, first_history_index[0]:, :, :],
                                      target_traj[:, first_history_index[0]:, :, :])
                multi_predict_loss = cvae_multi(all_dec_traj,target_traj, first_history_index[0])
                train_loss=  goal_loss + multi_predict_loss

                npsn_loss = torch.zeros([1]).to(device)
                distributed_loss = torch.zeros([1]).to(device)
            else:
                all_goal_traj, all_dec_distr, enc_hidden = model(input_traj, first_history_index[0])
                distributed_loss = distr_loss(all_dec_distr[:, first_history_index[0]:, :, :],
                                              target_traj[:, first_history_index[0]:, :, :])
                goal_loss = criterion(all_goal_traj[:, first_history_index[0]:, :, :],
                                      target_traj[:, first_history_index[0]:, :, :])
                npsn_loss = torch.zeros([1]).to(device)
                train_loss = goal_loss + distributed_loss


            # if distributed_loss.item() < 0:
            #     distributed_loss = distr_loss(all_dec_distr[:, first_history_index[0]:, :, :],
            #                                   target_traj[:, first_history_index[0]:, :, :])

            # train_loss = goal_loss + dec_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_distributed_loss += distributed_loss.item()* batch_size # batch的平均损失扩张为该batch的总损失
            total_npsn_loss += npsn_loss.item()* batch_size
            # total_multi_loss += multi_predict_loss.item() * batch_size

            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            epoch_plus_CurIter += epoch_per_Iter
            if lr_scheduler.__class__.__name__ == 'CosineAnnealingWarmRestarts':
                lr_scheduler.step(epoch_plus_CurIter)
        
    total_goal_loss /= count
    total_distributed_loss /= count
    total_npsn_loss /= count
    # total_multi_loss /= count

    return total_goal_loss, total_distributed_loss, total_npsn_loss, total_goal_loss + total_distributed_loss + total_npsn_loss
    # for multi predictor
    # return total_goal_loss, total_distributed_loss, total_multi_loss, total_goal_loss + total_distributed_loss + total_npsn_loss

def val(model, model_npsn, val_gen, criterion, device, method):
    total_goal_loss = 0
    total_distributed_loss = 0
    total_npsn_loss = 0
    count = 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    loader.set_description("val")
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_bbox_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            # target_bbox_st = data['target_y_st'].to(device)

            if method == 'npsn':
                with torch.no_grad():
                    all_goal_traj, all_dec_distr, enc_hidden = model(input_traj, first_history_index[0])
                mu, cov = generate_statistics_matrices(all_dec_distr)
                loc = model_npsn(enc_hidden, all_goal_traj[:, -1, :, :])
                loss_dist, loss_disc = model_npsn.get_loss(loc, mu[:, -1, ...], cov[:, -1, ...],
                                                           target_traj[:, -1, :, :])
                npsn_loss = loss_dist * 1.0 + loss_disc * 0.01
                # if npsn_loss.item() == float('inf'):
                #     loc = model_npsn(enc_hidden, all_goal_traj[:, -1, :, :])
            else:
                all_goal_traj, all_dec_distr, enc_hidden = model(input_traj, first_history_index[0])
                npsn_loss = torch.zeros([1]).to(device)

            distributed_loss = distr_loss(all_dec_distr[:, first_history_index[0]:, :, :],
                                          target_traj[:, first_history_index[0]:, :, :])
            goal_loss = criterion(all_goal_traj[:, first_history_index[0]:, :, :],
                                  target_traj[:, first_history_index[0]:, :, :])
            # if distributed_loss.item() < 0:
            #     distributed_loss = distr_loss(all_dec_distr[:, first_history_index[0]:, :, :],
            #                                   target_traj[:, first_history_index[0]:, :, :])

            # train_loss = goal_loss + dec_loss
            if method == 'npsn':
                val_loss = npsn_loss
            else:
                val_loss = goal_loss + dec_loss

            total_goal_loss += goal_loss.item() * batch_size
            total_distributed_loss += distributed_loss.item() * batch_size  # batch的平均损失扩张为该batch的总损失
            total_npsn_loss += npsn_loss.item() * batch_size

        total_goal_loss /= count
        total_distributed_loss /= count
        total_npsn_loss /= count

        return total_goal_loss, total_distributed_loss, total_npsn_loss, total_goal_loss + total_distributed_loss + total_npsn_loss

def test(model, model_npsn, test_gen, criterion, device, method):
    total_goal_loss = 0
    total_dec_loss = 0
    total_npsn_loss = 0
    ADE_08 = 0
    ADE_12 = 0 
    FDE_08 = 0 
    FDE_12 = 0 
    count = 0
    model.eval()
    if method == 'qmc':
        sobol_generator = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=0)
    loader = tqdm(test_gen, total=len(test_gen))
    loader.set_description("test")
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):

            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size

            input_traj = data['input_x'].to(device)
            input_bbox_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            with_neighbor = model.module.with_neighbors if model.__class__.__name__ == 'DistributedDataParallel' else model.with_neighbors
            neighbors_data = data['neighbors_data'].to(device) if with_neighbor else None
            neighbors_idx_start = data['neighbors_idx_start'].to(device) if with_neighbor else None
            neighbors_idx_end = data['neighbors_idx_end'].to(device) if with_neighbor else None

            # target_bbox_st = data['target_y_st'].to(device)

            if method == 'multi_predictor':
                all_goal_traj, all_dec_traj, _ = model(input_traj, first_history_index[0])
                distributed_loss = torch.zeros([1]).to(device)

            else:
                all_goal_traj, all_dec_distr, enc_hidden = model(input_traj, first_history_index[0])
                distributed_loss = distr_loss(all_dec_distr[:, first_history_index[0]:, :, :],
                                              target_traj[:, first_history_index[0]:, :, :])
                mu, cov = generate_statistics_matrices(all_dec_distr)
                N = 20

            goal_loss = criterion(all_goal_traj[:, first_history_index[0]:, :, :],
                                  target_traj[:, first_history_index[0]:, :, :])
            npsn_loss = torch.zeros([1])
                # all_dec_traj = sample_from_distr(all_dec_distr[:,first_history_index[0]:,:,:], N=20) # [bs, obs_len, 12, 20, 2]

            if method == 'mc':
                all_dec_traj = mc_sample(mu, cov, n_sample=N)
                all_dec_traj = all_dec_traj.permute(1, 2, 3, 0, 4)
            elif method == 'qmc':
                all_dec_traj = qmc_sample(mu, cov, n_sample=N, rng=sobol_generator)
                all_dec_traj = all_dec_traj.permute(1, 2, 3, 0, 4)
            elif method == 'npsn':
                loc = model_npsn(enc_hidden, all_goal_traj[:, -1, :, :])
                all_dec_traj = purposive_sample(mu[:, -1, ...], cov[:, -1, ...], n_sample=N, loc_sample=loc)  # npsn没用到qmc，靠loc抽取mu, cov
                loss_dist, loss_disc = model_npsn.get_loss(loc, mu[:, -1, ...], cov[:, -1, ...],
                                                           target_traj[:, -1, :, :])
                npsn_loss = loss_dist * 1.0 + loss_disc * 0.01
                all_dec_traj = all_dec_traj.permute(1, 2, 0, 3)
            elif method in ['npsn_orig_ablation', 'npsn_ablation']:
                if first_history_index[0] != 0:
                    continue
                loc_list = []
                for i in range(batch_size):
                    ego_v = input_traj[i, :, 2:4]  # 8,2 use the velocity as NPSN
                    nb_start_id = neighbors_idx_start[i]
                    nv_end_id = neighbors_idx_end[i]
                    neighbor_v = neighbors_data[nb_start_id:nv_end_id, :, 2:4]  # (num_ped, 8, 2)
                    V_obs = torch.cat((ego_v.unsqueeze(0), neighbor_v), dim=0)
                    V_obs = V_obs.permute(0, 2, 1).unsqueeze(0) # (1, 8, 1+num_peds, 2)
                    if method == 'npsn_orig_ablation':
                        loc_ = model_npsn(V_obs)  # should be (20, 2)
                    elif method == 'npsn_ablation':
                        loc_ = model_npsn(V_obs, enc_hidden=enc_hidden[i].unsqueeze(0), goal_traj=all_goal_traj[i, -1, :, :].unsqueeze(0))
                    loc_list.append(loc_)
                loc = torch.stack(loc_list, dim=0)  # should be (bs, 20, 2)
                all_dec_traj = purposive_sample(mu[:, -1, ...], cov[:, -1, ...], n_sample=N,
                                                loc_sample=loc)
                loss_dist, loss_disc = model_npsn.get_loss(loc, mu[:, -1, ...], cov[:, -1, ...],
                                                           target_traj[:, -1, :, :])
                npsn_loss = loss_dist * 1.0 + loss_disc * 0.01
                all_dec_traj = all_dec_traj.permute(1, 2, 0, 3)

            # dec_loss = multipred_loss(all_dec_traj, target_traj[:, first_history_index[0]:, :, :])

            total_goal_loss += goal_loss.item()* batch_size
            total_dec_loss += distributed_loss.item()* batch_size
            total_npsn_loss += npsn_loss.item()* batch_size

            all_dec_traj_np = all_dec_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()

            # Decoder
            if method in ['npsn', 'npsn_orig_ablation', 'npsn_ablation']:
                batch_results = \
                    eval_ethucy_cvae(input_traj_np, target_traj_np[:, -1, :, :], all_dec_traj_np)

            else:
                batch_results =\
                eval_ethucy_cvae(input_traj_np, target_traj_np[:,-1,:,:], all_dec_traj_np[:,-1,:,:,:])

            ADE_08 += batch_results['ADE_08']
            ADE_12 += batch_results['ADE_12']
            FDE_08 += batch_results['FDE_08']
            FDE_12 += batch_results['FDE_12']
            
    ADE_08 /= count
    ADE_12 /= count
    FDE_08 /= count
    FDE_12 /= count
    
    total_dec_loss /= count
    total_npsn_loss /= count
    test_loss = total_goal_loss/count + total_dec_loss/count + total_npsn_loss / count

    print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))
    return total_dec_loss, total_npsn_loss, ADE_08, FDE_08, ADE_12, FDE_12

def test_output_traj(model, model_npsn, test_gen, device, method):
    ADE_08 = 0
    ADE_12 = 0
    FDE_08 = 0
    FDE_12 = 0
    count = 0
    all_obs = []
    all_preds = []
    all_gt = []
    model.eval()
    model_npsn.eval()
    if method == 'qmc':
        sobol_generator = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=0)
    loader = tqdm(test_gen, total=len(test_gen))
    loader.set_description("test")
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):

            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size

            input_traj = data['input_x'].to(device)
            input_bbox_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            # target_bbox_st = data['target_y_st'].to(device)

            all_goal_traj, all_dec_distr, enc_hidden = model(input_traj, first_history_index[0])

            npsn_loss = torch.zeros([1])
            # all_dec_traj = sample_from_distr(all_dec_distr[:,first_history_index[0]:,:,:], N=20) # [bs, obs_len, 12, 20, 2]

            mu, cov = generate_statistics_matrices(all_dec_distr)
            N = 20
            if method == 'mc':
                all_dec_traj = mc_sample(mu, cov, n_sample=N)
                all_dec_traj = all_dec_traj.permute(1, 2, 3, 0, 4)
            elif method == 'qmc':
                all_dec_traj = qmc_sample(mu, cov, n_sample=N, rng=sobol_generator)
                all_dec_traj = all_dec_traj.permute(1, 2, 3, 0, 4)
            elif method == 'npsn':
                loc = model_npsn(enc_hidden, all_goal_traj[:, -1, :, :])
                all_dec_traj = purposive_sample(mu[:, -1, ...], cov[:, -1, ...], n_sample=N, loc_sample=loc)
                loss_dist, loss_disc = model_npsn.get_loss(loc, mu[:, -1, ...], cov[:, -1, ...],
                                                           target_traj[:, -1, :, :])
                npsn_loss = loss_dist * 1.0 + loss_disc * 0.01
                all_dec_traj = all_dec_traj.permute(1, 2, 0, 3)
            else:
                raise NotImplementedError

            all_dec_traj_np = all_dec_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()

            # Decoder
            if method == 'npsn':
                batch_results, input_selected, gt_selected, pred_selected = \
                    eval_ethucy_cvae_output_mintraj(input_traj_np, target_traj_np[:, -1, :, :], all_dec_traj_np)

                input_selected = torch.tensor(input_selected)
                gt_selected = torch.tensor(gt_selected)
                pred_selected = torch.tensor(pred_selected)
            else:
                batch_results = \
                    eval_ethucy_cvae(input_traj_np, target_traj_np[:,-1,:,:], all_dec_traj_np[:,-1,:,:,:])

            ADE_08 += batch_results['ADE_08']
            ADE_12 += batch_results['ADE_12']
            FDE_08 += batch_results['FDE_08']
            FDE_12 += batch_results['FDE_12']

            # output all
            all_obs.append(input_traj[:, :, :2]) # ignore fist_index
            all_gt.append(target_traj[:, -1, ...]+input_traj[:, -1, :2].unsqueeze(1))
            all_preds.append(all_dec_traj+input_traj[:, -1, :2].unsqueeze(1).unsqueeze(1))

            # output low error
            # all_obs.append(input_selected[:, :, :2]) # ignore fist_index
            # all_gt.append(gt_selected+input_selected[:, -1, :2].unsqueeze(1))
            # all_preds.append(pred_selected+input_selected[:, -1, :2].unsqueeze(1).unsqueeze(1))

    ADE_08 /= count
    ADE_12 /= count
    FDE_08 /= count
    FDE_12 /= count
    print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))

    trajectory_results = {}
    trajectory_results['obs'] = torch.cat(all_obs, dim=0).detach().cpu()
    trajectory_results['gt'] = torch.cat(all_gt, dim=0).detach().cpu()
    trajectory_results['preds'] = torch.cat(all_preds, dim=0).detach().cpu()
    return trajectory_results
