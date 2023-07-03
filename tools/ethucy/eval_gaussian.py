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
sys.path.append(os.path.abspath('./'))
import lib.utils as utl
from configs.ethucy import parse_sgnet_args as parse_args
from lib.models import build_model
import pickle
from lib.utils.ethucy_train_utils_Gaussian import test_output_traj
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# 设置字体
plt.rcParams['font.sans-serif']=['Times New Roman'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 #有中文出现的情况，需要u'内容'

def draw(i, trajectory_results, axs, title=''):

    ego_his = trajectory_results['obs'][i, :, :2] # 8, 2
    pred_traj = trajectory_results['preds'][i, ...] # 12, 20, 2
    N = pred_traj.shape[1]
    pred_traj = torch.cat((ego_his[-1, :].unsqueeze(0).unsqueeze(0).repeat(1,N,1),pred_traj), dim=0)
    gt_traj = trajectory_results['gt'][i, ...] # 12, 2

    # plt.scatter((ego_his[-1,0] + gt_traj[0,0])/2, (ego_his[-1,1] + gt_traj[0, 1])/2, color='blue',
    #             marker='*', linewidth=2, label='当前点')
    for mod in range(pred_traj.shape[1]):
        axs.plot(pred_traj[:, mod, 0], pred_traj[:, mod, 1],
                 color='green', linestyle='-', linewidth=3, label='Prediction')
    axs.plot(ego_his[:, 0], ego_his[:, 1], color='gold', linestyle='--', label='History', linewidth=3)
    axs.plot(gt_traj[:, 0], gt_traj[:, 1], color='red', label='GroundTruth', linestyle='--', linewidth=3)

    handles, labels = axs.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axs.legend(by_label.values(), by_label.keys(), fontsize=12)
    axs.set_title(title, fontsize=14)

    x_major_locator = MultipleLocator(0.4)
    y_major_locator = MultipleLocator(0.5)
    # axs.xaxis.set_major_locator(x_major_locator) # 是否生成密集刻度
    # axs.yaxis.set_major_locator(y_major_locator)
    axs.set_xlim(-6, 5)
    axs.set_ylim(-1.5, 1)

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', args.dataset, model_name, str(args.seed)+'_'+str(args.sample_method))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model = build_model(args)
    model_npsn = None

    if osp.isfile(args.checkpoint): # load the SGNet backbone
        load_path = args.checkpoint
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        del checkpoint
        print("load from", load_path)
    model = model.to(device)

    if args.sample_method == 'npsn':
        assert osp.isfile(args.checkpoint_npsn), "NPSN checkpoint path is not valid !"
        from lib.models.npsn.npsn_model import NPSN
        model_npsn = NPSN(hidden_dim=args.hidden_size, dec_steps=args.dec_steps, s=2, n=20, dropout=args.dropout).to(
            device)
        checkpoint = torch.load(args.checkpoint_npsn, map_location=device)
        model_npsn.load_state_dict(checkpoint['model_state_dict'], strict=True)
        del checkpoint
        print("load NPSN from", args.checkpoint_npsn)
        model_npsn = model_npsn.to(device)

    test_gen = utl.build_data_loader(args, 'test', batch_size = 1)
    print("Number of test samples:", test_gen.__len__())
    print("-----  using method: {} -----".format(args.sample_method))

    # test
    trajectory_results = test_output_traj(model, model_npsn, test_gen, device, args.sample_method)
    return trajectory_results


if __name__ == '__main__':
    args = parse_args()
    # # Opt 0: main directly
    # main(args)
    # exit()

    # # Opt 1: generate results by model
    # trajectory_results = main(args)
    # with open(f"D:\\方向：交通轨迹\\★代码\\SGNet和trajectron++\\results_trajectory\\{args.dataset}\\SGNet-NPSN-wrd-{args.dataset.lower}.pickle","wb") as f:
    #     pickle.dump(trajectory_results, f)

    # Opt 2: draw comparision figures
    dataset = args.dataset.upper()
    # for sample_idx in range(0, 30):
    sample_idx = 25
    print(f"idx={sample_idx}")
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
    with open(f"D:\方向：交通轨迹\★代码\SGNet和trajectron++\\results_trajectory\\{dataset}\\SGNet-NPSN-wrd-{dataset.lower()}.pickle","rb") as f:
        trajectory_results = pickle.load(f)
    draw(sample_idx, trajectory_results, axs.flat[0], title="OURS")
    # compared model
    with open(f"D:\方向：交通轨迹\★代码\SGNet和trajectron++\\results_trajectory\\{dataset}\\BitrapGMM-wrd-{dataset.lower()}.pickle","rb") as f:
        trajectory_results = pickle.load(f)
    draw(sample_idx, trajectory_results,axs.flat[1], title='Bitrap-GMM')
    with open(f"D:\方向：交通轨迹\★代码\SGNet和trajectron++\\results_trajectory\\{dataset}\\SGNet-cvae-wrd-{dataset.lower()}.pickle", "rb") as f:
        trajectory_results = pickle.load(f)
        draw(sample_idx, trajectory_results, axs.flat[2], title='SGNet')
    plt.show()
