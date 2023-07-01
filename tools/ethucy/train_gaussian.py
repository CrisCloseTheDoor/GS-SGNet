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

import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# from torch.utils.tensorboard import SummaryWriter
sys.path.append(".\\nuscenes\\python-sdk")
sys.path.append(os.path.abspath('./'))
from configs.ethucy import parse_sgnet_args as parse_args
import lib.utils as utl
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.ethucy_train_utils_Gaussian import train, val, test
from tensorboardX import SummaryWriter


def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', args.dataset, model_name, str(args.seed)+'_'+str(args.sample_method))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    utl.set_seed(int(args.seed))
    model = build_model(args)
    model_npsn = None

    if osp.isdir(args.pretrained_dir):
        load_path = os.path.join(args.pretrained_dir, args.dataset.lower()+'.pth')
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        to_be_train = ['dec_hidden2distr'] if args.sample_method not in ['npsn', 'npsn_orig_ablation', 'npsn_ablation'] else []
        for k, v in model.named_parameters():
            module_name = k.split('.')[0]
            if module_name not in to_be_train:
                v.requires_grad = False  # 固定参数
        del checkpoint
        print("load pretrained model from", load_path)
    model = model.to(device)

    if args.sample_method in ['npsn']:
        from lib.models.npsn.npsn_model import NPSN
        model_npsn = NPSN(hidden_dim=args.hidden_size, dec_steps=args.dec_steps,
                          s=2, n=20, dropout=args.dropout, init=args.init_model).to(device)

        model_npsn = model_npsn.to(device)
        optimizer = optim.AdamW(model_npsn.parameters(), lr=args.lr)
        # optimizer = optim.AdamW([{'params': model.parameters()},
        #                          {'params': model_npsn.parameters()}
        #                             ], lr=args.lr)
    else:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4, T_mult=2, eta_min=args.lr/500.0)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
    #                                                        min_lr=1e-10, verbose=1)

    if args.continue_training:
        if args.sample_method != 'npsn' and osp.isfile(args.checkpoint):
            load_path = args.checkpoint
            checkpoint = torch.load(load_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            to_be_train = ['dec_hidden2distr']
            for k, v in model.named_parameters():
                module_name = k.split('.')[0]
                if module_name not in to_be_train:
                    v.requires_grad = False  # 固定参数
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            args.start_epoch = checkpoint['epoch'] + 1
            print("Load Last ckpt From", args.checkpoint)
            del checkpoint
        elif args.sample_method == 'npsn' and osp.isfile(args.checkpoint_npsn):
            checkpoint_npsn = torch.load(args.checkpoint_npsn, map_location=device)
            model_npsn.load_state_dict(checkpoint_npsn['model_state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint_npsn['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint_npsn['scheduler_state_dict'])
            args.start_epoch = checkpoint_npsn['epoch'] + 1
            print("Load Last ckpt From", args.checkpoint_npsn)
            del checkpoint_npsn

    # if osp.isdir(args.pretrained_dir):
    #     load_path = os.path.join(args.pretrained_dir, args.dataset.lower()+'.pth')
    #     checkpoint = torch.load(load_path, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    #     to_be_train = ['dec_hidden2distr'] if args.sample_method != 'npsn' else []
    #     for k, v in model.named_parameters():
    #         module_name = k.split('.')[0]
    #         if module_name not in to_be_train:
    #             v.requires_grad = False  # 固定参数
    #     del checkpoint
    #
    # if osp.isfile(args.checkpoint):
    #     checkpoint = torch.load(args.checkpoint, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     args.start_epoch += checkpoint['epoch']
    #     del checkpoint


    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train', batch_size = 1)
    val_gen = utl.build_data_loader(args, 'val', batch_size = 1)
    test_gen = utl.build_data_loader(args, 'test', batch_size = 1)
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())
    print("-----  using method: {} -----".format(args.sample_method))

    log_path = save_dir
    log = SummaryWriter(log_path)
    # train
    min_loss = 1e6
    min_ADE_08 = 10e5
    min_FDE_08 = 10e5
    min_ADE_12 = 10e5
    min_FDE_12 = 10e5
    best_model = None
    best_model_metric = None
    lr_list = []
    ADE12_list = []
    FDE12_list = []

    for epoch in range(args.start_epoch, args.epochs + 1):

        train_goal_loss, total_distributed_loss, npsn_loss, total_train_loss = \
            train(epoch, model, model_npsn, train_gen, criterion, optimizer, lr_scheduler, device, args.sample_method)

        print('Train Epoch: {} \t Goal loss: {:.4f}\t Distr loss: {:.4f}\t NPSN loss: {:.4f}\t Total: {:.4f}'.format(
                epoch, train_goal_loss, total_distributed_loss, npsn_loss, total_train_loss))

        # log.add_scalar("goal_loss/train", train_goal_loss, epoch)
        log.add_scalar("distributed_loss/train", total_distributed_loss, epoch)
        log.add_scalar("npsn_loss/train", npsn_loss, epoch)
        # log.add_scalar("total_loss/train", total_train_loss, epoch)

        # val
        if lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
            val_goal_loss, val_distributed_loss, npsn_loss, total_val_loss = \
                val(model, model_npsn, val_gen, criterion, device, args.sample_method)
            lr_scheduler.step(total_val_loss)
        lr = lr_scheduler.optimizer.param_groups[0]['lr']
        # log.add_scalar("learning rate", lr, epoch)
        # log.add_scalar("goal_loss/val", val_goal_loss, epoch)
        # log.add_scalar("distributed_loss/val", val_distributed_loss, epoch)
        # log.add_scalar("npsn_loss/val", npsn_loss, epoch)
        # log.add_scalar("total_loss/val", total_val_loss, epoch)


        # test
        dec_loss, npsn_loss, ADE_08, FDE_08, ADE_12, FDE_12 = test(model, model_npsn, test_gen, criterion, device, args.sample_method)
        lr_list.append(lr)
        ADE12_list.append(float(ADE_12))
        FDE12_list.append(float(FDE_12))
        log.add_scalar("learning_rate", lr, epoch)
        log.add_scalar("ADE_12/test", ADE_12, epoch)
        log.add_scalar("FDE_12/test", FDE_12, epoch)
        # log.add_scalar("distributed_loss/test", dec_loss, epoch)
        log.add_scalar("npsn_loss/test", npsn_loss, epoch)

        # save checkpoints if loss decreases
        # if test_loss < min_loss:
        #     try:
        #         os.remove(best_model)
        #     except:
        #         pass

        #     min_loss = test_loss
        #     saved_model_name = 'epoch_' + str(format(epoch,'03')) + '_loss_%.4f'%min_loss + '.pth'

        #     print("Saving checkpoints: " + saved_model_name )
        #     if not os.path.isdir(save_dir):
        #         os.mkdir(save_dir)

        #     save_dict = {   'epoch': epoch,
        #                     'model_state_dict': model.state_dict(),
        #                     'optimizer_state_dict': optimizer.state_dict()}
        #     torch.save(save_dict, os.path.join(save_dir, saved_model_name))
        #     best_model = os.path.join(save_dir, saved_model_name)



        if ADE_12 < min_ADE_12:
            try:
                os.remove(best_model_metric)
            except:
                pass
            min_ADE_08 = ADE_08
            min_FDE_08 = FDE_08
            min_ADE_12 = ADE_12
            min_FDE_12 = FDE_12
            with open(os.path.join(save_dir, 'metric.txt'),"w") as f:
                f.write("ADE_08: %4f; FDE_08: %4f; ADE_12: %4f; FDE_12: %4f;" % (ADE_08, FDE_08,ADE_12,FDE_12))

            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            if args.sample_method == 'npsn':
                save_dict = {'epoch': epoch,
                             'model_state_dict': model_npsn.state_dict(),
                             'optimizer_state_dict': optimizer.state_dict(),
                             'scheduler_state_dict': lr_scheduler.state_dict()}
                saved_model_metric_name = 'NPSN_epoch_' + str(format(epoch, '03')) + '_ADE_%.4f' % ADE_12 + '.pth'
            else:
                save_dict = {   'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_scheduler.state_dict()}
                saved_model_metric_name = 'metric_epoch_' + str(format(epoch, '03')) + '_ADE_%.4f' % ADE_12 + '.pth'
            print("Saving checkpoints: " + saved_model_metric_name)
            torch.save(save_dict, os.path.join(save_dir, saved_model_metric_name))

            best_model_metric = os.path.join(save_dir, saved_model_metric_name)


if __name__ == '__main__':
    args = parse_args()
    main(args)
