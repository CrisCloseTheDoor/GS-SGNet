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
sys.path.append(".\\nuscenes\\python-sdk")
sys.path.append(os.path.abspath('./'))
from configs.ethucy import parse_sgnet_args as parse_args
import lib.utils as utl
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.ethucy_train_utils import train, val, test


def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', args.dataset, model_name, str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    utl.set_seed(int(args.seed))
    model = build_model(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 4, T_mult=2, eta_min=1e-5)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=5,
    #                                                        min_lr=1e-10, verbose=1)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    model = model.to(device)
    if osp.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        args.start_epoch += checkpoint['epoch']
        del checkpoint


    criterion = rmse_loss().to(device)

    train_gen = utl.build_data_loader(args, 'train', batch_size=1)
    val_gen = utl.build_data_loader(args, 'val', batch_size=1)
    test_gen = utl.build_data_loader(args, 'test', batch_size=1)
    print("Number of train samples:", train_gen.__len__())
    print("Number of validation samples:", val_gen.__len__())
    print("Number of test samples:", test_gen.__len__())
    # train
    min_loss = 1e6
    min_ADE_08 = 10e5
    min_FDE_08 = 10e5
    min_ADE_12 = 10e5
    min_FDE_12 = 10e5
    best_model = None
    best_model_metric = None

    for epoch in range(args.start_epoch, args.epochs+args.start_epoch):

        train_goal_loss, train_dec_loss, total_train_loss = train(epoch, model, train_gen, criterion, optimizer, lr_scheduler, device)

        print('Train Epoch: {} \t Goal loss: {:.4f}\t Decoder loss: {:.4f}\t Total: {:.4f}'.format(
                epoch, train_goal_loss, train_dec_loss, total_train_loss))

        # val
        val_loss = val(model, val_gen, criterion, device)
        lr_scheduler.step(val_loss)

        # test
        test_loss, ADE_08, FDE_08, ADE_12, FDE_12 = test(model, test_gen, device)

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

            saved_model_metric_name = 'metric_epoch_' + str(format(epoch,'03')) + '_ADE_%.4f'%ADE_12 + '.pth'


            print("Saving checkpoints: " + saved_model_metric_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            save_dict = {   'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}
            torch.save(save_dict, os.path.join(save_dir, saved_model_metric_name))


            best_model_metric = os.path.join(save_dir, saved_model_metric_name)

if __name__ == '__main__':
    main(parse_args())
