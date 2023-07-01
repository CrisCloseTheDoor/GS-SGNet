import argparse

__all__ = ['parse_base_args']

def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_method', default='mc', type=str, choices=['mc', 'qmc', 'npsn',
                                                                            'multi_predictor'])
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--checkpoint_npsn', default='', type=str)
    parser.add_argument('--modified_preprocess', action='store_true')
    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--pretrained_dir', type=str, default=None)
    parser.add_argument('--init_model', action='store_true', help='init model')
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--cpu', action='store_true', help='force to use cpu')
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--shuffle', default=True, type=bool)
    return parser
