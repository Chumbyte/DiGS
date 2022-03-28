# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import argparse
import torch
import os
import numpy as np

def add_args(parser):
    parser.add_argument('--gpu_idx', type=int, default=0, help='set < 0 to use CPU')
    parser.add_argument('--threads', type=int, default=4, help='num of threads for data loader')
    parser.add_argument('--logdir', type=str, default='./log/debug', help='log directory')
    parser.add_argument('--expname', type=str, default='shapespaceDIBS')
    parser.add_argument('--is_continue', default=False, action="store_true", help='continue')
    parser.add_argument('--timestamp', default='latest', type=str)
    parser.add_argument('--checkpoint', default='latest', type=str)

    parser.add_argument('--model_name', type=str, default='DiBS', help='trained model name')
    parser.add_argument('--seed', type=int, default=3627473, help='random seed')
    parser.add_argument('--dataset_path', type=str, default='~/PhD/DIBS/dfaust_processed', help='path to dataset folder')
    # parser.add_argument('--dataset_path', type=str, default='~/PhD/DIBS/dfaust_processed2', help='path to dataset folder')
    parser.add_argument('--gt_path', type=str, default='~/PhD/DIBS/DFaust/scripts', help='path to gt folder')
    parser.add_argument('--scan_path', type=str, default='~/PhD/DIBS/DFaust/scans', help='path to scan folder')
    parser.add_argument('--split_path', type=str, default='~/PhD/DIBS/DFaust/splits/dfaust/train_all.json', help='path to split file')
    # parser.add_argument('--file_name', type=str, default='lord_quas.ply',
    #                     help='name of file to reconstruc (withing the dataset path')
    # parser.add_argument('--n_samples', type=int, default=1024,
    #                     help='number of samples in the generated train and test set')
    # parser.add_argument('--refine_epoch', type=int, default=0, help='refine model from this epoch, '
    #                                                                 '0 for training from scratch')
    parser.add_argument('--model_dirpath', type=str,
                        default='/mnt/3.5TB_WD/PycharmProjects/DiBS/models',
                        help='path to model directory for backup')
    parser.add_argument('--parallel', type=int, default=False, help='use data parallel')

    # training parameters
    parser.add_argument('--num_epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
    parser.add_argument('--grad_clip_norm', type=float, default=10.0, help='Value to clip gradients to')
    parser.add_argument('--batch_size', type=int, default=8, help='number of separate point clouds in a minibatch')
    parser.add_argument('--effective_batch_size', type=int, default=16, help='Effective batch size from grad accumulation')
    parser.add_argument('--n_points', type=int, default=8000, help='number of points in each point cloud')
    parser.add_argument('--delta', type=float, default=0.25, help='clamping value')

    # Network architecture and loss
    parser.add_argument('--decoder_hidden_dim', type=int, default=256, help='length of decoder hidden dim')
    parser.add_argument('--encoder_hidden_dim', type=int, default=128, help='length of encoder hidden dim')
    parser.add_argument('--decoder_n_hidden_layers', type=int, default=8, help='number of decoder hidden layers')
    parser.add_argument('--nl', type=str, default='sine', help='type of non linearity sine | relu')
    # parser.add_argument('--nl', type=str, default='relu', help='type of non linearity sine | relu')
    parser.add_argument('--latent_size', type=int, default=256, help='number of elements in the latent vector')
    parser.add_argument('--sphere_init_params', nargs='+', type=float, default=[1.6, 1.0],
                    help='radius and scaling')

    parser.add_argument('--normalize_normal_loss', type=int, default=False, help='normal loss normalization flag')
    parser.add_argument('--unsigned_n', type=int, default=True, help='flag for unsigned normal loss')
    parser.add_argument('--unsigned_d', type=int, default=False, help='flag for unsigned distance loss')

    parser.add_argument('--encoder_type', type=str, default='autodecoder', help='type of encoder None | pointnet | autodecoder')
    parser.add_argument('--export_vis', type=int, default=True, help='export levelset visualization while training')

    # parser.add_argument('--loss_type', type=str, default='siren_wo_n_w_div', help='loss type to use: siren | siren_wo_n | igr')
    # parser.add_argument('--loss_type', type=str, default='igr', help='loss type to use: siren | siren_wo_n | igr')
    # parser.add_argument('--loss_type', type=str, default='siren_wo_n', help='loss type to use: siren | siren_wo_n | igr')
    parser.add_argument('--loss_type', type=str, default='siren_wo_n_w_div', help='loss type to use: siren | siren_wo_n | igr | siren_w_div | siren_wo_n_w_div')
    parser.add_argument('--n_loss_type', type=str, default='cos', help='type of normal los cos | ')
    parser.add_argument('--inter_loss_type', type=str, default='exp', help='type of inter los exp | unsigned_diff | signed_diff')
    parser.add_argument('--sampler_prob', type=str, default='none',
                        help='type of sampler probability for non manifold points on the grid none | div | curl')
    parser.add_argument('--div_decay_params', nargs='+', type=float, default=[0, 0.5, 0.75],
                        help='epoch number to evaluate')
    parser.add_argument('--div_type', type=str, default='l1', help='divergence term norm l1 | l2')
    parser.add_argument('--grid_res', type=int, default=128, help='uniform grid resolution')
    parser.add_argument('--div_clamp', type=float, default=50, help='divergence clamping value')
    parser.add_argument('--div_decay', type=str, default='linear',
                        help='divergence term importance decay none | step | linear')
    parser.add_argument('--nonmnfld_sample_type', type=str, default='uniform',
                        help='how to sample points off the manifold - grid | gaussian | combined')
    parser.add_argument('--init_type', type=str, default='mfgi',
                        help='initialization type siren | geometric_sine | geometric_relu | mfgi')
    parser.add_argument('--track_weights', type=bool, default=False,
                        help='flag to track the weights or not (increases train time')
    parser.add_argument('--loss_weights', nargs='+', type=float, default=[3e3, 1e2, 1e2, 5e1, 1e2, 1e0],
                        help='loss terms weights sdf | inter | normal | eikonal | div')
    parser.add_argument('--test_loss_weights', nargs='+', type=float, default=[3e3, 1e2, 1e2, 5e1, 0e2, 1e0],
                        help='loss terms weights sdf | inter | normal | eikonal | div')
    parser.add_argument('--num_latent_iters', type=int, default=100, help='iters to optimise latent code for')
    parser.add_argument('--test_split_path', type=str, default='~/PhD/DIBS/DFaust/splits/dfaust/test_all.json', help='path to split file')
    parser.add_argument('--test_res', type=int, default=100, help='resolution for testing')
    return parser


def get_parser():
    # command line args
    parser = argparse.ArgumentParser(description='Local implicit functions experiment')
    parser = add_args(parser)
    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    if 'igr' in args.loss_type:
        args.nl = 'softplus'
        args.decoder_hidden_dim = 512
        args.decoder_n_hidden_layers = 8
    if args.loss_type == 'siren_wo_n_w_div_w_dist':
        args.requires_dist = False
    else:
        args.requires_dist = False
    return args

# def get_test_args():
#     parser = argparse.ArgumentParser(description='Local implicit functions test experiment')
#     parser.add_argument('--gpu_idx', type=int, default=1, help='set < 0 to use CPU')
#     parser.add_argument('--logdir', type=str,
#                         default='/mnt/IronWolf/logs/DiBS/3D/grid_sampling_256/SIREN/',
#                         help='log directory')
#     parser.add_argument('--file_name', type=str, default='daratech.ply', help='trained model name')
#     parser.add_argument('--model_name', type=str, default='DiBS', help='trained model name')
#     parser.add_argument('--n_points', type=int, default=0, help='number of points in each point cloud, '
#                                                                   'if 0 use training options')
#     parser.add_argument('--batch_size', type=int, default=0, help='number of samples in a minibatch, if 0 use training')
#     # parser.add_argument('--epoch_n', nargs='+', default=np.arange(0, 3000, 20).tolist(), help='epoch number to evaluate')
#     parser.add_argument('--epoch_n', type=int, default=1, help='epoch number to evaluate')
#     parser.add_argument('--grid_res', type=int, default=512, help='grid resolution for reconstruction')
#     parser.add_argument('--export_mesh', type=bool, default=True, help='indicator to export mesh as ply file')
#     parser.add_argument('--dataset_path', type=str,
#                         default='/mnt/IronWolf/PycharmProjects/DiBS/Baselines/surface_reconstruction/scans',
#                         help='path to dataset folder')
#     test_opt = parser.parse_args()

#     test_opt.logdir = os.path.join(test_opt.logdir, test_opt.file_name.split('.')[0])
#     param_filename = os.path.join(test_opt.logdir, 'trained_models/', test_opt.model_name + '_params.pth')
#     train_opt = torch.load(param_filename)

#     test_opt.nl, test_opt.latent_size, test_opt.encoder_type, test_opt.n_samples, test_opt.seed, \
#         test_opt.decoder_hidden_dim, test_opt.encoder_hidden_dim, test_opt.decoder_n_hidden_layers = train_opt.nl,\
#         train_opt.latent_size, train_opt.encoder_type, train_opt.n_samples, \
#         train_opt.seed, train_opt.decoder_hidden_dim, train_opt.encoder_hidden_dim, train_opt.decoder_n_hidden_layers
#     test_opt.n_point_total = train_opt.n_points

#     if test_opt.n_points == 0:
#         test_opt.n_points = train_opt.n_points
#     if test_opt.batch_size == 0:
#         test_opt.batch_size = train_opt.batch_size
#     if "parallel" in train_opt:
#         test_opt.parallel = train_opt.parallel
#     else:
#         test_opt.parallel = False
#     return test_opt
