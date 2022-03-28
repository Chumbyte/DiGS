# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shapespace.dfaust_dataset import DFaustDataSet
import torch
import utils.visualizations as vis
import numpy as np
import models.DiGS as DiGS
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import utils.utils as utils
import shapespace.shapespace_dfaust_args as shapespace_dfaust_args
from shapespace.shapespace_utils import logging_print, mkdir_ifnotexists
from torch.utils.data._utils.collate import default_collate
from tensorboardX import SummaryWriter

# python train_shapespace.py --batch_size 4 --effective_batch_size 16 --n_points 8000


# get training parameters
args = shapespace_dfaust_args.get_args()
gpu_idx, nl, n_points, batch_size, effective_batch_size, latent_size, num_epochs, logdir, \
model_name, n_loss_type, normalize_normal_loss, unsigned_n, unsigned_d, loss_type, seed, encoder_type,\
    model_dirpath, inter_loss_type =\
    args.gpu_idx, args.nl, args.n_points, args.batch_size, args.effective_batch_size, args.latent_size, \
    args.num_epochs, args.logdir, args.model_name, args.n_loss_type, \
    args.normalize_normal_loss, args.unsigned_n, args.unsigned_d, args.loss_type, args.seed, args.encoder_type, \
    args.model_dirpath, args.inter_loss_type

os.makedirs(logdir, exist_ok=True)
mkdir_ifnotexists(os.path.join(logdir, "trained_models"))
# mkdir_ifnotexists(os.path.join(dibs_path, "trained_models", name))
project_dir = os.path.join(logdir, "trained_models")

import builtins
if not hasattr(builtins, 'oldprint'):
    builtins.oldprint = builtins.print
    log_file = os.path.join(project_dir, "log.txt")
    builtins.print = logging_print(log_file, oldprint)(builtins.oldprint)


log_writer_train = SummaryWriter(project_dir)

os.system('cp %s %s' % (__file__, project_dir))  # backup the current training file
os.system('cp %s %s' % ('../models/DiGS.py', project_dir))  # backup the models files
os.system('cp %s %s' % ('../models/losses.py', project_dir))  # backup the models files

print(args)
print()

print("NL: {}, bs: {} ({}), latent size: {}, num epochs: {} ".format(nl, batch_size, effective_batch_size, latent_size, num_epochs))
assert effective_batch_size % batch_size == 0, (batch_size, effective_batch_size)
print("Loss Type ", loss_type, 'init_type', args.init_type, 'div decay', (args.div_decay, args.div_decay_params))

# get data loaders
torch.manual_seed(0)  #change random seed for training set (so it will be different from test set
np.random.seed(0)
train_set = DFaustDataSet(args.dataset_path, args.split_path, gt_path=args.gt_path, scan_path=args.scan_path, \
    with_normals=True, points_batch=n_points)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.threads,
                                               pin_memory=True)
torch.manual_seed(seed)
np.random.seed(seed)

# get model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
device = torch.device("cuda")

DiGSNet = DiGS.DiGSNetwork(latent_size=args.latent_size, in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim,
                           nl=args.nl, encoder_type=args.encoder_type,
                           decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type,
                           sphere_init_params=args.sphere_init_params)
if args.parallel:
    if (device.type == 'cuda'):
        DiGSNet = torch.nn.DataParallel(DiGSNet)

n_parameters = utils.count_parameters(DiGSNet)
print("Number of parameters without latent vectors in the current model:{}".format(n_parameters))

num_scenes = len(train_set)
DiGSNet.latent_vecs = torch.zeros(num_scenes, latent_size).cuda()
DiGSNet.latent_vecs.requires_grad_()

n_parameters = utils.count_parameters(DiGSNet) + DiGSNet.latent_vecs.numel()
print("Number of parameters with latent vectors in the current model:{}".format(n_parameters))


# Setup Adam optimizers
optimizer = torch.optim.Adam(
    [
        {
            "params": DiGSNet.parameters(),
            "lr": args.lr,
            "weight_decay": 0
        },
        {
            "params": DiGSNet.latent_vecs,
            "lr": args.lr,
        },
    ])
n_iterations = len(train_dataloader)*(args.num_epochs)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1.0) # Does nothing
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(10, 60, 10), gamma=0.75)


DiGSNet.to(device)

criterion = DiGS.DiGSLoss(weights=args.loss_weights,loss_type=loss_type, div_decay=args.div_decay, 
                            div_type=args.div_type, div_clamp=args.div_clamp)

num_batches = len(train_dataloader)
refine_flag = True

# For each epoch
for epoch in range(num_epochs+1):
    optimizer.zero_grad()
    for batch_idx, data in enumerate(train_dataloader):
        DiGSNet.train()
        mnfld_points, nonmnfld_points, normals, indices = data # (bs, pnts, 3), (bs, pnts*9/8, 3), (bs, pnts, 3), (bs,)
        mnfld_points, nonmnfld_points, normals = mnfld_points.cuda(), nonmnfld_points.cuda(), normals.cuda()
        mnfld_points.requires_grad_(); nonmnfld_points.requires_grad_()

        latent = DiGSNet.latent_vecs[indices] # (bs, latent_size)
        mnfld_input = torch.cat([mnfld_points, latent.unsqueeze(1).repeat(1,mnfld_points.shape[1],1)], dim=-1) # (bs, pnts, 259)
        nonmnfld_input = torch.cat([nonmnfld_points, latent.unsqueeze(1).repeat(1,nonmnfld_points.shape[1],1)], dim=-1) # (bs, pnts*9/8, 259)

        output_pred = DiGSNet(nonmnfld_input, mnfld_input)
        # dict_keys(['manifold_pnts_pred', 'nonmanifold_pnts_pred', 'latent_reg', 'latent']), 
        # (bs, pnts), (bs, pnts*9/8), None, None
        mnfld_pred = output_pred['manifold_pnts_pred']
        nonmnfld_pred = output_pred['nonmanifold_pnts_pred']

        loss_dict, _ = criterion(output_pred, mnfld_points, nonmnfld_points, normals) # dict, mnfld_grad: (8, pnts, 3)
 
        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        loss_dict["lr"] = lr
        loss_dict["div_weight"] = torch.tensor(criterion.weights[4])
        utils.log_losses(log_writer_train, epoch, batch_idx, num_batches, loss_dict, batch_size)
        utils.log_weight_hist(log_writer_train, epoch, batch_idx, num_batches, DiGSNet.decoder.fc_block.net[:],
                                batch_size)
            
        loss_dict["loss"].backward()
        if (batch_idx+1) % (effective_batch_size//batch_size) == 0 or batch_idx == len(train_dataloader) - 1:
            torch.nn.utils.clip_grad_norm_(DiGSNet.parameters(), 10.)
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()

        if batch_idx % 20 == 0:
            weights = criterion.weights
            print("Weights: {}, lr={:.3e}".format(weights, lr))
            print('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                    'L_NonMnfld: {:.5f} + L_Nrml: {:.5f} + L_Eknl: {:.5f} + L_Div: {:.5f}  + L_Reg: {:.5f}'.format(
                epoch, batch_idx * batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                        loss_dict["loss"].item(), weights[0]*loss_dict["sdf_term"].item(), weights[1]*loss_dict["inter_term"].item(),
                        weights[2]*loss_dict["normals_loss"].item(), weights[3]*loss_dict["eikonal_term"].item(),
                        weights[4]*loss_dict["div_loss"].item(), weights[5]*loss_dict["latent_reg_term"].item()))
            print('Epoch: {} [{:4d}/{} ({:.0f}%)] Unweighted L_s : L_Mnfld: {:.5f},  '
                    'L_NonMnfld: {:.5f},  L_Nrml: {:.5f},  L_Eknl: {:.5f},  L_Div: {:.5f} , L_Reg: {:.5f}'.format(
                epoch, batch_idx * batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                        loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                        loss_dict["normals_loss"].item(), loss_dict["eikonal_term"].item(),
                        loss_dict["div_loss"].item(), loss_dict["latent_reg_term"].item()))
            print('')
    if epoch % 1 == 0 or epoch == num_epochs:
        model_save_path = os.path.join(project_dir, "model_{}.pkl".format(epoch))
        latent_save_path = os.path.join(project_dir, "latent_{}.pkl".format(epoch))
        print("Saving Model to {}".format(model_save_path))
        torch.save(DiGSNet.state_dict(), model_save_path)
        # print("Saving Latent Vectors to {}".format(latent_save_path))
        # torch.save(DiGSNet.latent_vecs, latent_save_path) # Very costly

        # Visualise one shape
        try:
            t0 = time.time()
            latent = DiGSNet.latent_vecs[0].detach()

            mesh_dict = utils.implicit2mesh(decoder=DiGSNet.decoder, latent=latent.cpu(), grid_res=128, 
                    get_mesh=True, device=next(DiGSNet.parameters()).device)
            
            out_dir = "{}/vis_results/".format(logdir)
            os.makedirs(out_dir, exist_ok=True)
            vis.plot_mesh(mesh_dict["mesh_trace"], mesh=mesh_dict["mesh_obj"], 
                    output_ply_path="{}/epoch_{}.ply".format(out_dir,epoch), show_ax=False,
                title_txt="Epoch {}".format(epoch), show=False)
            print('Plot took {:.3f}s'.format(time.time()-t0))
        except ValueError as e:
            print(e)
            print('Could not plot')
    
    # criterion.update_div_weight(epoch, num_epochs, args.div_decay_params)  # assumes batch size of 1
    criterion.update_div_weight(epoch, 100, args.div_decay_params)  # assumes batch size of 1
    scheduler.step()