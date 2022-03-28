# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os, sys, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recon_dataset as dataset
import torch
import utils.visualizations as vis
import numpy as np
import models.DiGS as DiGS
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import utils.utils as utils
import surface_recon_args
import traceback

# get training parameters
args = surface_recon_args.get_args()
gpu_idx, nl, n_points, batch_size, n_samples, latent_size, lr, num_epochs, logdir, \
model_name, n_loss_type, normalize_normal_loss, unsigned_n, unsigned_d, loss_type, seed, encoder_type,\
    model_dirpath, inter_loss_type =\
    args.gpu_idx, args.nl, args.n_points, args.batch_size, args.n_samples, args.latent_size, \
    args.lr, args.num_epochs, args.logdir, args.model_name, args.n_loss_type, \
    args.normalize_normal_loss, args.unsigned_n, args.unsigned_d, args.loss_type, args.seed, args.encoder_type, \
    args.model_dirpath, args.inter_loss_type

file_path = os.path.join(args.dataset_path, args.file_name)
logdir = os.path.join(logdir, args.file_name.split('.')[0])

# set up logging
log_file, log_writer_train, log_writer_test, model_outdir = utils.setup_logdir(logdir, args)
os.system('cp %s %s' % (__file__, logdir))  # backup the current training file
os.system('cp %s %s' % ('../models/DiGS.py', logdir))  # backup the models files
os.system('cp %s %s' % ('../models/losses.py', logdir))  # backup the losses files

# get data loaders
torch.manual_seed(0)  #change random seed for training set (so it will be different from test set
np.random.seed(0)
train_set = dataset.ReconDataset(file_path, n_points, n_samples, args.grid_res, args.nonmnfld_sample_type,
                                 requires_dist=args.requires_dist, requires_curvatures=args.requires_curvs)
train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4,
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
utils.log_string("Number of parameters in the current model:{}".format(n_parameters), log_file)

# Setup Adam optimizers
optimizer = optim.Adam(DiGSNet.parameters(), lr=lr, betas=(0.9, 0.999))
n_iterations = args.n_samples*(args.num_epochs + 1)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=1.0) # Does nothing
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(2000, args.n_samples*(num_epochs + 1), 2000), 
#       gamma=0.5)  # milestones in number of optimizer iterations

if not args.refine_epoch == 0:
    refine_model_filename = os.path.join(model_outdir,
                                         '%s_model_%d.pth' % (model_name, args.refine_epoch))
    DiGSNet.load_state_dict(torch.load(refine_model_filename, map_location=device))
    optimizer.step()

DiGSNet.to(device)

criterion = DiGS.DiGSLoss(weights=args.loss_weights, loss_type=loss_type, div_decay=args.div_decay, 
                            div_type=args.div_type, div_clamp=args.div_clamp)

num_batches = len(train_dataloader)
refine_flag = True
grid_points = train_set.grid_points

# For each epoch
for epoch in range(num_epochs+1):
    if epoch <= args.refine_epoch and refine_flag and not args.refine_epoch == 0:
        scheduler.step()
        continue
    else:
        refine_flag = False

    # For each batch in the dataloader
    for batch_idx, data in enumerate(train_dataloader):
        if batch_idx in [0, 1, 5, 10, 50, 100] or batch_idx % 500 == 0:
        # if batch_idx % 500 == 0 :
            try:
                shapename = args.file_name.split('.')[0]
                output_dir = os.path.join(logdir, '..', 'result_meshes')
                os.makedirs(output_dir, exist_ok=True)
                output_ply_filepath = os.path.join(output_dir, shapename+'_iter_{}.ply'.format(batch_idx))
                print('Saving to ', output_ply_filepath)
                cp, scale, bbox = train_set.cp, train_set.scale, train_set.bbox
                mesh_dict = utils.implicit2mesh(DiGSNet.decoder, None, 128, translate=-cp, scale=1/scale,
                                    get_mesh=True, device=device, bbox=bbox)
                mesh_dict["mesh_obj"].export(output_ply_filepath, vertex_normal=True)
            except:
                print(traceback.format_exc())
                print('Could not generate mesh')
                print()

        DiGSNet.zero_grad()
        DiGSNet.train()

        mnfld_points, mnfld_n_gt, nonmnfld_points, nonmnfld_dist, mnfld_curvs = \
            data['points'].to(device), data['mnfld_n'].to(device),  data['nonmnfld_points'].to(device),\
        data['nonmnfld_dist'].to(device), data['mnfld_curvs'].to(device)

        mnfld_points.requires_grad_()
        nonmnfld_points.requires_grad_()

        output_pred = DiGSNet(nonmnfld_points, mnfld_points)
        
        loss_dict, _ = criterion(output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt, nonmnfld_dist, mnfld_curvs,)
        lr = torch.tensor(optimizer.param_groups[0]['lr'])
        loss_dict["lr"] = lr
        utils.log_losses(log_writer_train, epoch, batch_idx, num_batches, loss_dict, batch_size)

        loss_dict["loss"].backward()
         
        torch.nn.utils.clip_grad_norm_(DiGSNet.parameters(), 10.)

        optimizer.step()

        # Output training stats
        if batch_idx % 10 == 0:
            # if args.track_weights:
            #     utils.log_weight_hist(log_writer_train, epoch, batch_idx, num_batches, DiGSNet.decoder.net[:],
            #                           batch_size)

            weights = criterion.weights
            utils.log_string("Weights: {}, lr={:.3e}".format(weights, lr), log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                    'L_NonMnfld: {:.5f} + L_Nrml: {:.5f} + L_Eknl: {:.5f} + L_Div: {:.5f}'.format(
                epoch, batch_idx * batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                        loss_dict["loss"].item(), weights[0]*loss_dict["sdf_term"].item(), weights[1]*loss_dict["inter_term"].item(),
                        weights[2]*loss_dict["normals_loss"].item(), weights[3]*loss_dict["eikonal_term"].item(),
                        weights[4]*loss_dict["div_loss"].item()), log_file)
            utils.log_string('Epoch: {} [{:4d}/{} ({:.0f}%)] Unweighted L_s : L_Mnfld: {:.5f},  '
                    'L_NonMnfld: {:.5f},  L_Nrml: {:.5f},  L_Eknl: {:.5f},  L_Div: {:.5f}'.format(
                epoch, batch_idx * batch_size, len(train_set), 100. * batch_idx / len(train_dataloader),
                        loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                        loss_dict["normals_loss"].item(), loss_dict["eikonal_term"].item(),
                        loss_dict["div_loss"].item()), log_file)
            utils.log_string('', log_file)
            
        criterion.update_div_weight(epoch * args.n_samples + batch_idx, (num_epochs + 1) * args.n_samples,
                                    args.div_decay_params)  # assumes batch size of 1
        scheduler.step()

        # save model
        if batch_idx % 1000 == 0 :
            utils.log_string("saving model to file :{}".format('%s_model_batch_%d.pth' % (model_name, batch_idx)),
                             log_file)
            torch.save(DiGSNet.state_dict(),
                       os.path.join(model_outdir, '%s_model_batch_%d.pth' % (model_name, batch_idx)))

    # save model
    if epoch % 50 == 0 or epoch == num_epochs:
        utils.log_string("saving model to file :{}".format('%s_generator_model_%d.pth' % (model_name, epoch)),
                         log_file)
        torch.save(DiGSNet.state_dict(),
                   os.path.join(model_outdir, '%s_model_%d.pth' % (model_name, epoch)))