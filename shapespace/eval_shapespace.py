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
# from shapespace.shapespace_utils import logging_print, mkdir_ifnotexists
import trimesh


def optimize_latent(num_latent_iters, mnfld_points, nonmnfld_points, normals, network, latent_size, lr=1.0e-3):
    # mnfld_points: (1, pnts, 3), nonmnfld_points: (1, pnts', 3)
    assert len(mnfld_points.shape)==3 and len(nonmnfld_points.shape)==3, (mnfld_points.shape, nonmnfld_points.shape)
    assert mnfld_points.shape[0]==1 and nonmnfld_points.shape[0]==1, (mnfld_points.shape, nonmnfld_points.shape)
    assert mnfld_points.shape[2]==3 and nonmnfld_points.shape[2]==3, (mnfld_points.shape, nonmnfld_points.shape)

    network.train()
    mnfld_points.requires_grad_(); nonmnfld_points.requires_grad_()

    latent = torch.ones(latent_size).normal_(0, 1 / latent_size).to(mnfld_points.device) # (ls,)
    latent.requires_grad = True
    optimizer = torch.optim.Adam([latent], lr=lr)
    import time
    t0 = time.time()
    for i in range(num_latent_iters):
        
        mnfld_input = torch.cat([mnfld_points, latent.unsqueeze(0).unsqueeze(0).repeat(1,mnfld_points.shape[1],1)], dim=-1) # (1,pnts, 259)
        nonmnfld_input = torch.cat([nonmnfld_points, latent.unsqueeze(0).unsqueeze(0).repeat(1,nonmnfld_points.shape[1],1)], dim=-1) # (1,pnts', 259)

        output_pred = network(nonmnfld_input, mnfld_input)
        
        loss_dict, _ = criterion(output_pred, mnfld_points, nonmnfld_points, normals) # dict, mnfld_grad: (8, pnts, 3)

        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()

        if i % 50 == 0 or i == num_latent_iters - 1:
            print('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                    'L_NonMnfld: {:.5f} + L_Nrml: {:.5f}+ L_Eknl: {:.5f}  + L_Div: {:.5f} + L_Reg: {:.5f}'.format(
                0, i, num_latent_iters, 100. * i / num_latent_iters,
                        loss_dict["loss"].item(), weights[0]*loss_dict["sdf_term"].item(), weights[1]*loss_dict["inter_term"].item(),
                        weights[2]*loss_dict["normals_loss"].item(), weights[3]*loss_dict["eikonal_term"].item(),
                        weights[4]*loss_dict["div_loss"].item(), weights[5]*loss_dict["latent_reg_term"].item()))
            print('Epoch: {} [{:4d}/{} ({:.0f}%)] Loss: {:.5f} = L_Mnfld: {:.5f} + '
                    'L_NonMnfld: {:.5f} + L_Nrml: {:.5f}+ L_Eknl: {:.5f}  + L_Div: {:.5f} + L_Reg: {:.5f}'.format(
                0, i, num_latent_iters, 100. * i / num_latent_iters,
                        loss_dict["loss"].item(), loss_dict["sdf_term"].item(), loss_dict["inter_term"].item(),
                        loss_dict["normals_loss"].item(), loss_dict["eikonal_term"].item(),
                        loss_dict["div_loss"].item(), loss_dict["latent_reg_term"].item()))

    print(("Time for latent opt", time.time()-t0))
    network.eval()
    return latent.detach()

digs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #DiGS/

args = shapespace_dfaust_args.get_args()
print(args)
print()
gpu_idx, nl, n_points, batch_size, effective_batch_size, latent_size, num_epochs, logdir, \
model_name, n_loss_type, normalize_normal_loss, unsigned_n, unsigned_d, loss_type, seed, encoder_type,\
    model_dirpath, inter_loss_type =\
    args.gpu_idx, args.nl, args.n_points, args.batch_size, args.effective_batch_size, args.latent_size, \
    args.num_epochs, args.logdir, args.model_name, args.n_loss_type, \
    args.normalize_normal_loss, args.unsigned_n, args.unsigned_d, args.loss_type, args.seed, args.encoder_type, \
    args.model_dirpath, args.inter_loss_type

# Evaluate at args.num_epochs
epoch = args.num_epochs

print("NL: {}, bs: {} ({}), latent size: {}, num epochs: {} ".format(nl, batch_size, effective_batch_size, latent_size, num_epochs))
assert effective_batch_size % batch_size == 0, (batch_size, effective_batch_size)
print("Loss Type ", loss_type, 'div decay', (args.div_decay, args.div_decay_params))

# get data loaders
torch.manual_seed(0)  #change random seed for training set (so it will be different from test set
np.random.seed(0)

torch.manual_seed(seed)
np.random.seed(seed)
# test_set = dataset.ReconDataset(file_path, n_points, n_samples, args.grid_res, args.nonmnfld_sample_type, requires_dist=args.requires_dist)
# test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4,
#                                               pin_memory=True)

# get model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
device = torch.device("cuda:" + str(gpu_idx) if (torch.cuda.is_available()) else "cpu")
DiGSNet = DiGS.DiGSNetwork(latent_size=latent_size, in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim,
                        nl=args.nl, encoder_type=args.encoder_type,
                        decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type)
DiGSNet.to(device)
if args.parallel:
    if (device.type == 'cuda'):
        DiGSNet = torch.nn.DataParallel(DiGSNet)

# Main eval arguments
test_split_path = args.test_split_path
split_name = os.path.basename(test_split_path)
weights = args.test_loss_weights
num_latent_iters = args.num_latent_iters
resolution = args.test_res

criterion = DiGS.DiGSLoss(weights=args.weights, loss_type=loss_type, div_decay=args.div_decay, 
                            div_type=args.div_type, div_clamp=args.div_clamp)

batch_size = 1 # For eval, always have bs=1

dataset = DFaustDataSet(args.dataset_path, test_split_path, gt_path=args.gt_path, scan_path=args.scan_path, \
    with_normals=True, points_batch=n_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                                # shuffle=True, 
                                                shuffle=False, 
                                                num_workers=args.threads,
                                                pin_memory=True)

model_path = os.path.join(logdir, "trained_models" , 'model_{}.pkl'.format(epoch))
print("Loading from {}".format(model_path))
DiGSNet.load_state_dict(torch.load(model_path))
DiGSNet.eval()

# For each epoch
for batch_idx, data in enumerate(dataloader):
    mnfld_points, nonmnfld_points, normals, indices = data # (bs, pnts, 3), (bs, pnts*9/8, 3), (bs, pnts, 3), (bs,)
    mnfld_points, nonmnfld_points, normals = mnfld_points.cuda(), nonmnfld_points.cuda(), normals.cuda()
    assert len(indices) == 1
    index = indices[0]

    info = dataset.get_info(index)
    shapename = str.join('_', info)
    pc_path = os.path.join(*info)
    gt_mesh_filename = dataset.gt_files[index]
    normalization_params_filename = dataset.normalization_files[index]
    normalization_params = np.load(normalization_params_filename,allow_pickle=True)
    scale = normalization_params.item()['scale']
    center = normalization_params.item()['center']
    
    latent = optimize_latent(num_latent_iters, mnfld_points, nonmnfld_points, normals, DiGSNet, latent_size)

    mnfld_points = mnfld_points.detach().squeeze() # (pnts, 3)

    with torch.no_grad():
        print('before digs implicit2mesh'); t0 = time.time()

        bbox = np.array([mnfld_points.min(axis=0)[0].cpu().numpy(), mnfld_points.max(axis=0)[0].cpu().numpy()]).transpose()
        # bbox = np.array([[-1,1], [-1,1], [-1,1]])*2
        # bbox = np.array([[-10,10], [-10,10], [-10,10]])
        gt_points = trimesh.sample.sample_surface(trimesh.load(gt_mesh_filename),30000)[0]
        
        try:
            mesh_dict = utils.implicit2mesh(decoder=DiGSNet.decoder, latent=latent.cpu(), grid_res=resolution, translate=-center, scale=scale, 
                    get_mesh=True, device=next(DiGSNet.parameters()).device, bbox=bbox)
            results_points = trimesh.sample.sample_surface(mesh_dict["mesh_obj"],30000)[0]
            print('after digs implicit2mesh', time.time()-t0); t0 = time.time()
            
            out_dir = "{}/vis_results/epoch_{}".format(logdir, epoch)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            vis.plot_mesh(mesh_dict["mesh_trace"], mesh=mesh_dict["mesh_obj"], 
                    output_ply_path="{}/{}_{}.ply".format(out_dir,split_name,shapename), show_ax=False,
                title_txt=shapename, show=False)
            print('after digs plot_mesh', time.time()-t0); t0 = time.time()

            # chamfer, hausdorff, one_sided_results, pod_data, cdp_data, malcv_data = utils.recon_metrics(results_points*scale+center, gt_points)
            chamfer, hausdorff, one_sided_results, pod_data, cdp_data, malcv_data = utils.recon_metrics(results_points, gt_points)
            print(chamfer, hausdorff, *one_sided_results)
            print('after digs res', time.time()-t0); t0 = time.time()
        except ValueError as e:
            print(e)
