# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import recon_dataset as dataset
import torch
import utils.visualizations as vis
import numpy as np
import models.DiGS as DiGS
import torch.nn.parallel
import utils.utils as utils
import surface_recon_args

# get training parameters
args = surface_recon_args.get_test_args()
gpu_idx, nl, n_points, batch_size, n_samples, logdir,  model_name, seed, encoder_type =\
    args.gpu_idx, args.nl, args.n_points, args.batch_size, args.n_samples,  args.logdir, \
    args.model_name, args.seed, args.encoder_type

file_path = os.path.join(args.dataset_path, args.file_name)
if args.export_mesh:
    outdir = os.path.join(os.path.dirname(logdir), 'result_meshes')
    os.makedirs(outdir, exist_ok=True)
    output_ply_filepath = os.path.join(outdir, args.file_name)
# get data loader
torch.manual_seed(seed)
np.random.seed(seed)
test_set = dataset.ReconDataset(file_path, n_points*n_samples, n_samples=1, res=args.grid_res, sample_type='grid',
                                requires_dist=False)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# get model
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
device = torch.device("cuda")

DiGSNet = DiGS.DiGSNetwork(latent_size=args.latent_size, in_dim=3, decoder_hidden_dim=args.decoder_hidden_dim,
                           nl=args.nl, encoder_type=args.encoder_type,
                           decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type)
if args.parallel:
    if (device.type == 'cuda'):
        DiGSNet = torch.nn.DataParallel(DiGSNet)

model_dir = os.path.join(logdir, 'trained_models')
if args.specific_model_path == '':
    trained_model_filename = os.path.join(model_dir, '%s_model_%d.pth' % (model_name, args.epoch_n))
    # trained_model_filename = os.path.join(model_dir, '%s_model_%s.pth' % (model_name, 'batch_5000'))
else:
    trained_model_filename = args.specific_model_path
    output_ply_filepath = os.path.join(outdir, os.path.basename(args.specific_model_path).split('.')[0] + '.ply')
DiGSNet.load_state_dict(torch.load(trained_model_filename, map_location=device))
DiGSNet.to(device)
latent = None

print("Converting implicit to mesh for file {}".format(args.file_name))
cp, scale, bbox = test_set.cp, test_set.scale, test_set.bbox
test_set, test_dataloader, clean_points_gt, normals_gt,  nonmnfld_points, data = None, None, None, None, None, None  # free up memory
mesh_dict = utils.implicit2mesh(DiGSNet.decoder, latent, args.grid_res, translate=-cp, scale=1/scale,
                                get_mesh=True, device=device, bbox=bbox)
vis.plot_mesh(mesh_dict["mesh_trace"], mesh=mesh_dict["mesh_obj"], output_ply_path=output_ply_filepath, show_ax=False,
              title_txt=args.file_name.split('.')[0], show=False)

print("Conversion complete.")
