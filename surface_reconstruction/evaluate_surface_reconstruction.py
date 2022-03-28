# Yizhak Ben-Shabat (Itzik) <sitzikbs@gmail.com>
# Chamin Hewa Koneputugodage <chamin.hewa@anu.edu.au>
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils.utils as utils
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='Berger_GT',
                    help='name of dataset')
parser.add_argument('--gt_path', type=str, default='../data/Berger_scans/',
                    help='path to ground truth data directory')
parser.add_argument('--results_path', type=str, default='./log/surface_reconstruction/DiGS_surf_recon_experiment/result_meshes',
                    help='path to results directory')
parser.add_argument('--output_path', type=str, default='./log/surface_reconstruction/DiGS_surf_recon_experiment/results/',
                    help='path to where output the restuls .txt file')
parser.add_argument('--baseline_name', type=str, default='DEBUG', help='name of baseline method to evaluate')
parser.add_argument('--sample_type', type=str, default='vertices', help='how to sample points on resulst for evaluation'
                                                                        ' vertices | other')
parser.add_argument('--n_points', type=int, default=30000,
                    help='number of points to sample on the mesh (if not using vertices')
parser.add_argument('--alphas', nargs='+', default=[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,
                                                    0.0009, 0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016,
                                                    0.0017, 0.0018, 0.0019, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02],
                    help='distances to evaluate pod metric')
parser.add_argument('--percentiles', nargs='+', default=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95],
                    help='distances to evaluate percentile chamfer distance metric')
parser.add_argument('--k_local_var', nargs='+', default=[10, 20, 30, 40, 50],
                    help='list of number of neighbors to consider when computing malcv metric')
parser.add_argument('--eval_mode', type=str, default='any', help='indicates which evaluation mode to use. any evaluates any file in the results path.'
                                                                 ' berger ealuates only the surface reconstruction benchmark files.  any | berger')
args = parser.parse_args()


gt_file_list = [f for f in os.listdir(args.gt_path) if os.path.isfile(os.path.join(args.gt_path, f))]
gt_file_list.sort()
results_file_list = [f for f in os.listdir(args.results_path) if os.path.isfile(os.path.join(args.results_path, f))]
results_file_list.sort()
if args.eval_mode == 'any':
    model_list = [model_name.split('.')[0] for model_name in results_file_list] # use this to evaluate all files in a directory
else:
    model_list = ['anchor', 'daratech', 'dc', 'gargoyle', 'lord_quas'] # use this to only evaluate the Berger dataset resutls

output_file_name = os.path.join(args.output_path, args.baseline_name + '_' + args.dataset_name + '_results.json')
os.makedirs(args.output_path, exist_ok=True)
output_dict = {}
if len(model_list) > 0 :
    for i, model in enumerate(model_list):
        print('Evaluating ' + args.baseline_name +' shape ' + model + '.ply ...')
        # load gt points
        gt_filename = [filname for filname in gt_file_list if model in filname]
        gt_pc_filename = os.path.join(args.gt_path, gt_filename[0])
        gt_points = utils.load_reconstruction_data(gt_pc_filename, n_points=args.n_points, sample_type=args.sample_type)
        # load results points
        # results_filename = [filname for filname in results_file_list if model in filname]
        results_filename = [filname for filname in results_file_list if model+'.ply' in filname]
        if len(results_filename) > 1:
            raise Warning("files with similar names may cause issues, rename files or revise code to fix this issue")
        elif len(results_filename) == 0:
            continue
        results_pc_filename = os.path.join(args.results_path, results_filename[0])
        results_points = utils.load_reconstruction_data(results_pc_filename, n_points=args.n_points, sample_type=args.sample_type)

        chamfer, hausdorff, one_sided_results, pod_data, cdp_data, malcv_data = \
            utils.recon_metrics(results_points, gt_points, one_sided=False, alphas=args.alphas,
                                percentiles=args.percentiles, k=args.k_local_var)

        output_dict[model] = {"dc": chamfer, "dh": hausdorff, "one_sided_dc": one_sided_results[0],
                              "one_sided_dh": one_sided_results[-2], "pod": pod_data, "cdp": cdp_data,
                              "malcv": malcv_data}
    output_dict["pod_alphas"] = args.alphas
    output_dict["cdp_percentiles"] = args.percentiles
    output_dict["k_local_var"] = args.k_local_var
    output_dict["percentiles"] = args.percentiles
    with open(output_file_name, "w") as outfile:
        json.dump(output_dict, outfile)

else:
    print("Sorry, no results models found for evaluation.")

print("Evaluation complete\n")