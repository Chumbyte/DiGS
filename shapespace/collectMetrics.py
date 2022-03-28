import argparse
import os
import sys
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)
import json
import torch
from pyhocon import ConfigFactory
import trimesh
from scipy.spatial import cKDTree as KDTree
import numpy as np
import gc

import re
from shapespace.dfaust_dataset import DFaustDataSet
import utils.utils as utils

#  python shapespace/eval.py --checkpoint 1200 --exp-name dfaust_pretrained --split dfaust/test_all.json --exps-dir trained_models

def path_to_info(path):
    seqs = ['one_leg_loose', 'jiggle_on_toes', 'personal_move', 'knees', 'punching', 'light_hopping_loose', 'running_on_spot_bugfix', 'running_on_spot', 'one_leg_jump', 'shake_arms', 'shake_hips', 'jumping_jacks', 'shake_shoulders', 'light_hopping_stiff', 'hips', 'chicken_wings']
    prefixed = ['running_on_spot_bugfix', 'shake_hips']
    
    seq = None # posename
    for seq_name in prefixed:
        if seq_name in path:
            seq = seq_name
    if seq is None:
        for seq_name in seqs:
            if seq_name in path:
                seq = seq_name
    if seq is None:
        print('Could not find seq for {}'.format(path))
        return None
    
    pid = re.search('500\d\d',path).group() # shapename
    tag = re.search('{}\.\d\d\d\d\d\d'.format(seq),path).group() # which temporal instance

    return pid, seq, tag

def get_all_chamfers(files, base_path, ds):
    gt_all_one_dists = []
    gt_all_two_dists = []
    scan_all_one_dists = []
    scan_all_two_dists = []
    print_every = 1
    for i, file in enumerate(files[:]):
        if 'ply' not in file:
            continue
        recon_path = os.path.expanduser(os.path.join(base_path,file))
        pid, seq, tag = path_to_info(recon_path)# 50002/shake_hips/shake_hips.004743.obj'
        if i % print_every == 0:
            print("{} / {}".format(i, len(files)))
            print(recon_path)
        reconstruction = trimesh.load(recon_path) # <trimesh.Trimesh(vertices.shape=(2864900, 3), faces.shape=(5726352, 3))>

        gt_path = os.path.join(os.path.expanduser('~/PhD/DIBS/DFaust/scripts/'),pid,seq,tag+'.obj')
        gt_registration = trimesh.load(gt_path)

        scan_path = os.path.join(os.path.expanduser('~/PhD/DIBS/DFaust/scans/'),pid,seq,tag+'.ply')
        scan = trimesh.load(gt_path)

        normalization_params_filename = os.path.join(os.path.expanduser('~/PhD/DIBS/dfaust_processed/'),pid,seq,tag+'_normalization.npy')
        normalization_params = np.load(normalization_params_filename,allow_pickle=True)
        # scale = normalization_params.item()['scale']
        # center = normalization_params.item()['center']
        scale = 1
        center = np.array([0,0,0])

        # ground_truth_points = trimesh.Trimesh(trimesh.sample.sample_surface(gt_registration,30000)[0])

        results_points = trimesh.sample.sample_surface(reconstruction, 30000)[0]
        gt_points = trimesh.sample.sample_surface(gt_registration,30000)[0]
        scan_points = trimesh.sample.sample_surface(scan,30000)[0]
        chamfer, hausdorff, one_sided_results, pod_data, cdp_data, malcv_data, gt_all_distances = \
            utils.recon_metrics(results_points*scale+center, gt_points, return_all=True)
        if i % print_every == 0:
            print(chamfer, hausdorff, *one_sided_results)

        chamfer, hausdorff, one_sided_results, pod_data, cdp_data, malcv_data, scan_all_distances = \
            utils.recon_metrics(results_points*scale+center, scan_points, return_all=True)
        if i % print_every == 0:
            print(chamfer, hausdorff, *one_sided_results)
            print()

        gt_all_one_dists.extend(gt_all_distances[0])
        gt_all_two_dists.extend(gt_all_distances[1])
        scan_all_one_dists.extend(scan_all_distances[0])
        scan_all_two_dists.extend(scan_all_distances[1])
    
    import matplotlib.pyplot as plt
    gt_all_one_dists.sort()
    gt_all_two_dists.sort()
    scan_all_one_dists.sort()
    scan_all_two_dists.sort()
    gt_all_one_dists_np = np.array(gt_all_one_dists)
    gt_all_two_dists_np = np.array(gt_all_two_dists)
    scan_all_one_dists_np = np.array(scan_all_one_dists)
    scan_all_two_dists_np = np.array(scan_all_two_dists)

    print("Recon-to-Reg,  Mean: {:.6f}, Median: {:.6g}".format(gt_all_two_dists_np.mean(), np.median(gt_all_two_dists_np)))
    print("Reg-to-Recon,  Mean: {:.6f}, Median: {:.6g}".format(gt_all_one_dists_np.mean(), np.median(gt_all_one_dists_np)))
    print("Recon-to-Scan,  Mean: {:.6f}, Median: {:.6g}".format(scan_all_two_dists_np.mean(), np.median(scan_all_two_dists_np)))
    print("Scan-to-Recon,  Mean: {:.6f}, Median: {:.6g}".format(scan_all_one_dists_np.mean(), np.median(scan_all_one_dists_np)))
    # fig = plt.figure(figsize=(15,5))
    # plt.subplot(1,4,1); plt.step(gt_all_two_dists_np*100, np.arange(len(gt_all_two_dists))/len(gt_all_two_dists)); plt.xlim(0,2)
    # plt.title('Recon-to-Reg'); plt.xlabel('Distance (cm)'); plt.ylabel('Coverage (%)')
    # plt.subplot(1,4,2); plt.step(gt_all_one_dists_np*100, np.arange(len(gt_all_one_dists))/len(gt_all_one_dists)); plt.xlim(0,2)
    # plt.title('Reg-to-Recon'); plt.xlabel('Distance (cm)'); plt.ylabel('Coverage (%)')
    # plt.subplot(1,4,3); plt.step(scan_all_two_dists_np*100, np.arange(len(scan_all_two_dists))/len(scan_all_two_dists)); plt.xlim(0,2)
    # plt.title('Recon-to-Scan'); plt.xlabel('Distance (cm)'); plt.ylabel('Coverage (%)')
    # plt.subplot(1,4,4); plt.step(scan_all_one_dists_np*100, np.arange(len(scan_all_one_dists))/len(scan_all_one_dists)); plt.xlim(0,2)
    # plt.title('Scan-to-Recon'); plt.xlabel('Distance (cm)'); plt.ylabel('Coverage (%)')
    # plt.show()
    # import pdb; pdb.set_trace()

if __name__ == '__main__':
    dataset_path = '~/PhD/DIBS/dfaust_processed'
    test_split = '~/PhD/DIBS/DFaust/splits/dfaust/test_all.json'
    gt_path = '~/PhD/DIBS/DFaust/scripts'
    scan_path = '~/PhD/DIBS/DFaust/scans'
    dataset = DFaustDataSet(dataset_path, test_split, gt_path=gt_path, scan_path=scan_path, \
        with_normals=True, points_batch=8000)

    # recon_path = '/home/chamin/PhD/DIBS/shapespace/DiBS/shapespace/results/2021-05-29_09-48-18'
    recon_path = '/home/chamin/PhD/DIBS/shapespace/DiBS/shapespace/results/2021-06-03_11-02-10'
    files = os.listdir(recon_path)
    get_all_chamfers(files, recon_path, dataset)
    # import pdb; pdb.set_trace()