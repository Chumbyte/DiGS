#!/bin/bash
DIGS_DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
DATASET_PATH=$DIGS_DIR'/data/deep_geometric_prior_data/' # change to your dataset path
echo "If $DIGS_DIR is not the correct path for your DiGS repository, set it manually at the variable DIGS_DIR"
echo "If $DATASET_PATH is not the correct path for the NSP dataset, change the variable DATASET_PATH"

cd $DIGS_DIR/surface_reconstruction/ # To call python scripts correctly

SCAN_PATH=$DATASET_PATH'/scans/'

# CHnge the below!
MESH_PATH='./surface_reconstruction/log/surface_reconstruction/DiGS_surf_recon_experiment/result_meshes'
# MESH_PATH='/home/chamin/DiGS-Github/DiBS/data/vises/Surface_reconstruction_results/DiGS'
MESH_PATH='/home/chamin/DiGS-Github/DiBS/data/vises/Surface_reconstruction_results/DiGS+n'
MESH_PATH='/home/chamin/DiGS-Github/DiBS/surface_reconstruction/log/surface_reconstruction_new_v37/DiGS_surf_recon_experiment/result_meshes'

LOGDIR='.'

python3 compute_metrics_srb.py --logdir $LOGDIR --dataset_path $DATASET_PATH --results_path $MESH_PATH