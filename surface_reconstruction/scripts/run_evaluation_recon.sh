#!/bin/bash
DIGS_DIR='/home/chamin/DiGSProject/DiBS'  # change to your DiGS path
DATASET_PATH=$DIGS_DIR'/data/deep_geometric_prior_data/' # change to your dataset path

cd $DIGS_DIR/surface_reconstruction/

LOGDIR='./log/surface_reconstruction/' #change to your desired log directory
SCAN_PATH=$DATASET_PATH'/scans/'
GRID_RES=256
UNITS=256
SAMPLE_TYPE='vertices'
N_SAMPLES=1000000

for IDENTIFIER in 'DiGS_surf_recon_experiment'
do
RESULTS_DIR=$LOGDIR$IDENTIFIER'/result_meshes'
OUTPUT_DIR=$LOGDIR$IDENTIFIER'/results'
python3 evaluate_surface_reconstruction.py --results_path $RESULTS_DIR --output_path $OUTPUT_DIR --baseline_name $IDENTIFIER --sample_type $SAMPLE_TYPE --n_points $N_SAMPLES --gt_path $DATASET_PATH'/scans' --dataset_name 'Berger_scans'
python3 evaluate_surface_reconstruction.py --results_path $RESULTS_DIR --output_path $OUTPUT_DIR --baseline_name $IDENTIFIER --sample_type $SAMPLE_TYPE --n_points $N_SAMPLES --gt_path $DATASET_PATH'/ground_truth' --dataset_name 'Berger_GT'
done