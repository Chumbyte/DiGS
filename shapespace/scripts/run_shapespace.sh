#!/bin/bash
DIGS_DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
echo "If $DIGS_DIR is not the correct path for your DiGS repository, set it manually at the variable DIGS_DIR"

cd $DIGS_DIR/shapespace/ # To call python scripts correctly

LOGDIR='./log/shapespace/' #change to your desired log directory
IDENTIFIER='DiGS_shapespace_experiment'
mkdir -p $LOGDIR
FILE=`basename "$0"`
cp scripts/$FILE $LOGDIR # saves copy of this script so you know the args used

### DFAUST DATASET PATHS ###
##############################
DATASET_PATH='~/PhD/DIBS/dfaust_processed' # Result of the processing from IGR/SAL
GT_PATH='~/PhD/DIBS/DFaust/scripts' # Path to the GT (the registrations)
SCAN_PATH='~/PhD/DIBS/DFaust/scans' # Path to the scans
TRAIN_SPLIT_PATH='~/PhD/DIBS/DFaust/splits/dfaust/train_all.json' # Split from IGR/SAL

### MODEL HYPER-PARAMETERS ###
##############################
LAYERS=8
DECODER_HIDDEN_DIM=256
NL='sine' # 'sine' | 'relu' | 'softplus'
# SPHERE_INIT_PARAMS=(1.6 1.0)
SPHERE_INIT_PARAMS=(1.6 0.1)
INIT_TYPE='mfgi' #siren | geometric_sine | geometric_relu | mfgi
TRACK_WEIGHTS=0
### LOSS HYPER-PARAMETERS ###
#############################
LOSS_TYPE='siren_wo_n_w_div' # 'siren_wo_n_w_div' | 'siren_wo_n' | 'siren_w_div' | 'siren'
LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 1e2 1e0)
DIV_TYPE='l1'
DIVDECAY='linear' # 'linear' | 'quintic' | 'step'
# # DECAY_PARAMS: see explanation in update_div_weight in DiGS.py . Allows for diverse decay options.
DECAY_PARAMS=(1e2 0.5 1e2 0.7 0.0 0.0)
### DOMAIN HYPER-PARAMETERS ###
###############################
GRID_RES=256
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=15000
# NPOINTS=150
### TRAINING HYPER-PARAMETERS ###
#################################
EFFECTIVE_BATCH_SIZE=16
# BATCH_SIZE=8
BATCH_SIZE=1
GPU=0
EPOCHS=500
LR=5e-5
GRAD_CLIP_NORM=10.0


### EVAL SETTINGS ###
#################################
TEST_EPOCH=500 # Set this to the epoch you want to eval at
TEST_LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 0e2 1e0) # At test time, don't need div loss
# NUM_LATENT_ITERS=100 # Number of iterations to optimise the latent codes for
NUM_LATENT_ITERS=800 # Number of iterations to optimise the latent codes for
TEST_SPLIT_PATH='~/PhD/DIBS/DFaust/splits/dfaust/test_all.json'
# TEST_SPLIT_PATH='~/PhD/DIBS/DFaust/splits/dfaust/train_all.json'
# TEST_SPLIT_PATH='~/PhD/DIBS/DFaust/splits/dfaust/vis.json'
# TEST_SPLIT_PATH='~/PhD/DIBS/DFaust/splits/dfaust/punching.json'
TEST_RES=100 # Resolution


python3 train_shapespace.py --logdir $LOGDIR$IDENTIFIER --dataset_path $DATASET_PATH --gt_path $GT_PATH --scan_path $SCAN_PATH --split_path TRAIN_SPLIT_PATH --grid_res $GRID_RES --loss_type $LOSS_TYPE --num_epochs $EPOCHS --gpu_idx $GPU --n_points $NPOINTS --batch_size $BATCH_SIZE  --lr $LR --effective_batch_size $EFFECTIVE_BATCH_SIZE --nonmnfld_sample_type $NONMNFLD_SAMPLE_TYPE --decoder_n_hidden_layers $LAYERS --decoder_hidden_dim ${DECODER_HIDDEN_DIM} --div_decay $DIVDECAY --div_decay_params ${DECAY_PARAMS[@]} --div_type $DIV_TYPE --init_type ${INIT_TYPE} --nl ${NL} --track_weights ${TRACK_WEIGHTS} --sphere_init_params ${SPHERE_INIT_PARAMS[@]} --loss_weights ${LOSS_WEIGHTS[@]} --grad_clip_norm ${GRAD_CLIP_NORM[@]}

python3 eval_shapespace.py --logdir $LOGDIR$IDENTIFIER  --num_epochs $TEST_EPOCH --test_loss_weights $TEST_LOSS_WEIGHTS --num_latent_iters $NUM_LATENT_ITERS --dataset_path $DATASET_PATH --gt_path $GT_PATH --scan_path $SCAN_PATH --grid_res $GRID_RES --loss_type $LOSS_TYPE --gpu_idx $GPU --n_points $NPOINTS --batch_size $BATCH_SIZE --lr $LR --effective_batch_size $EFFECTIVE_BATCH_SIZE --nonmnfld_sample_type $NONMNFLD_SAMPLE_TYPE --decoder_n_hidden_layers $LAYERS --decoder_hidden_dim ${DECODER_HIDDEN_DIM} --div_decay $DIVDECAY --div_decay_params ${DECAY_PARAMS[@]} --div_type $DIV_TYPE --init_type ${INIT_TYPE}} --nl ${NL} --sphere_init_params ${SPHERE_INIT_PARAMS[@]} --loss_weights ${LOSS_WEIGHTS[@]} --grad_clip_norm ${GRAD_CLIP_NORM[@]}

