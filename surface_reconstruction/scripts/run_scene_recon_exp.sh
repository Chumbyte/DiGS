#!/bin/bash
DIGS_DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
DATASET_PATH=$DIGS_DIR'/data/scene_reconstruction/' # change to your dataset path
echo "If $DIGS_DIR is not the correct path for your DiGS repository, set it manually at the variable DIGS_DIR"
echo "If $DATASET_PATH is not the correct path for the scene-recon dataset, change the variable DATASET_PATH"

cd $DIGS_DIR/surface_reconstruction/ # To call python scripts correctly

LOGDIR='./log/scene_reconstruction/' #change to your desired log directory
IDENTIFIER='DiGS_scene_recon_experiment'
mkdir -p $LOGDIR
FILE=`basename "$0"`
cp scripts/$FILE $LOGDIR # saves copy of this script so you know the args used
FILENAME='interior_room.ply'


### MODEL HYPER-PARAMETERS ###
##############################
# LAYERS=4
LAYERS=8
# DECODER_HIDDEN_DIM=256
DECODER_HIDDEN_DIM=512
NL='sine' # 'sine' | 'relu' | 'softplus'
# SPHERE_INIT_PARAMS=(1.6 1.0)
SPHERE_INIT_PARAMS=(1.6 0.1)
INIT_TYPE='siren' #siren | geometric_sine | geometric_relu | mfgi
TRACK_WEIGHTS=0
### LOSS HYPER-PARAMETERS ###
#############################
LOSS_TYPE='siren_wo_n_w_div' # 'siren_wo_n_w_div' | 'siren_wo_n' | 'siren_w_div' | 'siren'
LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 1e1)
# LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 1e2)
DIV_TYPE='l1'
DIVDECAY='linear' # 'linear' | 'quintic' | 'step'
DECAY_PARAMS=(1e1 0.1 1e1 0.3 0.0 0.0)
### DOMAIN HYPER-PARAMETERS ###
###############################
GRID_RES=256
TEST_GRID_RES=512
NONMNFLD_SAMPLE_TYPE='grid'
NPOINTS=15000
### TRAINING HYPER-PARAMETERS ###
#################################
# NSAMPLES=30000
NSAMPLES=300000
BATCH_SIZE=1
GPU=0
NEPOCHS=0
EVALUATION_EPOCH=0
# LR=1e-5
LR=1e-6
GRAD_CLIP_NORM=10.0

python3 train_surface_reconstruction.py --logdir $LOGDIR$IDENTIFIER --file_name $FILENAME --grid_res $GRID_RES --loss_type $LOSS_TYPE --inter_loss_type 'exp' --num_epochs $NEPOCHS --gpu_idx $GPU --n_samples $NSAMPLES --n_points $NPOINTS  --batch_size $BATCH_SIZE --lr $LR --nonmnfld_sample_type ${NONMNFLD_SAMPLE_TYPE} --dataset_path $DATASET_PATH --decoder_n_hidden_layers $LAYERS --decoder_hidden_dim $DECODER_HIDDEN_DIM --div_decay $DIVDECAY  --div_decay_params ${DECAY_PARAMS[@]} --div_type ${DIV_TYPE} --init_type ${INIT_TYPE} --nl ${NL} --track_weights ${TRACK_WEIGHTS} --sphere_init_params ${SPHERE_INIT_PARAMS[@]} --loss_weights ${LOSS_WEIGHTS[@]} --grad_clip_norm ${GRAD_CLIP_NORM[@]}
python3 test_surface_reconstruction.py --logdir $LOGDIR$IDENTIFIER --file_name $FILENAME --export_mesh 1 --dataset_path $DATASET_PATH --epoch_n $EVALUATION_EPOCH --grid_res $TEST_GRID_RES --gpu_idx $GPU
