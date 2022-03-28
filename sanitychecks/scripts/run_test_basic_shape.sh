#!/bin/bash
# run from DiGS/sanitychecks/ , i.e. `./scripts/run_train_test_shape.sh`
DIGS_DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
echo "If $DIGS_DIR is not the correct path for your DiGS repository, set it manually at the variable DIGS_DIR"
cd $DIGS_DIR/sanitychecks/ # To call python scripts correctly

LOGDIR='./log/' #change to your desired log directory
IDENTIFIER='my_experiment'
mkdir -p $LOGDIR
FILE=`basename "$0"`
cp scripts/$FILE $LOGDIR # saves copy of this script so you know the args used

GPU=0
NONMNFLD_SAMPLE_TYPE='grid'
GRID_RES=256
# EPOCHS_N_EVAL=($(seq 9900 -100 0))
# EPOCHS_N_EVAL=(9900)
EPOCHS_N_EVAL=($(seq 0 100 9900))

for SHAPE in   'L' 'snowflake' 'circle'
do
LOGDIR=${LOGDIRNAME}${NONMNFLD_SAMPLE_TYPE}'_sampling_'${GRID_RES}'/'${SHAPE}'/'${IDENTIFIER}'/'
python3 test_basic_shape.py --logdir $LOGDIR --gpu_idx $GPU --epoch_n "${EPOCHS_N_EVAL[@]}"
done