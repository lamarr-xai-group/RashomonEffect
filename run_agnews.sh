#!/bin/sh

echo "start agnews"
PARAMS="--directory ./data/TrainOnly --track-explanations 0 --data-seed 11880 --training-length 6000 --dataset agnews --num-runs 5 --max-epochs 5 --batch-sizes [64,300] --modelparams [128,128] --save-freq -1"
time python 1_train_models_collect_explanations.py --model-seed 664 --gpu-id 0 $PARAMS &
time python 1_train_models_collect_explanations.py --model-seed 666 --gpu-id 1 $PARAMS &
time python 1_train_models_collect_explanations.py --model-seed 668 --gpu-id 2 $PARAMS &
time python 1_train_models_collect_explanations.py --model-seed 999 --gpu-id 3 $PARAMS &
