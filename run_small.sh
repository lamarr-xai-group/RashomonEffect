#!/bin/sh

# hacky: max epochs needs to be high to not collide with training-length

META_PARAMS="--directory ./data/TrainOnly --track-explanations 0"
echo "Using the following params for all: $META_PARAMS"

PARAMS="--dataset beans --training-length 800 --modelparams [16,16,16] --batch-sizes [32,1050] --data-seed 11880 --num-runs 10 --max-epochs 1000"
#echo "Using the following params for  beans: $PARAMS"
#
echo "starting beans"
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS  --model-seed 666 &
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS --model-seed 745616 &
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS --model-seed 615645 &
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS --model-seed 154665 &
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS --model-seed 532465 &
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS --model-seed 724357 &
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS --model-seed 268423 &
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS --model-seed 964732 &
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS --model-seed 251332 &
python 1_train_models_collect_explanations.py $META_PARAMS $PARAMS --model-seed 672532 &

PARAMS=" --data-seed 11880 --num-runs 100 --max-epochs 1000"
echo "params for ionosphere, breast cancer: $PARAMS"
## breast cancer
echo "starting breast cancer"
python 1_train_models_collect_explanations.py $META_PARAMS --dataset breastcancer --training-length 200 --modelparams [16] --batch-sizes [16,300] $PARAMS &
#
## ionosphere
echo "starting ionosphere"
python 1_train_models_collect_explanations.py $META_PARAMS --dataset ionosphere --training-length 400 --modelparams [8] --batch-sizes [16,300] $PARAMS &
