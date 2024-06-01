#!/bin/bash

NUMA=-1
OOD_BINS=15
OOD_THRESH=0.8
SAM_MODEL='h'
OUTPUT_FOLDER='PINEAPPLES2_5mts_nosplits_SAM_ALONE'
MODEL_NAME='xcit_nano_12_p8_224.fb_dist_in1k'
NUM_CLASSES=0
DATASET_PATH='./pineapples_5m'
DEVICE='cuda'
DATASET='coco2017'
METHOD='samAlone'
USE_SAM_EMBEDDINGS=0
BATCH_SIZE=4
BATCH_SIZE_LABELED=1
BATCH_SIZE_UNLABELED=10
BATCH_SIZE_VALIDATION=1
OOD_UNLABELED_SAMPLES=100
TRAIN_NUM=1
VAL_NUM=1
SEEDS_NUMBERS=('0' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15')
PROPOSAL_MODELS=('fastsam' 'mobilesam' 'sam')
for SEED in "${SEEDS_NUMBERS[@]}" 
do
    for PROPOSAL_MODEL in "${PROPOSAL_MODELS[@]}" 
    do
        python methods.py \
            --root ${DATASET_PATH} \
            --num-classes ${NUM_CLASSES} \
            --load-pretrained \
            --timm-model ${MODEL_NAME} \
            --dataset ${DATASET} \
            --batch-size ${BATCH_SIZE} \
            --batch-size-labeled ${TRAIN_NUM} \
            --batch-size-unlabeled ${BATCH_SIZE_UNLABELED} \
            --batch-size-validation ${BATCH_SIZE_VALIDATION} \
            --ood-labeled-samples ${TRAIN_NUM} \
            --ood-unlabeled-samples ${OOD_UNLABELED_SAMPLES} \
            --ood-validation-samples ${VAL_NUM} \
            --numa ${NUMA} \
            --output-folder ${OUTPUT_FOLDER} \
            --seed ${SEED} \
            --sam-model ${SAM_MODEL} \
            --ood-thresh ${OOD_THRESH} \
            --ood-histogram-bins ${OOD_BINS} \
            --use-sam-embeddings ${USE_SAM_EMBEDDINGS} \
            --method ${METHOD} \
            --device ${DEVICE} \
            --sam-proposal ${PROPOSAL_MODEL} 
        sleep 5
    done
done