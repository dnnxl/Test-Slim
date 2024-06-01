#!/bin/bash

NUMA=-1
OOD_BINS=15
OOD_THRESH=0.8
SAM_MODEL='h'
OUTPUT_FOLDER='PINEAPPLES2_5mts_nosplits_100224'
MODEL_NAME='xcit_nano_12_p8_224.fb_dist_in1k'
NUM_CLASSES=0
DATASET_PATH='./pineapples_5m'
DEVICE='cuda'
DATASET='coco2017'
METHOD='fewshotMahalanobis'
PROPOSAL_MODEL='fastsam'
USE_SAM_EMBEDDINGS=0
BATCH_SIZE=4
BATCH_SIZE_LABELED=1
BATCH_SIZE_UNLABELED=10
BATCH_SIZE_VALIDATION=1
OOD_UNLABELED_SAMPLES=100
SEEDS_NUMBERS=('0' '1' '5' '7' '11')
TRAINING_VALIDATION_NUMBER=('1,1' '2,1' '3,2' '4,2' '5,2')
BETAS_NUMBERS=('0' '1' '2')
MAHALANOBIS_MODE=('normal' 'regularization')
DIMENSION_VALUES=('4' '8' '16' '32' '64')
DIMENSION_REDUCTION=('none' 'svd')
for TRAIN_VALID_NUM in "${TRAINING_VALIDATION_NUMBER[@]}" 
do
    IFS=',' read -r TRAIN_NUM VAL_NUM <<< "${TRAIN_VALID_NUM}"
    for SEED in "${SEEDS_NUMBERS[@]}" 
    do
        for MAHALANOBIS in "${MAHALANOBIS_MODE[@]}" 
        do
            BETAS_NUMBERS=('0' '1' '2')
            if [ "${MAHALANOBIS}" == "normal" ]; then
                BETAS_NUMBERS=('0')
            fi
            for BETA in "${BETAS_NUMBERS[@]}" 
            do
                for DIM_RED in "${DIMENSION_REDUCTION[@]}" 
                do
                    DIMENSION_VALUES=('4' '8' '16' '32' '64')
                    if [ "${DIM_RED}" == "none" ]; then
                        DIMENSION_VALUES=('0')
                    fi
                    for DIM in "${DIMENSION_VALUES[@]}" 
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
                            --sam-proposal ${PROPOSAL_MODEL} \
                            --dim-red ${DIM_RED} \
                            --n-components ${DIM} \
                            --mahalanobis ${MAHALANOBIS} \
                            --beta ${BETA}
                        sleep 5
                    done 
                done
            done
        done
    done
done