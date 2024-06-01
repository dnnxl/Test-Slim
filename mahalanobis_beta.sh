GPU_NUM=-1
export CUDA_VISIBLE_DEVICES=${GPU_NUM}
DATASET_NAME="PINEAPPLES2_5mts_nosplits"
ROOT="./pineapples_5m" 
NUM_CLASSES=0
DATASET="coco2017"
BATCH_SIZE_LABELED=1
BATCH_SIZE_UNLABELED=1
OOD_LABELED_SAMPLES=1
OOD_UNLABELED_SAMPLES=100
NEW_SAMPLE_SIZE=256
OOD_THRESH=0.8
OOD_BINS=15

DEVICE="cuda"

LABELED_NUMBER=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10')
VALIDATION_NUMBER=('1' '1' '2' '2' '2' '3' '3' '3' '4' '4')
SEEDS_NUMBERS=('0' '1' '2' '3' '4')
BETA=('0' '1' '2' '3' '4')
DIMENSION=('4' '8' '16' '32' '64')
MAHALANOBIS_MODES=('normal' 'regularization')
MODELS=('xcit_nano_12_p8_224.fb_dist_in1k' 'swinv2_base_window8_256.ms_in1k' 'tf_efficientnet_l2.ns_jft_in1k_475' 'vit_base_patch16_clip_224.openai_ft_in1k')
RED_DIM=('none' 'svd')

for curr_label_num in "${LABELED_NUMBER[@]}"
do
    for val_num in "${VALIDATION_NUMBER[@]}"
    do
        for seed in "${SEEDS_NUMBERS[@]}"
        do
            for bet in "${BETA[@]}"
            do
                for mode in "${MAHALANOBIS_MODES[@]}"
                do
                    for dim in "${DIMENSION[@]}"
                    do
                        for red_dim in "${RED_DIM[@]}"
                        do
                            python methods.py \
                                --root ${ROOT} \
                                --num-classes ${NUM_CLASSES} \
                                --load-pretrained \
                                --timm-model ${curr_model} \
                                --loss "mae" \
                                --optim "sgd" \
                                --val-freq "1" \
                                --use-semi-split \
                                --semi-percentage "1.0" \
                                --dataset ${DATASET} \
                                --batch-size "4" \
                                --batch-size-val "1" \
                                --batch-size-labeled ${BATCH_SIZE_LABELED} \
                                --batch-size-unlabeled ${BATCH_SIZE_UNLABELED} \
                                --batch-size-validation "1" \
                                --ood-labeled-samples ${OOD_LABELED_SAMPLES} \
                                --ood-unlabeled-samples ${OOD_UNLABELED_SAMPLES} \
                                --ood-validation-samples ${OOD_UNLABELED_SAMPLES} \
                                --aug-method "rand_augmentation" \
                                --img-resolution "1024" \
                                --new-sample-size ${NEW_SAMPLE_SIZE} \
                                --numa ${GPU_NUM} \
                                --output-folder ${DATASET_NAME} \
                                --run-name "temp" \
                                --seed ${seed} \
                                --sam-model "h" \
                                --ood-thresh "0.8" \
                                --ood-histogram-bins "15" \
                                --use-sam-embeddings "0" \
                                --method "fewshotMahalanobis" \
                                --device ${DEVICE} \
                                --sam-proposal "fastsam" \
                                --dim-red ${red_dim} \
                                --n-components ${dim} \
                                --beta ${bet} \
                            sleep 10
                        done
                    done
                done
            done
        done
    done
done