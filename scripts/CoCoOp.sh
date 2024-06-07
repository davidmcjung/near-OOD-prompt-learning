#!/bin/bash

# custom config
DATA=./data

for SEED in 1 2 3
do
    echo "Seed: $SEED"
    for SHOTS in 16 8 4 2 1
    do
        echo "Shots: $SHOTS"
        for DATASET in "stanford_cars" "eurosat" "dtd" "caltech101" "fgvc_aircraft" "food101" "oxford_flowers" "oxford_pets" "ucf101" "sun397" "imagenet"
        do
            echo "Dataset: $DATASET"
            # Training
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer CoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
            --ood_method energy max_logit mcm \
            --output-dir output/${DATASET}/shots_${SHOTS}/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base 

            # Relative scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer CoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
            --model-dir output/${DATASET}/shots_${SHOTS}/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            --output-dir output/${DATASET}/shots_${SHOTS}/CoCoOp_relative/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            --load-epoch 10 \
            --ood_method energy max_logit mcm \
            --eval-only \
            --empty_cls_prompt \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base 

        done
    done
done

for SEED in 1 2 3
do
    echo "Seed: $SEED"
    for SHOTS in 16 8 4 2 1
    do
        echo "Shots: $SHOTS"
        for DATASET in "cifar10" "cifar100"
        do
            echo "Dataset: $DATASET"
            # Training
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer CoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
            --ood_method energy max_logit mcm \
            --output-dir output/${DATASET}/shots_${SHOTS}/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES all 

            # Relative scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer CoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
            --model-dir output/${DATASET}/shots_${SHOTS}/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            --output-dir output/${DATASET}/shots_${SHOTS}/CoCoOp_relative/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            --load-epoch 10 \
            --ood_method energy max_logit mcm \
            --eval-only \
            --empty_cls_prompt \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES all 

        done
    done
done

for SEED in 1 2 3
do
    echo "Seed: $SEED"
    for SHOTS in 16 8 4 2 1
    do
        echo "Shots: $SHOTS"
        for DATASET in "imagenet" 
        do
            echo "Dataset: $DATASET" 
            # OOD with original scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer CoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
            --model-dir output/${DATASET}/shots_${SHOTS}/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            --output-dir output/${DATASET}/shots_${SHOTS}/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            --load-epoch 10 \
            --ood_method energy max_logit mcm \
            --eval-only \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base 

            # OOD with relative scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer CoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml \
            --model-dir output/${DATASET}/shots_${SHOTS}/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            --output-dir output/${DATASET}/shots_${SHOTS}/CoCoOp_relative/vit_b16_c4_ep10_batch1_ctxv1/seed${SEED} \
            --load-epoch 10 \
            --ood_method energy max_logit mcm \
            --eval-only \
            --empty_cls_prompt \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base 

        done
    done
done