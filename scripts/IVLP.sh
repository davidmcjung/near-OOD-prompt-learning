#!/bin/bash

# custom config
DATA=/home/david/tmp/CoOp/data

for SEED in 1 2 3
do
    echo "Seed: $SEED"
    for SHOTS in 16 8 4 2 1
    do
        echo "Shots: $SHOTS"
        for DATASET in "stanford_cars" "eurosat" "dtd" "caltech101" "fgvc_aircraft" "food101" "oxford_flowers" "oxford_pets" "ucf101" "sun397" "imagenet"
        do
            # Training
            echo "Dataset: $DATASET"
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer IVLP \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml \
            --ood_method energy max_logit mcm \
            --output-dir output/${DATASET}/shots_${SHOTS}/IVLP/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base

            # Relative scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer IVLP \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml \
            --model-dir output/${DATASET}/shots_${SHOTS}/IVLP/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            --output-dir output/${DATASET}/shots_${SHOTS}/IVLP_relative/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            --load-epoch 5 \
            --eval-only \
            --ood_method energy max_logit mcm \
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
            # Training
            echo "Dataset: $DATASET"
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer IVLP \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml \
            --ood_method energy max_logit mcm \
            --output-dir output/${DATASET}/shots_${SHOTS}/IVLP/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES all

            # Relative scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer IVLP \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml \
            --model-dir output/${DATASET}/shots_${SHOTS}/IVLP/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            --output-dir output/${DATASET}/shots_${SHOTS}/IVLP_relative/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            --load-epoch 5 \
            --eval-only \
            --ood_method energy max_logit mcm \
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
            # OOD with original scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer IVLP \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml \
            --model-dir output/${DATASET}/shots_${SHOTS}/IVLP/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            --output-dir output/${DATASET}/shots_${SHOTS}/IVLP/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            --load-epoch 5 \
            --eval-only \
            --ood_method energy max_logit mcm \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base

            # OOD with relative scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer IVLP \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml \
            --model-dir output/${DATASET}/shots_${SHOTS}/IVLP/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            --output-dir output/${DATASET}/shots_${SHOTS}/IVLP_relative/vit_b16_c2_ep5_batch4_2+2ctx/seed${SEED} \
            --load-epoch 5 \
            --eval-only \
            --ood_method energy max_logit mcm \
            --empty_cls_prompt \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        done
    done
done