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
            --trainer LoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/LoCoOp/vit_b16_ep50.yaml \
            --ood_method energy max_logit mcm \
            --output-dir output_edited/${DATASET}/shots_${SHOTS}/LoCoOp/vit_b16_ep50/seed${SEED} \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.LOCOOP.N_CTX 16 \
            TRAINER.LOCOOP.CSC False \
            TRAINER.LOCOOP.CLASS_TOKEN_POSITION end \
            DATASET.SUBSAMPLE_CLASSES base

            # Relative scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer LoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/LoCoOp/vit_b16_ep50.yaml \
            --output-dir output/${DATASET}/shots_${SHOTS}/LoCoOp_relative/vit_b16_ep50/seed${SEED} \
            --ood_method energy max_logit mcm \
            --eval-only \
            --model-dir output/${DATASET}/shots_${SHOTS}/LoCoOp/vit_b16_ep50/seed${SEED} \
            --load-epoch 50 \
            --empty_cls_prompt \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.LOCOOP.N_CTX 16 \
            TRAINER.LOCOOP.CSC False \
            TRAINER.LOCOOP.CLASS_TOKEN_POSITION end \
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
            --trainer LoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/LoCoOp/vit_b16_ep50.yaml \
            --ood_method energy max_logit mcm \
            --output-dir output_edited/${DATASET}/shots_${SHOTS}/LoCoOp/vit_b16_ep50/seed${SEED} \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.LOCOOP.N_CTX 16 \
            TRAINER.LOCOOP.CSC False \
            TRAINER.LOCOOP.CLASS_TOKEN_POSITION end \
            DATASET.SUBSAMPLE_CLASSES all

            # Relative scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer LoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/LoCoOp/vit_b16_ep50.yaml \
            --output-dir output/${DATASET}/shots_${SHOTS}/LoCoOp_relative/vit_b16_ep50/seed${SEED} \
            --ood_method energy max_logit mcm \
            --eval-only \
            --model-dir output/${DATASET}/shots_${SHOTS}/LoCoOp/vit_b16_ep50/seed${SEED} \
            --load-epoch 50 \
            --empty_cls_prompt \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.LOCOOP.N_CTX 16 \
            TRAINER.LOCOOP.CSC False \
            TRAINER.LOCOOP.CLASS_TOKEN_POSITION end \
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
            --trainer LoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/LoCoOp/vit_b16_ep50.yaml \
            --output-dir output/${DATASET}/shots_${SHOTS}/LoCoOp/vit_b16_ep50/seed${SEED} \
            --ood_method energy max_logit mcm \
            --eval-only \
            --model-dir output/${DATASET}/shots_${SHOTS}/LoCoOp/vit_b16_ep50/seed${SEED} \
            --load-epoch 50 \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.LOCOOP.N_CTX 16 \
            TRAINER.LOCOOP.CSC False \
            TRAINER.LOCOOP.CLASS_TOKEN_POSITION end \
            DATASET.SUBSAMPLE_CLASSES base
        
            # OOD with relative scores
            python maple_based/train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer LoCoOp \
            --dataset-config-file maple_based/configs/datasets/${DATASET}.yaml \
            --config-file maple_based/configs/trainers/LoCoOp/vit_b16_ep50.yaml \
            --output-dir output/${DATASET}/shots_${SHOTS}/LoCoOp_relative/vit_b16_ep50/seed${SEED} \
            --ood_method energy max_logit mcm \
            --eval-only \
            --model-dir output/${DATASET}/shots_${SHOTS}/LoCoOp/vit_b16_ep50/seed${SEED} \
            --load-epoch 50 \
            --empty_cls_prompt \
            --no_eval_near_ood \
            --eval_ood \
            DATASET.NUM_SHOTS ${SHOTS} \
            TRAINER.LOCOOP.N_CTX 16 \
            TRAINER.LOCOOP.CSC False \
            TRAINER.LOCOOP.CLASS_TOKEN_POSITION end \
            DATASET.SUBSAMPLE_CLASSES base

        done
    done
done