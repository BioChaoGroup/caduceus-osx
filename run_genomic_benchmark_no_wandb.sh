#!/bin/bash

# Set environment variables for genomic benchmark training
export TASK="dummy_mouse_enhancers_ensembl"
export BATCH_SIZE="128"
export RC_AUG="false"
export CONJOIN_TEST="false"
export CONJOIN_TRAIN_DECODER="false"
export LR="6e-4"
export WANDB_NAME="test_run"
export HYDRA_RUN_DIR="./outputs/test_run"
export seed="5"

# Run the training command with wandb disabled
export https_proxy=http://192.168.31.33:7897 http_proxy=http://192.168.31.33:7897 all_proxy=socks5://192.168.31.33:7897

# Use MPS accelerator for Apple Silicon Macs
python -m train \
    experiment=hg38/genomic_benchmark_hf_caduceus \
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
    dataset.dataset_name="${TASK}" \
    dataset.train_val_split_seed=${seed} \
    dataset.batch_size=${BATCH_SIZE} \
    dataset.rc_aug="${RC_AUG}" \
    +dataset.conjoin_train=false \
    +dataset.conjoin_test="${CONJOIN_TEST}" \
    +model.conjoin_test="${CONJOIN_TEST}" \
    +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
    +decoder.conjoin_test="${CONJOIN_TEST}" \
    optimizer.lr="${LR}" \
    trainer.max_epochs=10 \
    trainer.accelerator=mps \
    trainer.devices=1 \
    wandb.mode=disabled \
    wandb.group="downstream/gb_cv5" \
    wandb.job_type="${TASK}" \
    wandb.name="${WANDB_NAME}" \
    wandb.id="gb_cv5_${TASK}_${WANDB_NAME}_seed-${seed}" \
    +wandb.tags=\["seed-${seed}"\] \
    hydra.run.dir="${HYDRA_RUN_DIR}"
