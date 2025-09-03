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

# Run the training command
python -m train \
    experiment=hg38/genomic_benchmark \
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
    dataset.dataset_name="${TASK}" \
    dataset.train_val_split_seed=${seed} \
    dataset.batch_size=${BATCH_SIZE} \
    dataset.rc_aug="${RC_AUG}" \
    +dataset.conjoin_train=false \
    +dataset.conjoin_test="${CONJOIN_TEST}" \
    model="caduceus" \
    model._name_="dna_embedding_caduceus" \
    +model.conjoin_test="${CONJOIN_TEST}" \
    +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
    +decoder.conjoin_test="${CONJOIN_TEST}" \
    optimizer.lr="${LR}" \
    trainer.max_epochs=10 \
    wandb.group="downstream/gb_cv5" \
    wandb.job_type="${TASK}" \
    wandb.name="${WANDB_NAME}" \
    wandb.id="gb_cv5_${TASK}_${WANDB_NAME}_seed-${seed}" \
    +wandb.tags=\["seed-${seed}"\] \
    hydra.run.dir="${HYDRA_RUN_DIR}"
