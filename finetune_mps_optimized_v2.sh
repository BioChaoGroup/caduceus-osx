#!/bin/bash

# MPS-optimized fine-tuning script for Caduceus model - Version 2
# Fixed multi-process data loading issues for better CPU-GPU balance

# Set environment variables for fine-tuning
export TASK="dummy_mouse_enhancers_ensembl"
export BATCH_SIZE="256"  # Optimized for MPS memory
export RC_AUG="false"
export CONJOIN_TEST="false"
export CONJOIN_TRAIN_DECODER="false"
export LR="6e-4"  # Learning rate optimized for fine-tuning
export WANDB_NAME="mps_finetune_run_v2"
export HYDRA_RUN_DIR="./outputs/mps_finetune_v2_$(date +%Y%m%d_%H%M%S)"
export seed="5"

# MPS-specific environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Proxy settings (if needed)
export https_proxy=http://192.168.31.33:7897 http_proxy=http://192.168.31.33:7897 all_proxy=socks5://192.168.31.33:7897

echo "Starting MPS-optimized fine-tuning (v2)..."
echo "Task: ${TASK}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "Seed: ${seed}"
echo "Output Directory: ${HYDRA_RUN_DIR}"

# Create output directory
mkdir -p ${HYDRA_RUN_DIR}

# Run the fine-tuning command with optimized MPS settings
# Using fewer workers to avoid multi-process issues on MPS
mamba run -n caduceus_env python -m train \
    experiment=hg38/genomic_benchmark_hf_caduceus \
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000 \
    dataset.dataset_name="${TASK}" \
    dataset.train_val_split_seed=${seed} \
    dataset.batch_size=${BATCH_SIZE} \
    dataset.rc_aug="${RC_AUG}" \
    +dataset.num_workers=16 \
    +dataset.pin_memory=true \
    +dataset.drop_last=true \
    +dataset.conjoin_train=false \
    +dataset.conjoin_test="${CONJOIN_TEST}" \
    +model.conjoin_test="${CONJOIN_TEST}" \
    +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
    +decoder.conjoin_test="${CONJOIN_TEST}" \
    optimizer.lr="${LR}" \
    trainer.max_epochs=10 \
    trainer.accelerator=mps \
    trainer.devices=1 \
    trainer.precision=32 \
    trainer.gradient_clip_val=1.0 \
    +trainer.gradient_clip_algorithm=norm \
    trainer.accumulate_grad_batches=1 \
    +trainer.enable_progress_bar=true \
    +trainer.enable_checkpointing=true \
    +trainer.enable_model_summary=false \
    trainer.log_every_n_steps=1 \
    +trainer.deterministic=false \
    +trainer.benchmark=true \
    +train.compile_model=false \
    +train.mps_optimizations=true \
    wandb.mode=disabled \
    wandb.group="downstream/gb_cv5_mps_finetune" \
    wandb.job_type="${TASK}" \
    wandb.name="${WANDB_NAME}" \
    wandb.id="gb_cv5_${TASK}_${WANDB_NAME}_seed-${seed}" \
    +wandb.tags=\["seed-${seed}","mps-optimized","apple-silicon","finetune"\] \
    hydra.run.dir="${HYDRA_RUN_DIR}"

echo "Fine-tuning completed!"
echo "Checkpoints and logs saved to: ${HYDRA_RUN_DIR}"
