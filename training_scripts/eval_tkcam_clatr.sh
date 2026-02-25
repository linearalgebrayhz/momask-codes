#!/bin/bash

# Evaluation script for TKCAM models using CLaTr metrics
# Models: rvq_realestate10k_6_final_1, mtrans_realestate10k_6_final_1, rtrans_realestate10k_6_final_1

# Configuration
DATASET_NAME="realestate10k_6"  # Dataset name (maps to RealEstate10K_6feat directory)
VQ_NAME="rvq_realestate10k_6_final_1"
TRANS_NAME="mtrans_realestate10k_6_final_1"
RES_NAME="rtrans_realestate10k_6_final_1"

GPU_ID=0
WHICH_EPOCH="net_best_acc.tar"  # Use best accuracy checkpoint (FID is from inaccurate eval wrapper)

# CLaTr checkpoint paths
CLATR_TEXT_CKPT="/home/haozhe/CamTraj/CLaTr/lightning_logs/version_27/checkpoints/last.ckpt"
CLATR_FRAME_CKPT="/home/haozhe/CamTraj/CLaTr/lightning_logs/version_28/checkpoints/last.ckpt"

# Number of samples to evaluate (None = all test set)
NUM_SAMPLES=""  # Leave empty for all, or set to number like 1000

# Generate extension name for logging
EXT="clatr_eval_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "TKCAM CLaTr Evaluation"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "VQ Model: $VQ_NAME"
echo "Transformer: $TRANS_NAME"
echo "Residual: $RES_NAME"
echo "CLaTr Text Checkpoint: $CLATR_TEXT_CKPT"
echo "Checkpoint to evaluate: $WHICH_EPOCH"
echo "GPU: $GPU_ID"
echo "=========================================="

cd /home/haozhe/CamTraj/momask-codes

# Run evaluation with transformer + residual
CUDA_VISIBLE_DEVICES=5 python eval_tkcam_clatr.py \
    --name $TRANS_NAME \
    --dataset_name $DATASET_NAME \
    --gpu_id $GPU_ID \
    --which_epoch $WHICH_EPOCH \
    --res_name $RES_NAME \
    --load_frames \
    --clatr_frame_ckpt /home/haozhe/CamTraj/CLaTr/lightning_logs/version_28/checkpoints/last.ckpt \
    --clatr_text_ckpt $CLATR_TEXT_CKPT \
    # --debug \
    # --debug_samples 100 \
    --ext $EXT \
    ${NUM_SAMPLES:+--num_eval_samples $NUM_SAMPLES}

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "Check results in: checkpoints/$DATASET_NAME/$TRANS_NAME/eval_clatr/"
echo "=========================================="
