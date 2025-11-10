#!/bin/bash
LOG_DIR="./log"

echo "Starting Camera RVQ Training..."
# echo "GPU Info:"
# nvidia-smi

# overfitting experiment on one sample
EXPERIMENT_NAME="rvq_realestate10k_6_eval"
TENSORBOARD_DIR="./log/vq/realestate10k_6/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="./checkpoints/realestate10k_6/${EXPERIMENT_NAME}"

echo ""
echo "Logging Configuration:"
echo "- Experiment name: ${EXPERIMENT_NAME}"
echo "- TensorBoard logs: ${TENSORBOARD_DIR}"
echo "- Model checkpoints: ${CHECKPOINT_DIR}"
echo "- Job logs: ${LOG_DIR}"
echo ""
echo "To monitor training in real-time (from another terminal):"
echo "  cd /home/hy4522/Research/momask-codes"
echo "  conda activate momask"
echo "  tensorboard --logdir=${TENSORBOARD_DIR} --port=6006"
echo "  # Then access via browser at: http://localhost:6006"
echo ""

# overfitting experiment on one sample
CUDA_VISIBLE_DEVICES=2 python train_vq.py \
    --name ${EXPERIMENT_NAME} \
    --gpu_id 0 \
    --dataset_name realestate10k_6 \
    --batch_size 256 \
    --num_quantizers 8 \
    --max_epoch 20 \
    --quantize_dropout_prob 0.1 \
    --gamma 0.8 \
    --lr 5e-5 \
    --warm_up_iter 5000 \
    --commit 0.02 \
    --loss_vel 1.0 \
    --recons_loss l1_smooth \
    --log_every 100 \
    --log_codebook_usage \
    --log_gradients \
    --window_size 64 \
    --eval_on \
    # --is_continue

echo "RVQ Training completed!"

# Training logs and TensorBoard files will be saved to:
# - TensorBoard logs: ./log/vq/cam/rvq_camera_trajectory_50/
# - Model checkpoints: ./checkpoints/cam/rvq_camera_trajectory_50/
# - Training animations: ./checkpoints/cam/rvq_camera_trajectory_50/animation/

echo "To monitor training progress with TensorBoard:"
echo "tensorboard --logdir=./log/vq/cam/${EXPERIMENT_NAME}/"
echo ""
echo "Log files locations:"
echo "- TensorBoard: ./log/vq/cam/${EXPERIMENT_NAME}/"
echo "- Checkpoints: ./checkpoints/cam/${EXPERIMENT_NAME}/"
echo "- Job logs: ${LOG_DIR}/"
