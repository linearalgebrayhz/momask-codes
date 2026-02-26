#!/bin/bash
LOG_DIR="./log"

echo "Starting Camera RVQ Training..."
# echo "GPU Info:"
# nvidia-smi

DATASET_NAME="realestate10k_rotmat"

# overfitting experiment on one sample
EXPERIMENT_NAME="rvq_window64_overfit50"
TENSORBOARD_DIR="./log/vq/${DATASET_NAME}/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="./checkpoints/${DATASET_NAME}/${EXPERIMENT_NAME}"

echo ""
echo "Logging Configuration:"
echo "- Experiment name: ${EXPERIMENT_NAME}"
echo "- TensorBoard logs: ${TENSORBOARD_DIR}"
echo "- Model checkpoints: ${CHECKPOINT_DIR}"
echo "- Job logs: ${LOG_DIR}"
echo ""
echo "To monitor training in real-time (from another terminal):"
echo "  cd /home/haozhe/CamTraj/momask-codes/"
echo "  conda activate momask"
echo "  tensorboard --logdir=${TENSORBOARD_DIR} --port=6006"
echo "  # Then access via browser at: http://localhost:6006"
echo ""

# overfitting experiment on one sample
CUDA_VISIBLE_DEVICES=3 python train_vq.py \
    --name ${EXPERIMENT_NAME} \
    --gpu_id 0 \
    --dataset_name ${DATASET_NAME} \
    --data_root ./dataset/RealEstate10K_rotmat1_overfit50 \
    --batch_size 32 \
    --num_quantizers 8 \
    --max_epoch 400 \
    --quantize_dropout_prob 0.1 \
    --gamma 0.8 \
    --lr 1e-4 \
    --warm_up_iter 500 \
    --commit 0.02 \
    --loss_vel 1.0 \
    --recons_loss l1_smooth \
    --log_every 50 \
    --log_codebook_usage \
    --log_gradients \
    --window_size 64 \
    --loss_smoothness 0.0 \
    --num_vis_samples 2 \
    # --is_continue \
    # --loss_orthogonality 0.1 \
    # --code_dim 256 \
    # --nb_code 384 \
    # --is_continue \
    # --eval_on \


echo "RVQ Training completed!"

# Training logs and TensorBoard files will be saved to:
# - TensorBoard logs: ./log/vq/${DATASET_NAME}/${EXPERIMENT_NAME}/
# - Model checkpoints: ./checkpoints/${DATASET_NAME}/${EXPERIMENT_NAME}/
# - Training animations: ./checkpoints/${DATASET_NAME}/${EXPERIMENT_NAME}/animation/

echo "To monitor training progress with TensorBoard:"
echo "tensorboard --logdir=./log/vq/${DATASET_NAME}/${EXPERIMENT_NAME}/"
echo ""
echo "Log files locations:"
echo "- TensorBoard: ./log/vq/${DATASET_NAME}/${EXPERIMENT_NAME}/"
echo "- Checkpoints: ./checkpoints/${DATASET_NAME}/${EXPERIMENT_NAME}/"
echo "- Job logs: ${LOG_DIR}/"
