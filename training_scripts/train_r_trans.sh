echo "Starting Camera Residual Transformer Training..."

EXPERIMENT_NAME="rtrans_realestate10k_6_resnet18"
TENSORBOARD_DIR="./log/vq/realestate10k_6/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="./checkpoints/realestate10k_6/${EXPERIMENT_NAME}"

echo "Logging Configuration:"
echo "- Experiment name: ${EXPERIMENT_NAME}"
echo "- TensorBoard logs: ${TENSORBOARD_DIR}"
echo "- Model checkpoints: ${CHECKPOINT_DIR}"
echo ""

echo "To monitor training in real-time (from another terminal):"
echo "  cd /home/hy4522/Research/momask-codes"
echo "  conda activate momask"
echo "  tensorboard --logdir=${TENSORBOARD_DIR} --port=6006"
echo "  # Then access via browser at: http://localhost:6006"
echo ""

CUDA_VISIBLE_DEVICES=3 python train_res_transformer.py \
    --name ${EXPERIMENT_NAME} \
    --gpu_id 0 \
    --dataset_name realestate10k_6 \
    --batch_size 16 \
    --vq_name rvq_realestate10k_6_70K \
    --cond_drop_prob 0.1 \
    --share_weight \
    --latent_dim 384 \
    --ff_size 1024 \
    --n_layers 8 \
    --n_heads 8 \
    --dropout 0.1 \
    --max_epoch 400 \
    --lr 5e-5 \
    --keyframe_arch resnet18 \
    --use_frames

echo "Residual Transformer Training completed!"
