echo "Starting Camera Residual Transformer Training..."

dataset_name="realestate10k_rotmat"

EXPERIMENT_NAME="rtrans_overfit50_t5cond"
TENSORBOARD_DIR="./log/res/${dataset_name}/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="./checkpoints/${dataset_name}/${EXPERIMENT_NAME}"

echo "Logging Configuration:"
echo "- Experiment name: ${EXPERIMENT_NAME}"
echo "- TensorBoard logs: ${TENSORBOARD_DIR}"
echo "- Model checkpoints: ${CHECKPOINT_DIR}"
echo ""

echo "To monitor training in real-time (from another terminal):"
echo "  cd /home/haozhe/CamTraj/momask-codes"
echo "  conda activate momask"
echo "  tensorboard --logdir=${TENSORBOARD_DIR} --port=6006"
echo "  # Then access via browser at: http://localhost:6006"
echo ""

CUDA_VISIBLE_DEVICES=3 python train_res_transformer.py \
    --name ${EXPERIMENT_NAME} \
    --gpu_id 0 \
    --dataset_name ${dataset_name} \
    --data_root ./dataset/RealEstate10K_rotmat1_overfit50 \
    --batch_size 8 \
    --vq_name rvq_window64_overfit50 \
    --conditioning_mode t5 \
    --cond_drop_prob 0.0 \
    --latent_dim 384 \
    --ff_size 1024 \
    --n_layers 8 \
    --n_heads 8 \
    --dropout 0.1 \
    --max_epoch 400 \
    --lr 5e-5 \
    # --share_weight \
    # --is_continue \
    # --keyframe_arch resnet18 \
    # --use_frames \

echo "Residual Transformer Training completed!"
