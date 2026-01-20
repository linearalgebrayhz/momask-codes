echo "Starting Camera Masked Transformer Training..."

dataset_name="realestate10k_12"

EXPERIMENT_NAME="mtrans_new"
TENSORBOARD_DIR="./log/t2m/${dataset_name}/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="./checkpoints/${dataset_name}/${EXPERIMENT_NAME}"

echo "Logging Configuration:"
echo "- Experiment name: ${EXPERIMENT_NAME}"
echo "- TensorBoard logs: ${TENSORBOARD_DIR}"
echo "- Model checkpoints: ${CHECKPOINT_DIR}"
echo ""

echo "To monitor training in real-time (from another terminal):"
echo "  cd /home/haozhe/CamTraj/momask-codes"
echo "  conda activate TKCAM"
echo "  tensorboard --logdir=${TENSORBOARD_DIR} --port=6006"
echo "  # Then access via browser at: http://localhost:6006"
echo ""

CUDA_VISIBLE_DEVICES=1 python train_t2m_transformer.py \
    --name ${EXPERIMENT_NAME} \
    --gpu_id 0 \
    --dataset_name ${dataset_name} \
    --batch_size 64 \
    --vq_name rvq_baseline_window128 \
    --latent_dim 384 \
    --ff_size 1024 \
    --n_layers 8 \
    --n_heads 8 \
    --dropout 0.1 \
    --cond_drop_prob 0.1 \
    --max_epoch 200 \
    --lr 5e-5 \
    --warm_up_iter 2000
    # Uncomment to enable extra features (all disabled by default):
    # --smooth_loss_weight 0.02 \
    # --direction_loss_weight 0.05 \
    # --finetune_clip \
    # --finetune_clip_layers 2 \
    # --use_frames \
    # --keyframe_arch resnet18 \
    # --is_continue \

echo "Masked Transformer Training completed!"