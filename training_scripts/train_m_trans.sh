echo "Starting Camera Masked Transformer Training..."

DATASET_NAME="realestate10k_rotmat"

EXPERIMENT_NAME="mtrans_3k_clip_crossattn"
TENSORBOARD_DIR="./log/t2m/${DATASET_NAME}/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="./checkpoints/${DATASET_NAME}/${EXPERIMENT_NAME}"


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

CUDA_VISIBLE_DEVICES=3 python train_t2m_transformer.py \
    --name ${EXPERIMENT_NAME} \
    --gpu_id 0 \
    --dataset_name ${DATASET_NAME} \
    --data_root ./dataset/RealEstate10K_rotmat_3k \
    --batch_size 32 \
    --vq_name rvq_window64_3k \
    --conditioning_mode clip \
    --cond_drop_prob 0.1 \
    --latent_dim 384 \
    --ff_size 1024 \
    --n_layers 8 \
    --n_heads 8 \
    --dropout 0.1 \
    --max_epoch 600 \
    --lr 5e-5 \
    # --is_continue \
    # --finetune_clip \
    # --finetune_clip_layers 2 \
    # --keyframe_arch resnet18 \
    # --use_frames \
    # --is_continue \

echo "Masked Transformer Training completed!"