echo "Starting Camera Masked Transformer Training..."

DATASET_NAME="realestate10k_rotmat"

EXPERIMENT_NAME="mtrans_25k_xframe_r4_prefix_self_attn"
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

CUDA_VISIBLE_DEVICES=4 python train_t2m_transformer.py \
    --name ${EXPERIMENT_NAME} \
    --gpu_id 0 \
    --dataset_name ${DATASET_NAME} \
    --data_root ./dataset/RealEstate10K_rotmat_25k \
    --batch_size 64 \
    --vq_name rvq_window128_25k_4quantizers \
    --conditioning_mode t5 \
    --cond_drop_prob 0.2 \
    --latent_dim 384 \
    --ff_size 1024 \
    --n_layers 4 \
    --n_heads 6 \
    --dropout 0.2 \
    --max_epoch 600 \
    --lr 5e-5 \
    --evaluator_ckpt ./checkpoints/evaluator/clatr_universal_eval_RE10K_ET_GenDoP_9d_ld256_od512/best.pt \
    --vis_vel_integration \
    --use_sparse_frames  \
    --max_sparse_frames 4 \
    --visual_drop_prob 0.2 \
    --condition_fusion prefix_self_attn
    # --use_first_frame \
    # --visual_drop_prob 0.2 \



echo "Masked Transformer Training completed!"