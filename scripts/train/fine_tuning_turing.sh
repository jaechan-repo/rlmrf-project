
MODEL_DIR="../../models/llava-v1.5-7b"
OUTPUT_DIR="../../ckpt/iter1/reward_image/llava-v1.5-7b-lora-turing"
DATA_PATH="../../data_in/iter1/sft_image.json"
INCLUDES_NON_VQA=False
EPOCHS=5

TF32=False
BF16=False
FP16=True
TRAINER="train.py"

deepspeed $TRAINER \
    --model_name_or_path $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --data_path $DATA_PATH \
    --image_folder ../../data \
    --group_by_modality_length $INCLUDES_NON_VQA \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./zero3.json \
    --version v1 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 $BF16 \
    --fp16 $FP16 \
    --tf32 $TF32 \
    --num_train_epochs $EPOCHS \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb