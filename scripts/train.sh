
output_dir="arjudge-v1"

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    --mixed_precision bf16 \
    --num_processes 6 \
    train.py configs/config_full.yaml \
    --data_path="data/train/train.json" \
    --num_train_epochs=2 \
    --output_dir=outputs/$output_dir
