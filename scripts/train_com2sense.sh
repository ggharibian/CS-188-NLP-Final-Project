TASK_NAME="com2sense"
DATA_DIR="datasets/com2sense"
# MODEL_TYPE="bert-base-cased"
# MODEL_TYPE="xlm-roberta-base"
# MODEL_TYPE="roberta-base"
# MODEL_TYPE="outputs/com2sense/ckpts/checkpoint-best-bert"
MODEL_TYPE="outputs/com2sense/ckpts/checkpoint-best-roberta"


python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_not_load_optimizer \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --per_gpu_train_batch_size 20 \
  --per_gpu_eval_batch_size 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 100.0 \
  --max_seq_length 128 \
  --output_dir "${TASK_NAME}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --overwrite_output_dir \
  --save_steps 20 \
  --logging_steps 5 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "binary" \
  --iters_to_eval 20 40 \
  --overwrite_output_dir \
  --eval_split "dev" \
  # --max_eval_steps 1000 \
