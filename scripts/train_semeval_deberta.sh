DATA_DIR="datasets/semeval_2020_task4"
#MODEL_TYPE="bert-base-cased"
#MODEL_TYPE="roberta-base"
#MODEL_TYPE="microsoft/deberta-base"
MODEL_TYPE="outputs/semeval/ckpts/ckpts/checkpoint-best-deberta"
TASK_NAME="semeval"
OUTPUT_DIR=${TASK_NAME}/ckpts


CUDA_VISIBLE_DEVICES=0 python3 -m trainers.train \
  --model_name_or_path ${MODEL_TYPE} \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --gradient_accumulation_steps 4 \
  --per_gpu_train_batch_size 10 \
  --per_gpu_eval_batch_size 2 \
  --learning_rate 5e-6 \
  --max_steps 4000 \
  --max_seq_length 128 \
  --output_dir "${OUTPUT_DIR}/ckpts" \
  --task_name "${TASK_NAME}" \
  --data_dir "${DATA_DIR}" \
  --save_steps 1000 \
  --logging_steps 1000 \
  --warmup_steps 100 \
  --eval_split "dev" \
  --score_average_method "micro" \
  --do_not_load_optimizer \
  --overwrite_output_dir \
