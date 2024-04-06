export KMER=4
export TRAIN_FILE=../pretrain_data/pretrain_4mer_all.txt
export TEST_FILE=../pretrain_data/pretrain_4mer_all.txt

export SOURCE=../../..
export OUTPUT_PATH=../pretrain_result/4mer_2k4k

export TRAIN_BATCH=5
export EVAL_BATCH=3


python ../../run_pretrain.py \
 --output_dir $OUTPUT_PATH \
 --model_type=dna \
 --tokenizer_name=dna$KMER \
 --config_name=$SOURCE/src/transformers/dnabert-config/bert-config-$KMER/config.json \
 --do_train \
 --train_data_file=$TRAIN_FILE \
 --do_eval \
 --eval_data_file=$TEST_FILE \
 --mlm \
 --gradient_accumulation_steps 25 \
 --per_gpu_train_batch_size $TRAIN_BATCH \
 --per_gpu_eval_batch_size $EVAL_BATCH \
 --save_steps 500 \
 --save_total_limit 20 \
 --max_steps 10000 \
 --evaluate_during_training \
 --logging_steps 500 \
 --line_by_line \
 --learning_rate 4e-4 \
 --block_size 512  \
 --adam_epsilon 1e-6 \
 --weight_decay 0.01 \
 --beta1 0.9 \
 --beta2 0.98 \
 --mlm_probability 0.025 \
 --warmup_steps 500  \
 --overwrite_output_dir \
 --n_process 24

