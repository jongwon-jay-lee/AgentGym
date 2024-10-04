exp_name="behavioral_clone_gsm8k"

n_epochs='1'

# accelerator config
num_processes='8'
main_process_port='8895'
config_file="/home/sr5/jay722.lee/.cache/huggingface/accelerate/default_config.yaml"

# training arguments
train_file='/home/sr5/cara/jay722.lee/data/gsm8k/train/train-00000-of-00001.parquet'
test_file='/home/sr5/cara/jay722.lee/data/gsm8k/test/test-00000-of-00001.parquet'
model_train_path="/home/sr5/cara/jay722.lee/ckpt/Llama-2-7b-chat-hf"
model_save_path="${model_train_path}-${exp_name}/"

batch_size="2"
eval_batch_size="1"
gradient_accumulation_steps="2"
max_input_length="4096"
num_workers="8"
learning_rate="1e-5"
weight_decay="0"
warmup_step="-100"
clip_grad_norm="1"
seed="42"

logging_epoch_freq="1"
saving_epoch_freq="1"
logging_step_freq="5"

# wandb config
wandb_log="True"
wandb_project="agentenv"
wandb_run_name="${exp_name}"

# environment parameters
data_len="200"
timeout="2400"

# eval
task_list=("webshop" "alfworld" "textcraft" "sciworld")
# eval parameters
max_round_list=("6" "30" "20" "30")
env_server_base_list=("http://127.0.0.1:36004" "http://127.0.0.1:36002" "http://127.0.0.1:36008" "http://127.0.0.1:36010")

mkdir -p "${model_save_path}"
# step1: train
accelerate launch \
        --config_file "${config_file}" \
        --num_processes=${num_processes} \
        --main_process_port=${main_process_port} \
        train_behavioral_clone_wo_task.py \
        --train_file "${train_file}" \
        --inference_file "${test_file}" \
        --test_file "${test_file}" \
        --model_train_path "${model_train_path}" \
        --model_save_path "${model_save_path}" \
        --task_name "${task_list[1]}" \
        --batch_size "${batch_size}" \
        --eval_batch_size "${eval_batch_size}" \
        --n_epochs "${n_epochs}" \
        --num_workers "${num_workers}" \
        --learning_rate "${learning_rate}" \
        --weight_decay "${weight_decay}" \
        --warmup_step "${warmup_step}" \
        --clip_grad_norm "${clip_grad_norm}" \
        --logging_epoch_freq "${logging_epoch_freq}" \
        --saving_epoch_freq "${saving_epoch_freq}" \
        --logging_step_freq "${logging_step_freq}" \
        --seed "${seed}" \
        --max_input_length "${max_input_length}" \
        --max_round "${max_round_list[1]}" \
        --gradient_accumulation_steps "${gradient_accumulation_steps}" \
        --env_server_base "${env_server_base_list[1]}" \
        --data_len "${data_len}" \
        --timeout "${timeout}"
