# Data Preparation for AgentEvol

## 1) Prepare 3 files 
* Script: `agentenv/examples/agentevol/discrete_supercom_train_agenteovl_alfworld_temp_0.sh`
1. Train (`alfworld_train.json`) from [AgentGym/AgentTraj-L](https://huggingface.co/datasets/AgentGym/AgentTraj-L)
2. Test (`alfworld_test.json`) from [AgentGym/AgentEval](https://huggingface.co/datasets/AgentGym/AgentEval)
3. Inference (`alfworld.json`), currently same amount as training but empty by code below:
```
import json
train_len = 2420 # num of train

train_lst = []
for idx in range(train_len):
    train_lst.append({
        "item_id": f"alfworld_{idx}",
        "conversation": []
    })

fileout_path = "./alfworld.json"
with open(fileout_path, "w", encoding="utf-8") as f_out:
    json.dump(train_lst, f_out, ensure_ascii=False, indent=4)
```


## 2) Setup Env

Follow instruction in https://github.com/jongwon-jay-lee/AgentGym/tree/main/agentenv-alfworld


# Train `AgentEvol`

Edit the paths in `agentenv/examples/agentevol/discrete_supercom_train_agenteovl_alfworld_temp_0.sh`:

* model_train_path
* config_file (accelerate)
* start_iter / end_iter
* env_server_base_list

```
cd agentenv/examples/agentevol
bash discrete_supercom_train_agenteovl_alfworld_temp_0.sh
```
* The script has 5 steps:
  * Train a model
  * Eval on test dataset
  * Inference on train dataset
  * Filter inference results (keep only correct ones)
  * Update model path


# PDDL Solution 
## 1) extract_pddl_solution.py
* Edit the path:
  * solution_path
  * fd_path
  * problem_path
  * num_prob / max_prob (for num of solutions)
* Run:
```
python extract_pddl_solution.py
```

## 2) convert_pddl_to_nl.py
* Edit the `problem_path`
* Run (NL outputs located in the same path as `problem_path`):
```
python convert_pddl_to_nl.py
```

## 3) verify_nl_solution.py
* Edit the `nl_solution_path` if necessary
* Run:
```
python verify_nl_solution.py
```
