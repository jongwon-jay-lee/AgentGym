import os
import json
import re
import pandas as pd

from transformers import AutoModel, AutoTokenizer
from vllm import LLM, SamplingParams

def generate_examples(file_path, ext_type):
    
    if ext_type == "jsonl":
        data_lst = []
        with open(file_path, "r", encoding="utf-8") as f_in:
            for _, row in enumerate(f_in):
                data = json.loads(row)
                data_lst.append(data)
        return data_lst
    elif ext_type == "parquet":
        curr_table = pd.read_parquet(file_path)
        total_len = len(curr_table)
        json_data = json.loads(curr_table.to_json())
        questions = json_data["question"]
        answers = json_data["answer"]
        data_lst = []
        for idx in range(0, total_len):
            # print(idx)
            data_lst.append({
                "question": questions[str(idx)],
                "answer": answers[str(idx)]
            })
        return data_lst
    

def extract_answer(_answer, _dataset_name):
    
    if _dataset_name == "math":
        
        # # MATH_ANS_PATTERN =  re.compile(r"\\boxed\{(.*)\}\.*\$*")
        # MATH_ANS_PATTERN =  re.compile(r"\\boxed\{((.|\s)+)\}\.*\$*")
        # m = MATH_ANS_PATTERN.findall(_answer)
        # if m:
        #     # return m[0].split("}$")
        #     return m[0][0]
        # else:
        #     raise ValueError(f"{_answer}")
        
        return _answer.split("\\boxed{")[-1].split("}")[0]
    elif _dataset_name == "gsm8k":
        return _answer.split("#### ")[1]
    else:
        raise ValueError(f"{_dataset_name} NOT SUPPORTED")
    

def evaluate(llm, sampling_params, examples, dataset_name="math", q_key="problem", a_key="solution", name=None):
    results = []
    num_total = 0
    num_correct = 0
    prompt = "Question: {}\nAnswer: "
    print(f"START {name}")
    for idx, example in enumerate(examples):
        curr_q = example.get(q_key, "")
        curr_a = example.get(a_key, "")        
        if curr_q and curr_a:
            
            extracted_a = extract_answer(curr_a, dataset_name)
            
            generated_text = llm.generate(prompt.format(curr_q), sampling_params, use_tqdm=False)[0].outputs[0].text
            print(f"#{idx}:\t{extracted_a}\t{generated_text}\n")
            
            results.append({
                "id": idx,
                "question": curr_q,                
                "generated": generated_text,
                "extracted_a": extracted_a,
                "answer": curr_a,
            })

            if str(extracted_a).lower().strip() == str(generated_text).lower().strip():
                num_correct += 1
            num_total += 1
        else:
            print(f"{idx} MISSING Q OR A")

    print(f"DONE {name}")
    
    return {
        "name": name,
        "total": num_total,
        "correct": num_correct,
        "results": results
    }


def main():
    ckpt_path = "/home/sr5/cara/jay722.lee/ckpt/llama-3"

    sampling_params = SamplingParams(
        temperature=0, max_tokens=1000, skip_special_tokens=False, stop=['<|end_of_text|>', 'Question:', "Explanation:"])

    llm = LLM(model=ckpt_path, tensor_parallel_size=8, max_model_len=8192)

    data_path = "/home/sr5/cara/jay722.lee/data/MATH/test"

    whole_result = []
    for subdir, _, files in os.walk(data_path):
        for file in files:
            if file.endswith("jsonl"):
                curr_file = os.path.join(subdir, file)
                examples = generate_examples(curr_file, "jsonl")
                eval_result = evaluate(
                    llm, sampling_params, examples, dataset_name="math", q_key="problem", a_key="solution", name=file[:-len(".jsonl")])
                whole_result.append(eval_result)

            elif file.endswith("parquet"):
                curr_file = os.path.join(subdir, file)
                examples = generate_examples(curr_file, "parquet")
                eval_result = evaluate(
                    llm, sampling_params, examples, dataset_name="gsm8k", q_key="question", a_key="answer", name=file[:-len(".parquet")])
                whole_result.append(eval_result)

    with open(os.path.join(data_path, "test_result.json"), "w", encoding="utf-8") as f_out:
        json.dump(whole_result, f_out, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
    