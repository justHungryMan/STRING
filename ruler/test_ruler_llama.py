# -*- coding:utf-8 -*-
import importlib
from re import I
import re
import yaml
import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import sys
import json
from utils import read_manifest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main():
    fw = open(file_name, "a")
    scores = []
    save_ds = []
    for i, data in enumerate(data_list):
        text_inputs = data["input"]
        inputs = tokenizer(text_inputs, return_tensors="pt", return_token_type_ids=False).to(model.device)
        prompt_length = inputs.input_ids.size()[-1]
        sample = model.generate(**inputs, repetition_penalty=1, do_sample=False, max_new_tokens=max_new_tokens)
        output = tokenizer.decode(sample[0][prompt_length:])
        output = " ".join(output.split())
        save_d = {}
        ref = data["outputs"]
        print(f"----------------- sample {i} -----------------")
        print('[Model Prediction]',output)
        print('[Ground Truth]', ref)
        if "qa" in args.task:
            score = max([r.lower() in output.lower() for r in ref])
        else:
            score_curr = [1.0 if r.lower() in output.lower() else 0.0 for r in ref]
            score = sum(score_curr) / len(score_curr)
        print("[score]:", score)
        scores.append(score)
        print(f"===== step {i}, ctx len {prompt_length}, avg score {sum(scores) / len(scores)} =====")
        print(f"step {i}, ctx len {prompt_length}, avg score {sum(scores) / len(scores)}", file=fw)
        fw.flush()
        save_d["ctx_len"] = prompt_length
        save_d["pred"] = output
        save_d["needle"] = ref
        save_d["score"] = score
        save_ds.append(save_d)

    for save_d in save_ds:
        fw.write(json.dumps(save_d) + '\n')
    fw.write(f"avg:{sum(scores) / len(scores)}\n")
    fw.close()
    print(f"avg:{sum(scores) / len(scores)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument("--data_dir", type=str, help='path to load the dataset jsonl files')
    parser.add_argument("--benchmark", type=str, default='synthetic', help='Options: [synthetic]')
    parser.add_argument("--task", type=str, help='Options: tasks in benchmark')
    parser.add_argument('--shifted_ratio', default=0.33, type=float, help="set to 0.0 to disable string")
    parser.add_argument('--local_value', default=128, type=int)
    args = parser.parse_args()

    test_max_length = int(re.search(r'\d{4,}', args.data_dir).group())
    print("the maximum input length is ", test_max_length)

    # copied from https://github.com/hsiehjackson/RULER/blob/main/scripts/data/synthetic/constants.py#L24
    if "vt" in args.task :
        max_new_tokens = 30
    elif "cwe" in args.task:
        max_new_tokens = 120
    elif "fwe" in args.task:
        max_new_tokens = 50
    elif "qa" in args.task:
        max_new_tokens = 32
    elif "niah"  in args.task:
        max_new_tokens = 128
    else:
        raise NotImplementedError("Unsupported task")

    model_path = args.model_path
    open_source_model = model_path.split("/")[-1]
    if len(open_source_model) == 0:
        open_source_model = model_path.split("/")[-2]

    print("*" * 10, "Data loading", "*"*10)
    if args.data_dir is None:
        args.data_dir = f"jsonl_data/{args.task}/{open_source_model}-{test_max_length}.jsonl"
        print("data dir is not specified. We load from", args.data_dir)
    curr_folder = os.path.dirname(os.path.abspath(__file__))


    with open(os.path.join(curr_folder, f"{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)
        if args.task not in tasks_customized:
            raise ValueError(f'{args.task} is not found in config_tasks.yaml')

    task_file = args.data_dir
    data_list = read_manifest(task_file)
    print("*" * 10, "loading ends..", "*"*10)

    pred_save_path = f"Predictions/{args.task}/{open_source_model}/"
    print(f"Your prediction file will be saved to: {pred_save_path}  , press enter to confirm...")
    os.makedirs(pred_save_path, exist_ok=True)
    file_name = os.path.join(pred_save_path, f"string-<S={args.shifted_ratio}L>-<W={args.local_value}>.jsonl")

    if "mistral" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path)
    training_length = config.max_position_embeddings    
    shifted_offset = int(args.shifted_ratio * test_max_length)
    
    if shifted_offset > 1:
        print(f"========== Test Length: {test_max_length} & Shifted offset: {shifted_offset} & Training Length: {training_length} ==========")
        from string_for_llama import replace_with_string
        replace_with_string(test_max_length, shifted_offset=shifted_offset, small_local_value=args.local_value)
    else:
        print(f"========== Test Length: {test_max_length} & Training Length: {training_length} ==========")
        from string_for_llama import replace_with_oom
        replace_with_oom()
        
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, attn_implementation="flash_attention_2",
                                                 device_map="auto",
                                                 trust_remote_code=True, torch_dtype=torch.bfloat16)

    model = model.eval()
    sys.exit(main())


