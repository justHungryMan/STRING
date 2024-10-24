# -*- coding:utf-8 -*-

import random
from xml.parsers.expat import model
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
random.seed(42)


def text_to_num_tokens(tokenizer, prompt):
    return tokenizer(prompt, return_tensors="pt").input_ids.size(1)

def generate_random_segments(N):
    segment_length = 0.99 / N
    random_numbers = []
    for j in range(N):
        random_number = j * segment_length + random.uniform(0, segment_length)
        random_numbers.append(random_number)
    return random_numbers


def get_input_ctx_multi(tokenizer, ctx_len, question="", needles=[]):
    depths = generate_random_segments(len(needles))
    text_list = json.load(open('data/PaulGrahamEssays.json', 'r'))["text"].split(".")
    res = []
    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.\n"
    if ctx_len > 64000:
        step = 50
    else:
        step = 1
    for i in range(0, len(text_list), step):
        text_chunk = ".".join(text_list[i:(i + step)])
        context = ""
        needle_places = [0]
        for j, depth in enumerate(depths):
            needle_places.append(int(depth * len(res)))
        for j in range(1, len(needles) + 1):
            begin = needle_places[j - 1]
            end = needle_places[j]
            context += ".".join(res[begin:end]) + ". " + needles[j - 1]
        context += ". ".join(res[needle_places[-1]:]) + ". \n" + question
        chunk_len = text_to_num_tokens(tokenizer, context) + text_to_num_tokens(tokenizer, task_description)
        if chunk_len > ctx_len:  # for system prompt and output
            if context[0] == ".":
                context = context[1:]
            return task_description + context
        else:
            res.append(text_chunk)


def get_needles(model_path):
    if "llama3.1" in model_path:
        # The postfix `the numbers are` have a very negative impact on Llama3.1's performamce
        num_4_question = "\n\nWhat are the magic numbers mentioned in the provided text?\n "
    else:
        num_4_question = "\n\nWhat are the magic numbers mentioned in the provided text?\nThe numbers are"
    begin = 100000
    end = begin * 10
    num_4_keys = [str(random.randint(begin, end)) for _ in range(4)]
    num_4_needles = [f" One of the magic number is: {num_4_keys[i]}. " for i in range(4)]
    print("using 4 needles")
    return num_4_question, num_4_needles, num_4_keys


def get_output():
    text_inputs = get_input_ctx_multi(tokenizer=tokenizer, ctx_len=test_max_length, question=question, needles=needles)
    inputs = tokenizer(text_inputs, return_tensors="pt", return_token_type_ids=False).to(model.device)
    prompt_length = inputs.input_ids.size()[-1]
    sample = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    output = tokenizer.decode(sample[0][prompt_length:])
    output = " ".join(output.split())
    score = 0
    for i, ans in enumerate(expected_answer):
        if ans.lower() in output.lower():
            score += 1
    score = (score / len(expected_answer)) * 100
    return output, score, prompt_length


def main():
    file_name = os.path.join(pred_save_path, f"string-<S={args.shifted_ratio}L>-<W={args.local_value}>.jsonl")
    fw = open(file_name, "a")
    from datetime import datetime
    start_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"--------------------- <New RUN> {start_time}-----------------------")
    fw.write(f"--------------------- <New RUN> {start_time}-----------------------\n")
    save_ds = []
    more_than_2 = []
    for i in range(args.num_test):
        save_d = {}
        output, score, prompt_length = get_output()
        print(f"----------------- sample {i} -----------------")
        print('Input Length', prompt_length)
        print('Model Prediction', output)
        print("score:", score)
        if score >= 50:
            more_than_2.append(1.0)
        else:
            more_than_2.append(0.0)
        print(f"step {i}, Acc for >=2 retrieved from 4 needles: {100 * sum(more_than_2) / len(more_than_2)}")
        print(f">=2 retrieved acc:  {100 * sum(more_than_2) / len(more_than_2)}", file=fw)
        fw.flush()
        save_d["ctx_len"] = prompt_length
        save_d["pred"] = output
        save_d["needle"] = expected_answer
        save_d["score"] = score
        save_ds.append(save_d)

    for save_d in save_ds:
        fw.write(json.dumps(save_d) + '\n')
    fw.write(f"avg:{sum(more_than_2) / len(more_than_2)}\n")
    fw.close()

    # break


if __name__ == "__main__":
    max_new_tokens = 128
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--num_test', default=500, type=int)
    parser.add_argument('--test_max_length', default=None, type=int)
    parser.add_argument('--shifted_ratio', default=0.5, type=float, help="set to 1.0 to disable string")
    parser.add_argument('--local_value', default=128, type=int)
    args = parser.parse_args()

    question, needles, expected_answer = get_needles(args.model_path)

    model_path = args.model_path
    if model_path[-1] == "/":
        model_path = model_path[:-1]
    open_source_model = model_path.split("/")[-1]

    pred_save_path = f"niah-results/{open_source_model}/"
    print(f"Your prediction file will be saved to: {pred_save_path}  , press enter to confirm...")
    os.makedirs(pred_save_path, exist_ok=True)

    if "mistral" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path)
    training_length = config.max_position_embeddings
    if args.test_max_length:
        test_max_length = args.test_max_length
    else:
        test_max_length = training_length

    test_max_length = test_max_length - max_new_tokens
    shifted_offset = int(args.shifted_ratio * test_max_length)
    if shifted_offset > 1:
        print(f"========== Test Length: {test_max_length} & Shifted Offset: {shifted_offset} & Training Length: {training_length} ==========")
        from string_for_llama import replace_with_string
        replace_with_string(training_length, shifted_offset=shifted_offset, small_local_value=args.local_value)
    else:
        print(f"========== Test Length: {test_max_length} & Training Length: {training_length} ==========")
        
    model = AutoModelForCausalLM.from_pretrained(model_path, config=config, attn_implementation="flash_attention_2",
                                                 device_map="auto",
                                                 trust_remote_code=True, torch_dtype=torch.bfloat16)

    model = model.eval()

    sys.exit(main())
