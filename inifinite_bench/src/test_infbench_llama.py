import json
from lib2to3 import refactor
from pathlib import Path
import time
from typing import List, Tuple, Any
import pdb
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM
from compute_scores import *
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
# from vllm import LLM, SamplingParams
from args import parse_args


MAX_POSITION_ID = 131072  # Determined by the model
TRUNCATE_LEN = 131072

def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    # print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    # print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    inputs = tok(input_text, return_tensors="pt", return_token_type_ids=False).to(model.device)
    prompt_length = inputs.input_ids.size()[-1]
    print('document len', prompt_length)
    sample = model.generate(**inputs, repetition_penalty=1, do_sample=False, max_new_tokens=max_tokens)
    output = tok.decode(sample[0][prompt_length:])
    output = " ".join(output.split())
    return output


def load_model(
    model_name: str = "",
):
    if args.shifted_ratio > 0.01:
        args.output_dir += "-string"
        from string_for_llama import replace_with_string
        replace_with_string(TRUNCATE_LEN, shifted_offset=int(args.shifted_ratio*TRUNCATE_LEN), small_local_value=args.local_value)
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(args.model_path, attn_implementation="flash_attention_2", device_map="auto",
                                            trust_remote_code=True, torch_dtype=torch.bfloat16)
    return llm, tok  # type: ignore


if __name__ == "__main__":
    args = parse_args()
    
    model_name = args.model_name
    print(json.dumps(vars(args), indent=4))
    data_name = args.task
    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args.model_path)

    result_dir = Path(args.output_dir, data_name)
    result_dir.mkdir(exist_ok=True, parents=True)

    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"{model_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"{model_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    scores = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Max tokens: {max_tokens}")
    fw = open(output_path, "a")
    print("Your predictions are saved to", output_path)
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred = get_pred(
            model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose
        )

        ref =  get_answer(eg, data_name)
        score = get_score_one(pred=pred, label=ref, task_name=args.task, model_name=args.model_name)
        scores.append(score)
        print("prediction:", pred)
        print("Reference:", ref)
        print("score:", score)
        print(f"avg of {len(scores)} samples:", sum(scores) / len(scores))
        print(f"step {i}, avg score {sum(scores) / len(scores)}", file=fw)
        print("-" * 20)
        fw.flush()
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
    dump_jsonl(preds, output_path)
