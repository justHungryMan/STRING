import os

# For Windows
if os.name == 'nt':
    os.system('cls')
# For MacOS and Linux
else:
    os.system('clear')

import torch
# Set the random seed
seed = 42
torch.manual_seed(seed)

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig,TextStreamer
from string_for_llama import replace_with_string, replace_with_oom
import fitz
import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--shifted_offset", type=int, default=0)
parse.add_argument("--input", type=str, default="2407.21783.pdf")
args = parse.parse_args()

def parse_pdf2text(filename):
    doc = fitz.open(filename)
    text = ""
    for i, page in enumerate(doc):  # iterate the document pages
        text += f"<{filename}>: " + page.get_text()  # get plain text encoded as UTF-8
    print("read from: ", filename)
    return  text


model_path = ""
config = AutoConfig.from_pretrained(model_path)
# STRING
if args.shifted_offset > 0:
    replace_with_string(max_test_length=131072, shifted_offset=args.shifted_offset)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="flash_attention_2", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")

prompt = parse_pdf2text(args.input)
# Use TextStreamer for token-by-token streaming
streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True, do_sample=False)
# our questions
# How many long-context training stages does Llama3 have
# Describe the differences in model architecture between Llama3 and Llama2
# How does Llama3 perform context parallelism in training
# Describe the sources of Llama3's SFT data
# From what is Llama-3's multilingual SFT data derived

while True:
    # question = "Describe the differences of model architecture between Llama-3 and Llama-2."
    question = input("[User]: ")
    input_text = "Read the this paper and answer the question after the paper.\n\n\n\n" + prompt + f"\n\n\n\n{question}"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    print("input length: ",prompt_length)
    print("[Model Response]: ")
    # text
    model.generate(
        **inputs,
        max_new_tokens=256,
        streamer=streamer
    )
    print("-"*30)
