import gradio as gr
import fitz
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from string_for_llama import replace_with_string

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


def greet(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    prompt_length = inputs["input_ids"].shape[1]
    output = tokenizer.decode(model.generate(**inputs, max_new_tokens=256)[0][prompt_length:], skip_special_tokens=True)
    return output

prompt = parse_pdf2text(args.input)

demo = gr.Interface(
    fn=greet,
    inputs=["text"],
    outputs=["text"],
)

demo.launch(server_name='10.124.44.231', server_port=7680)
