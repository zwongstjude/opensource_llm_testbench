# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datetime import datetime
import torch
import argparse
import os
import json

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, nargs='?', default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument("--quantization", type=int, nargs='?', default=16)
parser.add_argument("--token", type=str, nargs='?', default="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")

args = parser.parse_args()

print(args)

quantize_4bit = True if args.quantization == 4 else False
quantize_8bit = True if args.quantization == 8 else False

token = args.token

tokenizer = AutoTokenizer.from_pretrained(args.model, token=token)
model = AutoModelForCausalLM.from_pretrained(args.model, device_map=device, load_in_4bit=quantize_4bit, load_in_8bit=quantize_8bit, token=token)
streamer = TextStreamer(tokenizer)

q_bank = []
bank = []

with open("prompt.json") as f:
    data = json.load(f)
    q_bank = data['prompts']
    bank = data['labels']

timing = []
for i in range(len(bank)):
    print("performing: ", bank[i], " test:")
    begin = datetime.now()
    inputs = tokenizer(q_bank[i], return_tensors='pt').to(device)
    outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=100)
    generation_time = datetime.now()-begin
    print("time: ", generation_time)
    print("seconds per token: ", generation_time.total_seconds() / (outputs[0].shape[0] - inputs.input_ids[0].shape[0]))
    timing.append((outputs[0].shape[0] - inputs.input_ids[0].shape[0]) / generation_time.total_seconds())
    print("tokens per seconds: ", (outputs[0].shape[0] - inputs.input_ids[0].shape[0]) / generation_time.total_seconds())

vram = int(os.popen('nvidia-smi | grep -oP "[0-9]+(?=MiB \/)"').read())

with open('out.txt', 'w') as f:
    f.write(f"tokens per second {timing}\n\n")
    f.write(f"vram {vram}\n\n")
    f.write(f"{os.popen('nvidia-smi').read()}")

# Manual testing at the end
inp=""
while inp!="stop":
    inp = input()
    begin = datetime.now()
    inputs = tokenizer(inp, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, streamer=streamer, max_new_tokens=100)
    generation_time = datetime.now()-begin
    print("time: ", generation_time)
    print("seconds per token: ", generation_time / outputs[0].shape[0])
