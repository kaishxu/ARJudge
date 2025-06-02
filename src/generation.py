import os
import argparse
import json
import random
import copy
import time

from vllm import LLM, SamplingParams
import torch

prompt_no_input = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{user_input}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    batch_data.append(data_list[last_start:])
    return batch_data

def generate(instruction_lst, sampling_params, llm, batch_size=1):

    batch_instruction_lst = batch_data(instruction_lst, batch_size=batch_size)
    res_completions = []
    for idx, prompt in enumerate(batch_instruction_lst):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        start_time = time.time()
        completions = llm.generate(prompt, sampling_params)
        end_time = time.time()
        elapsed_time = (end_time - start_time) / len(completions)
        print(f"Average Generation Timing: {elapsed_time:.4f} seconds")
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    return res_completions

def inference(llm, sampling_params, sample_lst, args):

    prompt_lst = []
    attributes = []
    for idx in range(len(sample_lst)):
        prompt_tmp = prompt_no_input.format(user_input=sample_lst[idx]["prompt"])
        prompt_lst.append(prompt_tmp)
        attributes.append(copy.deepcopy(sample_lst[idx]))

    completions_lst = generate(prompt_lst, sampling_params, llm, args.batch_size)
    assert len(completions_lst) == len(sample_lst)
    for idx in range(len(attributes)):
        attributes[idx]["model_output"] = completions_lst[idx]

    return attributes

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='')  # model path
    parser.add_argument("--data_path", type=str, default='')  # data path
    parser.add_argument("--batch_size", type=int, default=1000)  # batch_size
    parser.add_argument("--tensor_parallel_size", type=int, default=8)  # tensor_parallel_size
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swap", action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    sampling_params = SamplingParams(temperature=args.temp, top_p=args.top_p, top_k=50, max_tokens=args.max_length)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, dtype=torch.bfloat16, seed=args.seed)

    # eval
    with open("prompt_refine.json", "r") as infile:
        eval_template = infile.read()

    with open(args.data_path, "r") as infile:
        sample_lst = json.load(infile)

    for idx in range(len(sample_lst)):

        if "prompt" in sample_lst[idx]:
            continue

        instruction = sample_lst[idx]["instruction"]
        response1 = sample_lst[idx]["response1"]
        response2 = sample_lst[idx]["response2"]
        prompt_tmp = eval_template.format(instruction=instruction, response1=response1, response2=response2, analysis=sample_lst[idx]["analysis"])
        sample_lst[idx]["prompt"] = prompt_tmp

    sample_lst = inference(llm=llm, sampling_params=sampling_params, sample_lst=sample_lst, args=args)

    with open(args.save_path, "w") as outfile:
        json.dump(sample_lst, outfile, indent=4)

    # eval with swap
    if args.swap:
        with open(args.data_path.split(".json")[0] + "_swap.json", "r") as infile:
            sample_lst = json.load(infile)
        for idx in range(len(sample_lst)):
            instruction = sample_lst[idx]["instruction"]
            response1 = sample_lst[idx]["response1"]
            response2 = sample_lst[idx]["response2"]
            prompt_tmp = eval_template.format(instruction=instruction, response1=response1, response2=response2, analysis=sample_lst[idx]["analysis"])
            sample_lst[idx]["prompt"] = prompt_tmp

        sample_lst = inference(llm=llm, sampling_params=sampling_params, sample_lst=sample_lst, args=args)

        with open(args.save_path.split(".json")[0] + "_swap.json", "w") as outfile:
            json.dump(sample_lst, outfile, indent=4)
