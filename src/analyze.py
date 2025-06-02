import sys
import re
import argparse
import json
from vllm import LLM, SamplingParams
import torch
import copy
from io import StringIO

prompt_no_input = (
    "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{user_input}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

with open("prompt_sft_criteria.txt", "r") as infile:
    prompt_criteria = infile.read()

with open("prompt_sft_analysis.txt", "r") as infile:
    prompt_eval_each = infile.read()

def clean_code(code):
    code_lst = code.split("\n")
    new_code_lst = []
    for x in code_lst:
        if x.startswith("def") and x[-1] == ":":
            new_code_lst.append(x)
        elif x.startswith("    "):
            new_code_lst.append(x)
        elif x == "":
            new_code_lst.append(x)
        elif x.startswith("import "):
            new_code_lst.append(x)
        elif x.startswith("from "):
            new_code_lst.append(x)
        elif x.startswith("try") and x[-1] == ":":
            new_code_lst.append(x)
        elif x.startswith("except") and x[-1] == ":":
            new_code_lst.append(x)
        elif x.startswith("#"):
            new_code_lst.append(x)
        elif x.startswith("nltk.download"):
            new_code_lst.append(x)
        else:
            new_code_lst[-1] += "\\n" + x
    code = "\n".join(new_code_lst)
    return code

example_str = """```python
# Example usage
responses = [response1, response2]
for i, response in enumerate(responses, 1):
    print("\nEvaluating Response {}:".format(i))
    evaluate(response)
```"""
example_cleaned_str = "\n".join(example_str.split("\n")[1:-1])

example_str_ = """```python
# Example usage
responses = [response1, response2]
for i, response in enumerate(responses, 1):
    print("\\nEvaluating Response {}:".format(i))
    evaluate(response)
```"""
example_cleaned_str_ = "\n".join(example_str_.split("\n")[1:-1])

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
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    return res_completions

def eval(llm, sampling_params, sample_lst, args, swap=False):

    prompt_lst = []
    attributes = []
    for idx in range(len(sample_lst)):
        prompt_tmp = prompt_criteria.format(instruction=sample_lst[idx]["instruction"])
        prompt_tmp = prompt_no_input.format(user_input=prompt_tmp)
        prompt_lst.append(prompt_tmp)
        attributes.append(copy.deepcopy(sample_lst[idx]))

    print("Generate evaluation questions ...")

    # for ARJudger
    evaluate_questions_lst = generate(prompt_lst, sampling_params, llm, args.batch_size)
    assert len(evaluate_questions_lst) == len(sample_lst)

    prompt_lst = []
    attributes = []
    for idx in range(len(evaluate_questions_lst)):

        # general evaluation questions
        evaluate_questions_lst[idx] += "\n2. Does the response honestly/precisely/closely execute the instruction?"

        sample_evaluate_questions = re.findall(r"\d\. (.+?)\n", evaluate_questions_lst[idx]+"\n")
        if len(sample_evaluate_questions) > 0:
            for q in sample_evaluate_questions:
                prompt_tmp = prompt_eval_each.format(instruction=sample_lst[idx]["instruction"], response1=sample_lst[idx]["response1"], response2=sample_lst[idx]["response2"], criteria=q)
                prompt_tmp = prompt_no_input.format(user_input=prompt_tmp)
                prompt_lst.append(prompt_tmp)
                attributes.append(copy.deepcopy(sample_lst[idx]))
                attributes[-1]["question"] = q

    print("Generate evaluations ...")
    evaluation_lst = generate(prompt_lst, sampling_params, llm, args.batch_size)
    for idx in range(len(attributes)):
        attributes[idx]["evaluation"] = evaluation_lst[idx]

    eval_dict = {}
    for idx in range(len(attributes)):
        if attributes[idx]["idx"] not in eval_dict:
            eval_dict[attributes[idx]["idx"]] = [attributes[idx]]
        else:
            eval_dict[attributes[idx]["idx"]].append(attributes[idx])

    prompt_lst = []
    for idx in eval_dict:
        instruction = eval_dict[idx][0]["instruction"]
        response1 = eval_dict[idx][0]["response1"]
        response2 = eval_dict[idx][0]["response2"]
        analysis = ""
        for i, x in enumerate(eval_dict[idx]):
            if x["evaluation"].startswith("Let's evaluate whether responses meet the given criteria:"):
                analysis += f"{i+1}. " + x["question"] + " "
                analysis += x["evaluation"][57:].strip() + "\n"
            elif x["evaluation"].startswith("Let's write a Python function to evaluate"):
                tmp = re.findall(r"```python\n(.+?)\n```", x["evaluation"], flags=re.DOTALL)
                response_str = f"response1 = '''{response1}'''\n\nresponse2 = '''{response2}'''"
                
                if len(tmp) >= 1:
                    code = tmp[0].split("\n# Example usage")[0]
                    for _ in range(3):
                        code = clean_code(code)
                    exec_code = code + "\n\n" + response_str
                    exec_code += "\n\n" + example_cleaned_str_

                    # Redirect stdout to both file and StringIO
                    output_stream = StringIO()
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()

                    # Capture output in StringIO
                    output_capture = StringIO()
                    sys.stdout = output_capture

                    try:
                        exec(exec_code)
                        
                        # Get captured output
                        output_text = output_capture.getvalue()

                        analysis += f"Evaluation Question {i+1}: " + x["question"] + "\n" + f"Analysis {i+1}: "
                        analysis += output_text.strip() + "\n"
                    except Exception as e:
                        print("ERROR!", e)

                    # Close and Restore stdout
                    output_capture.close()
                    sys.stdout = old_stdout

        prompt_lst.append({
            "idx": idx,
            "instruction": instruction,
            "response1": response1,
            "response2": response2,
            "analysis": analysis,
            "winner": eval_dict[idx][0]["winner"]
        })
    
    if not swap:
        save_path = "results/ARjudge_prompt_w_analysis.json"
    else:
        save_path = "results/ARjudge_prompt_w_analysis_swap.json"
    
    with open(save_path, "w") as outfile:
        json.dump(prompt_lst, outfile, indent=4)

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
    parser.add_argument("--swap", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    with open(args.data_path, "r") as infile:
        sample_lst = json.load(infile)

    sampling_params = SamplingParams(temperature=args.temp, top_p=args.top_p, top_k=50, max_tokens=args.max_length)
    llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size, dtype=torch.bfloat16, seed=args.seed)

    # no swap
    eval(llm=llm, sampling_params=sampling_params, sample_lst=sample_lst, args=args)

    # swap
    if args.swap:
        for idx in range(len(sample_lst)):
            # swap the responses
            sample_lst[idx]["winner"] = 3 - sample_lst[idx]["winner"]
            tmp = sample_lst[idx]["response1"]
            sample_lst[idx]["response1"] = sample_lst[idx]["response2"]
            sample_lst[idx]["response2"] = tmp

        eval(llm=llm, sampling_params=sampling_params, sample_lst=sample_lst, args=args, swap=True)