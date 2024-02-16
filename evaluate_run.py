from human_eval.data import write_jsonl, read_problems 
from transformers import LlamaTokenizer, LlamaForCausalLM 
import argparse 

import socket

hostname = socket.gethostname()
print("Hostname:", hostname)

if "lovelace" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    dir_unprocessed_dataset = "/home/yangzho6/c4_parts/downloads/" 
elif "ada" in hostname: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/home/beidic/yangzho6/model_checkpoints/" 
    dir_sdata = "/home/beidic/yangzho6/c4llm_synthesized/" 
    dir_unprocessed_dataset = "/home/beidic/yangzho6/c4_parts/downloads/" 
else: 
    # cache_dir = "/home/bc20/yang/transformersprofiling" 
    dir_models = "/fsx-storygen/beidic/yang/model_checkpoints/" 
    # dir_sdata = "/home/yangzho6/c4llm_synthesized/" 
    dir_sdata = "/fsx-storygen/beidic/yang/c4llm_synthesized/" 

args = argparse.ArgumentParser() 
args.add_argument("--model_name", type = str, required = True) 

args = args.parse_args() 

problems = read_problems() 

num_samples_per_task = 200 
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in problems
    for _ in range(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples) 

def generate_one_completion(prompt): 
    tokenizer = LlamaTokenizer.from_pretrained(args.model_name, cache_dir = dir_models) 
    model = LlamaForCausalLM.from_pretrained(args.model_name, cache_dir = dir_models) 
    input_ids = tokenizer.encode(prompt, return_tensors = "pt") 
    
    output_sequences = model.generate(
        input_ids, 
        max_length = 128, 
        temperature = 0.7, 
        top_p = 0.9, 
        num_return_sequences = 1, 
        pad_token_id = tokenizer.eos_token_id, 
        do_sample = True, 
    ) 
    
    generated_sequence = tokenizer.decode(output_sequences[0], skip_special_tokens = True) 
    
    completion = generated_sequence[len(prompt):] 
    
    return completion.strip() 
