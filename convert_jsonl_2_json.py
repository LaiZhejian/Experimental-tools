import json

file_path = "/home/nfs06/laizj/code/LLaMA-Factory/data/MGSM8KInstruct_Parallel.jsonl"
output_path = "/home/nfs06/laizj/code/LLaMA-Factory/data/MGSM8KInstruct_Parallel.json"

with open(file_path) as f:
    data = []
    for line in f:
        data.append(json.loads(line))

with open(output_path, "w") as f:
    f.write(json.dumps(data, indent=4, ensure_ascii=False))