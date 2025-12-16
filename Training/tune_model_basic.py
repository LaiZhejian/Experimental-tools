# %%
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training
import torch
from torch import Tensor
from tqdm.auto import tqdm

# %%
# model_path = r'G:\LLM\llama-7b-hf'
# model_path = r'G:\LLM\meta-llama_Llama-2-13b-hf'
# model_path = 'meta-llama/Llama-2-13b-hf'
# model_path = '/home/nfs01/llama2/hf/Llama-2-13b-hf'
# model_path = '/home/liusz/hpc_cloud/decapoda-research_llama-7b-hf'
model_path = '/data0/liusz/meta-llama_Llama-2-13b-hf'
# model_path = 'facebook/opt-125m'
#train_data_path = r'E:\NLP\corpus\llm_reasoning\strategyqa_github\data\strategyqa\train.json'
train_data_path = 'strategyqa/data/strategyqa/train.json'
output_base_dir = 'llm_reasoning_outputs/llama2_13b/20230910_lr1e-5'

# %%
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)


# %%
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=False,
    )
)

print('model loaded!')
#exit()

def get_train_dataset(path: str):
    # load data here
    raise NotImplementedError()

# %%
prepare_model_for_kbit_training(model, use_gradient_checkpointing=True) # this must be before lora wrapping
peft_config = LoraConfig(
    r=32,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
)
peft_model = PeftModel(model, peft_config)
peft_model.train()
peft_model.print_trainable_parameters()

# %%
opt = torch.optim.AdamW(
    (p for p in peft_model.parameters() if p.requires_grad),
    lr=1e-5,
    # lr=5e-5,
    betas=(0.9, 0.98), 
    eps=1e-4,
    fused=False,
    differentiable=False,
)

# %%
def model_forward(input_ids: Tensor, labels: Tensor, label_smoothing: float = 0.0, **kwargs) -> Tensor:
    import torch.backends.cuda
    with torch.backends.cuda.sdp_kernel(
        enable_flash=True,
        enable_math=True,
        enable_mem_efficient=True,
    ):
        outputs = peft_model(
            input_ids=input_ids.to(model.device),
            labels=labels.to(model.device),
        )
    
    return outputs.loss


# %%
dataset = get_train_dataset(train_data_path)
dataset_train = dataset[100:]
dataset_valid = dataset[:100]
total_epoch = 3
total_step = total_epoch * len(dataset)
current_step = 0
accumulate_step = 4
eval_steps = 200
assert eval_steps % accumulate_step == 0
lrs = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: step / total_step * 10 if step / total_step < 0.1 else 1)

loss_acc = 0
for epoch in range(total_epoch):
    for s in tqdm(dataset_train):
        peft_model.train()
        inputs = s.to(peft_model.device)
        # inputs = {'input_ids': torch.tensor([s]), 'labels': torch.tensor([label])}
        # inputs = tokenizer(s, text_target=s, return_tensors='pt', max_length=1024, truncation=True)
        loss = model_forward(**inputs)
        loss.backward()
        loss_acc += loss.cpu().item()
        current_step += 1
        lrs.step()
        if current_step % accumulate_step > 0:
            continue
        opt.step()
        opt.zero_grad()

        if current_step % eval_steps == 0:
            loss_avg = loss_acc / eval_steps
            print(f'{epoch=},{current_step=}')
            print(f'train: {loss_avg=}')
    
            loss_acc = 0
            for s in dataset_valid:
                peft_model.eval()
                # inputs = {'input_ids': torch.tensor([s])}
                # inputs['labels'] = inputs['input_ids']
                inputs = s.to(peft_model.device)
                with torch.no_grad():
                    loss_acc += model_forward(**inputs).cpu().item()
            loss_avg = loss_acc / len(dataset_valid)
            print(f'valid: {loss_avg=}')

            loss_acc = 0

            peft_model.save_pretrained(
                f'{output_base_dir}/adapter/step{current_step:05}',
                safe_serialization=True,
            )

# %%
peft_model.save_pretrained(
    f'{output_base_dir}/adapter/test',
    safe_serialization=True,
)
