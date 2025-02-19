import torch
from vllm import LLM, SamplingParams

# Define the model and sampling parameters
model_name = "your-model-name"  # Replace with your model name or path
sampling_params = SamplingParams(
    temperature=0.7,  # Controls randomness (lower = more deterministic)
    top_p=0.9,        # Nucleus sampling (top-p sampling)
    max_tokens=100,    # Maximum number of tokens to generate
    n=1,              # Number of outputs to generate
)

# Initialize the LLM
llm = LLM(model=model_name)

# Define your input prompts
prompts = [
    "What is the capital of France?",
    "Explain the theory of relativity in simple terms.",
    "Write a short story about a robot learning to love.",
]

# Run inference
outputs = llm.generate(prompts, sampling_params)

# Print the results
for i, output in enumerate(outputs):
    print(f"Prompt: {prompts[i]}")
    print(f"Generated text: {output.outputs[0].text}")
    print("-" * 50)
