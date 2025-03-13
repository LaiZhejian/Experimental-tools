import argparse
import json
from vllm import LLM, SamplingParams

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run inference using vLLM.")
    parser.add_argument("--model_name", type=str, required=True, help="Name or path of the model to use.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling (default: 0.7).")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling (default: 0.9).")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate (default: 100).")
    parser.add_argument("--num_outputs", type=int, default=1, help="Number of outputs to generate per prompt (default: 1).")
    parser.add_argument("--test_set", type=str, required=True, help="Path to a JSON file containing a list of prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the JSON file to store outputs.")
    args = parser.parse_args()

    # Load prompts from the JSON test set file
    with open(args.test_set, "r", encoding="utf-8") as file:
        prompts = json.load(file)
    
    if not isinstance(prompts, list):
        raise ValueError("The test set JSON file must contain a list of prompts.")

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_outputs,
    )

    # Initialize the LLM
    llm = LLM(model=args.model_name)

    # Run inference
    outputs = llm.generate(prompts, sampling_params)

    # Prepare results in JSON format
    results = []
    for i, output in enumerate(outputs):
        result_entry = {
            "prompt": prompts[i],
            "generated_texts": [gen.text for gen in output.outputs]
        }
        results.append(result_entry)

    # Save results to a JSON file
    with open(args.output_file, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=4, ensure_ascii=False)

    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
