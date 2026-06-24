import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_split(name: str):
    if name == "gsm8k":
        return load_dataset("gsm8k", "main", split="test"), "question", "answer"
    if name == "math500":
        return load_dataset("HuggingFaceH4/MATH-500", split="test"), "problem", "answer"
    raise ValueError(f"unsupported dataset: {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "math500"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds, question_key, answer_key = load_split(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    out_path = out_dir / f"{args.dataset}_generations.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            prompt = row[question_key]
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            do_sample = args.temperature > 0
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
            )
            completion = tokenizer.decode(generated[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
            f.write(json.dumps({
                "idx": i,
                "prompt": prompt,
                "prediction": completion,
                "answer": row.get(answer_key),
            }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
