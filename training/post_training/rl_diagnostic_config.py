import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["grpo", "dpo"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-completion-length", type=int, default=8192)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "method": args.method,
        "model": args.model,
        "train_file": args.train_file,
        "notes": (
            "This file records the public diagnostic interface for the text-only "
            "RL branches. These branches were not used to produce the released "
            "OpenAster-1 checkpoints."
        ),
    }
    if args.method == "grpo":
        config["grpo"] = {
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "num_generations": args.num_generations,
        }
    else:
        config["dpo"] = {"beta": args.beta}

    with (output_dir / "rl_diagnostic_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {output_dir / 'rl_diagnostic_config.json'}")


if __name__ == "__main__":
    main()
