import argparse
import io
import json
import math
import os
import random
import zipfile
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, CLIPImageProcessor, LlavaForConditionalGeneration, get_cosine_schedule_with_warmup

from training_checkpoint import (
    ResumableDistributedSampler,
    load_training_checkpoint_state,
    save_training_checkpoint_state,
)


IGNORE_INDEX = -100
IMAGE_TOKEN = "<|image_pad|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["projector_pretrain", "mm_sft"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--image-seq-len", type=int, default=576)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action="store_true")
    parser.add_argument("--image-zip", default="")
    parser.add_argument("--resume-training-state", action="store_true")
    parser.add_argument("--resume-state-dir", default="")
    return parser.parse_args()


def distributed_info() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def expand_image_tokens(text: str, image_seq_len: int) -> str:
    return text.replace(IMAGE_TOKEN, IMAGE_TOKEN * image_seq_len)


def render_messages(tokenizer, messages: list[dict[str, Any]], image_seq_len: int, add_generation_prompt: bool) -> str:
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return expand_image_tokens(text, image_seq_len)


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    normalized: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            return None
        if role == "assistant":
            if not isinstance(content, str) or not content.strip():
                return None
            normalized.append({"role": role, "content": content.strip()})
        else:
            if isinstance(content, list):
                normalized.append({"role": role, "content": content})
            elif isinstance(content, str) and content.strip():
                normalized.append({"role": role, "content": content.strip()})
            else:
                return None
    if len(normalized) < 2 or normalized[0]["role"] != "user":
        return None
    if not any(message["role"] == "assistant" for message in normalized):
        return None
    return normalized


def normalize_image_assets(row: dict[str, Any], default_image_zip: str = "") -> list[dict[str, str]] | None:
    assets: list[dict[str, str]] = []

    image_paths = row.get("images")
    if isinstance(image_paths, list):
        for image_path in image_paths:
            if not isinstance(image_path, str) or not image_path:
                return None
            if not os.path.isfile(image_path):
                return None
            assets.append({"image": image_path, "image_zip": "", "image_member": ""})
    elif isinstance(image_paths, str) and image_paths:
        if not os.path.isfile(image_paths):
            return None
        assets.append({"image": image_paths, "image_zip": "", "image_member": ""})

    image_path = row.get("image") or ""
    if image_path:
        if not os.path.isfile(image_path):
            return None
        assets.append({"image": image_path, "image_zip": "", "image_member": ""})

    image_zip = row.get("image_zip") or default_image_zip
    image_rel = row.get("image_rel") or ""
    if image_zip and image_rel:
        assets.append({"image": "", "image_zip": image_zip, "image_member": image_rel})

    image_zips = row.get("image_zips")
    image_rels = row.get("image_rels")
    if isinstance(image_zips, list) or isinstance(image_rels, list):
        if not isinstance(image_zips, list) or not isinstance(image_rels, list) or len(image_zips) != len(image_rels):
            return None
        for zip_path, member in zip(image_zips, image_rels):
            if not isinstance(zip_path, str) or not isinstance(member, str) or not zip_path or not member:
                return None
            assets.append({"image": "", "image_zip": zip_path, "image_member": member})

    # Preserve old rows while allowing explicit multi-image rows.
    if not assets:
        return []
    return assets


def build_labels(tokenizer, messages: list[dict[str, Any]], image_seq_len: int, max_length: int) -> tuple[list[int], list[int]] | None:
    full_text = render_messages(tokenizer, messages, image_seq_len, add_generation_prompt=False)
    input_ids = tokenizer(full_text, add_special_tokens=False).input_ids[:max_length]
    if not input_ids:
        return None

    labels = [IGNORE_INDEX] * len(input_ids)
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        start_text = render_messages(tokenizer, messages[:idx], image_seq_len, add_generation_prompt=True)
        end_text = render_messages(tokenizer, messages[: idx + 1], image_seq_len, add_generation_prompt=False)
        start = len(tokenizer(start_text, add_special_tokens=False).input_ids)
        end = len(tokenizer(end_text, add_special_tokens=False).input_ids)
        if start >= max_length:
            continue
        end = min(end, max_length, len(input_ids))
        if end > start:
            labels[start:end] = input_ids[start:end]

    if all(label == IGNORE_INDEX for label in labels):
        return None
    return input_ids, labels


class VlmJsonlDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        image_processor,
        max_length: int,
        image_seq_len: int,
        limit: int = 0,
        image_zip: str = "",
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_seq_len = image_seq_len
        self.image_zip = image_zip
        self.path = path
        self.offsets: list[int] = []
        with open(path, "rb") as fh:
            while True:
                if limit and len(self.offsets) >= limit:
                    break
                offset = fh.tell()
                line = fh.readline()
                if not line:
                    break
                if line.strip():
                    self.offsets.append(offset)
        if not self.offsets:
            raise ValueError(f"No usable VLM samples loaded from {path}")

    def encode_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        messages = normalize_messages(row.get("messages") or [])
        if messages is None:
            return None
        image_assets = normalize_image_assets(row, self.image_zip)
        if image_assets is None:
            return None
        built = build_labels(self.tokenizer, messages, self.image_seq_len, self.max_length)
        if built is None:
            return None
        input_ids, labels = built
        image_token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)

        image_token_count = input_ids.count(image_token_id)
        expected_image_tokens = len(image_assets) * self.image_seq_len
        if image_token_count != expected_image_tokens:
            return None
        return {
            "input_ids": input_ids,
            "labels": labels,
            "image_assets": image_assets,
        }

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # A few samples can be skipped after tokenization if too long; advance deterministically.
        for attempt in range(16):
            real_index = (index + attempt) % len(self.offsets)
            with open(self.path, "rb") as fh:
                fh.seek(self.offsets[real_index])
                row = json.loads(fh.readline().decode("utf-8"))
            encoded = self.encode_row(row)
            if encoded is not None:
                return encoded
        raise ValueError(f"Could not encode a usable sample near index {index}")


@dataclass
class VlmDataCollator:
    pad_token_id: int
    image_processor: Any
    _zip_files: dict[str, zipfile.ZipFile] = field(default_factory=dict, init=False)

    def open_image(self, asset: dict[str, str]) -> Image.Image:
        image_path = asset.get("image") or ""
        if image_path:
            return Image.open(image_path).convert("RGB")

        image_zip = asset.get("image_zip") or ""
        image_member = asset.get("image_member") or ""
        if not image_zip or not image_member:
            raise ValueError("Missing image path and zip-backed image metadata")
        zf = self._zip_files.get(image_zip)
        if zf is None:
            zf = zipfile.ZipFile(image_zip)
            self._zip_files[image_zip] = zf
        data = zf.read(image_member)
        return Image.open(io.BytesIO(data)).convert("RGB")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(item["input_ids"]) for item in features)
        input_ids = []
        labels = []
        attention_mask = []
        images = []
        for item in features:
            ids = item["input_ids"]
            item_labels = item["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            labels.append(item_labels + [IGNORE_INDEX] * pad_len)
            attention_mask.append([1] * len(ids) + [0] * pad_len)
            for asset in item.get("image_assets") or []:
                images.append(self.open_image(asset))
        batch = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }
        if images:
            batch["pixel_values"] = self.image_processor(images=images, return_tensors="pt").pixel_values
        return batch


def set_trainable(model: LlavaForConditionalGeneration, stage: str) -> None:
    for param in model.parameters():
        param.requires_grad = False
    if stage == "projector_pretrain":
        for param in model.model.multi_modal_projector.parameters():
            param.requires_grad = True
    elif stage == "mm_sft":
        for param in model.model.multi_modal_projector.parameters():
            param.requires_grad = True
        for param in model.model.language_model.parameters():
            param.requires_grad = True
        for param in model.lm_head.parameters():
            param.requires_grad = True
    else:
        raise ValueError(stage)


def patch_saved_config(output_dir: str, image_token_id: int, image_seq_len: int) -> None:
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    config["image_token_id"] = image_token_id
    config["image_token_index"] = image_token_id
    config["image_seq_length"] = image_seq_len
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = distributed_info()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        print(f"world_size={world_size}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, fix_mistral_regex=True)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    image_processor = CLIPImageProcessor.from_pretrained(args.model)
    dataset = VlmJsonlDataset(
        args.train_jsonl,
        tokenizer,
        image_processor,
        args.max_length,
        args.image_seq_len,
        args.limit,
        args.image_zip,
    )
    sampler = ResumableDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=VlmDataCollator(tokenizer.pad_token_id, image_processor),
        pin_memory=True,
    )

    model = LlavaForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model.config.image_token_id = image_token_id
    model.config.image_token_index = image_token_id
    model.config.image_seq_length = args.image_seq_len
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_trainable(model, args.stage)
    model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=args.ddp_find_unused_parameters,
        )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"loaded_samples={len(dataset)} trainable_params={trainable_params}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    local_batches_per_epoch = math.ceil(sampler.num_samples / args.per_device_batch_size)
    update_steps_per_epoch = math.ceil(local_batches_per_epoch / args.gradient_accumulation_steps)
    total_update_steps = max(1, update_steps_per_epoch * args.epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_update_steps * args.warmup_ratio), total_update_steps)

    global_step = 0
    start_epoch = 0
    resume_step_in_epoch = 0
    running_loss = 0.0
    if args.resume_training_state:
        resume_dir = args.resume_state_dir or args.model
        state = load_training_checkpoint_state(resume_dir, optimizer=optimizer, scheduler=scheduler, rank=rank, map_location=device)
        global_step = int(state["global_step"])
        start_epoch = int(state["epoch"])
        resume_step_in_epoch = int(state["step_in_epoch"])
        running_loss = float(state.get("running_loss", 0.0))
        if resume_step_in_epoch >= local_batches_per_epoch:
            start_epoch += 1
            resume_step_in_epoch = 0
        if rank == 0:
            print(
                "loaded_training_state="
                f"{resume_dir} global_step={global_step} epoch={start_epoch} step_in_epoch={resume_step_in_epoch}"
            )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        start_step = resume_step_in_epoch if epoch == start_epoch else 0
        sampler.set_epoch(epoch)
        sampler.set_start_index(start_step * args.per_device_batch_size)
        for step, batch in enumerate(dataloader, start=start_step + 1):
            batch = {
                key: value.to(device, non_blocking=True, dtype=torch.bfloat16)
                if key == "pixel_values"
                else value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == local_batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and global_step % args.logging_steps == 0:
                    avg_loss = running_loss / (args.logging_steps * args.gradient_accumulation_steps)
                    print(f"epoch={epoch} global_step={global_step} loss={avg_loss:.6f} lr={scheduler.get_last_lr()[0]:.3e}")
                    running_loss = 0.0

                if rank == 0 and args.save_steps and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    module = model.module if isinstance(model, DistributedDataParallel) else model
                    module.save_pretrained(ckpt_dir, safe_serialization=True, max_shard_size="2GB")
                    tokenizer.save_pretrained(ckpt_dir)
                    image_processor.save_pretrained(ckpt_dir)
                    patch_saved_config(ckpt_dir, image_token_id, args.image_seq_len)
                if args.save_steps and global_step % args.save_steps == 0:
                    ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_training_checkpoint_state(
                        ckpt_dir,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        global_step=global_step,
                        epoch=epoch,
                        step_in_epoch=step,
                        running_loss=running_loss,
                        rank=rank,
                        world_size=world_size,
                        extra={"args": vars(args)},
                    )
                    if world_size > 1:
                        dist.barrier()

    if rank == 0:
        module = model.module if isinstance(model, DistributedDataParallel) else model
        module.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(args.output_dir)
        image_processor.save_pretrained(args.output_dir)
        patch_saved_config(args.output_dir, image_token_id, args.image_seq_len)
        print(f"saved {args.output_dir} global_step={global_step}")
    save_training_checkpoint_state(
        args.output_dir,
        optimizer=optimizer,
        scheduler=scheduler,
        global_step=global_step,
        epoch=args.epochs,
        step_in_epoch=0,
        running_loss=running_loss,
        rank=rank,
        world_size=world_size,
        extra={"args": vars(args)},
    )
    if world_size > 1:
        dist.barrier()

    cleanup_distributed()


if __name__ == "__main__":
    main()
