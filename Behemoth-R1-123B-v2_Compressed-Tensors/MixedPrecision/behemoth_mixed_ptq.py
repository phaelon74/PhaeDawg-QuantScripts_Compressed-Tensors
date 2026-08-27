"""
Behemoth-R1-123B-v2 mixed W4A16 / W8A16 PTQ harness.

  - llm-compressor oneshot, compressed-tensors save
  - Algorithms: autoround | awq | gptq | awq_gptq
  - Optional W8 promotions via regex or down_proj layer indices
  - Packed-size preflight (rejects > --max-disk-gib)
  - Calibration from Recipes/Datasets/*.yaml

Run in the llm-compressor venv, not the vLLM KLD venv.

Examples:
  python behemoth_mixed_ptq.py SRC DST recipes/baseline_512.yaml --dry-run
  python behemoth_mixed_ptq.py SRC DST recipes/baseline_512.yaml \\
      --algorithm autoround --group-size 32 --autoround-batch-size 1
  python behemoth_mixed_ptq.py SRC DST recipes/baseline_512.yaml \\
      --algorithm awq --group-size 32 --asymmetric
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil

import torch.nn as nn
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import transformers.modeling_utils as _tmu

if not hasattr(_tmu, "TORCH_INIT_FUNCTIONS"):
    _tmu.TORCH_INIT_FUNCTIONS = {
        "uniform_": nn.init.uniform_,
        "normal_": nn.init.normal_,
        "trunc_normal_": nn.init.trunc_normal_,
        "constant_": nn.init.constant_,
        "xavier_uniform_": nn.init.xavier_uniform_,
        "xavier_normal_": nn.init.xavier_normal_,
        "kaiming_uniform_": nn.init.kaiming_uniform_,
        "kaiming_normal_": nn.init.kaiming_normal_,
        "uniform": nn.init.uniform,
        "normal": nn.init.normal,
        "xavier_uniform": nn.init.xavier_uniform,
        "xavier_normal": nn.init.xavier_normal,
        "kaiming_uniform": nn.init.kaiming_uniform,
        "kaiming_normal": nn.init.kaiming_normal,
    }

from compressed_tensors.offload import dispatch_model
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.modifiers.quantization import QuantizationModifier

from estimate_packed_size import (
    GIB,
    assign_bits,
    build_inventory,
    estimate,
    parse_layer_list,
)

SIDECAR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "chat_template.json",
    "generation_config.json",
)

IGNORE_ALWAYS = ["lm_head"]


def _import_awq():
    try:
        from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier

        return AWQModifier, AWQMapping, True
    except ImportError:
        from llmcompressor.modifiers.awq import AWQModifier

        try:
            from llmcompressor.modifiers.awq import AWQMapping
        except ImportError:
            AWQMapping = None
        return AWQModifier, AWQMapping, False


def mistral_awq_mappings(AWQMapping):
    if AWQMapping is None:
        return None
    return [
        AWQMapping(
            "re:.*input_layernorm$",
            ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
        ),
        AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
        AWQMapping(
            "re:.*post_attention_layernorm$",
            ["re:.*gate_proj$", "re:.*up_proj$"],
        ),
        AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
    ]


def copy_sidecars(source_dir: str, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    copied = []
    for filename in SIDECAR_FILES:
        src = os.path.join(source_dir, filename)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(save_dir, filename))
            copied.append(filename)
    try:
        for filename in os.listdir(source_dir):
            if filename.endswith(".jinja") and filename not in copied:
                shutil.copy2(
                    os.path.join(source_dir, filename),
                    os.path.join(save_dir, filename),
                )
                copied.append(filename)
    except OSError as e:
        print(f"WARNING: could not scan sidecars: {e}")
    if copied:
        print(f"Copied sidecars: {', '.join(copied)}")
    else:
        print(f"WARNING: no sidecars found under {source_dir}")


def messages_to_text(tokenizer, messages) -> str:
    if not messages:
        return ""
    norm = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = m.get("role", m.get("from", "user"))
        if role in ("human", "Human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "Assistant", "bot"):
            role = "assistant"
        elif role == "system":
            role = "system"
        content = m.get("content", m.get("value", m.get("text", "")))
        content = str(content).strip() if content is not None else ""
        if content:
            norm.append({"role": role, "content": content})
    if not norm:
        return ""
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                norm, tokenize=False, add_generation_prompt=False
            )
        except (ValueError, TypeError, RuntimeError):
            pass
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in norm)


def generate_assistant_mask(messages, tokenizer, max_seq_length):
    try:
        result = tokenizer.apply_chat_template(
            messages,
            return_assistant_tokens_mask=True,
            return_dict=True,
            add_special_tokens=False,
            max_length=max_seq_length,
            truncation=True,
        )
        mask = result.get("assistant_tokens_mask")
        if mask is not None:
            return list(mask)[:max_seq_length]
    except (TypeError, ValueError, KeyError):
        pass
    try:
        if not messages or messages[-1].get("role", "user") != "assistant":
            return [0] * max_seq_length
        prompt_messages = messages[:-1] + [{"role": "assistant", "content": ""}]
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        full_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            max_length=max_seq_length,
            truncation=True,
        )
        prompt_len = min(len(prompt_ids), len(full_ids), max_seq_length)
        seq_len = min(len(full_ids), max_seq_length)
        return [0] * prompt_len + [1] * (seq_len - prompt_len)
    except Exception:
        return [1] * max_seq_length


def _align_mask(mask, seq_len, pad=0):
    m = list(mask)[:seq_len]
    if len(m) < seq_len:
        m.extend([pad] * (seq_len - len(m)))
    return m


def format_sharegpt(example, columns, tokenizer, use_loss_mask=False):
    formatted = []
    if len(columns) >= 2 and "system" in columns[0].lower():
        system_prompt = example.get(columns[0], "")
        if system_prompt:
            formatted.append({"role": "system", "content": str(system_prompt)})
        conv_column = columns[1]
    else:
        conv_column = columns[0]
    messages = example.get(conv_column, [])
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            formatted.append({"role": "user", "content": messages})
            text = messages_to_text(tokenizer, formatted)
            if use_loss_mask:
                return {"text": text, "messages": json.dumps(formatted)}
            return {"text": text}
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", msg.get("from", "user"))
                content = msg.get("content", msg.get("value", ""))
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "bot"):
                    role = "assistant"
                elif role == "system":
                    role = "system"
                if content:
                    formatted.append({"role": role, "content": str(content)})
            elif isinstance(msg, str):
                idx = len([m for m in formatted if m["role"] != "system"])
                role = "user" if idx % 2 == 0 else "assistant"
                formatted.append({"role": role, "content": str(msg)})
    if not formatted:
        return {"text": "", "messages": ""} if use_loss_mask else {"text": ""}
    text = messages_to_text(tokenizer, formatted)
    if use_loss_mask:
        return {"text": text, "messages": json.dumps(formatted)}
    return {"text": text}


def format_prompt_answer(example, columns, tokenizer, use_loss_mask=False):
    prompt = example.get(columns[0], "")
    answer = example.get(columns[1], "") if len(columns) > 1 else ""
    messages = [
        {"role": "user", "content": str(prompt)},
        {"role": "assistant", "content": str(answer)},
    ]
    text = messages_to_text(tokenizer, messages)
    if use_loss_mask:
        return {"text": text, "messages": json.dumps(messages)}
    return {"text": text}


def format_rombo_reasoning(example, columns, tokenizer, use_loss_mask=False):
    """Match the original Behemoth GS32 instruction + multi-turn formatter."""
    instruction = example.get(columns[0], "")
    raw_inputs = example.get(columns[1], [])
    raw_outputs = example.get(columns[2], [])
    inputs = raw_inputs if isinstance(raw_inputs, list) else [raw_inputs]
    outputs = raw_outputs if isinstance(raw_outputs, list) else [raw_outputs]

    messages = [{"role": "system", "content": str(instruction)}]
    for i in range(max(len(inputs), len(outputs))):
        if i < len(inputs) and inputs[i]:
            messages.append({"role": "user", "content": str(inputs[i])})
        if i < len(outputs) and outputs[i]:
            messages.append({"role": "assistant", "content": str(outputs[i])})

    text = messages_to_text(tokenizer, messages)
    if use_loss_mask:
        return {"text": text, "messages": json.dumps(messages)}
    return {"text": text}


def format_chat_completion(example, columns, tokenizer, use_loss_mask=False):
    for col in columns:
        if col not in example:
            continue
        data = example[col]
        if isinstance(data, list) and data:
            if isinstance(data[0], dict):
                text = messages_to_text(tokenizer, data)
                if use_loss_mask:
                    return {"text": text, "messages": json.dumps(data)}
                return {"text": text}
            messages = []
            for i, item in enumerate(data):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": str(item)})
            text = messages_to_text(tokenizer, messages)
            if use_loss_mask:
                return {"text": text, "messages": json.dumps(messages)}
            return {"text": text}
        if isinstance(data, str):
            if use_loss_mask:
                return {"text": str(data), "messages": ""}
            return {"text": str(data)}
    text = " ".join(str(example.get(col, "")) for col in columns)
    if use_loss_mask:
        return {"text": text, "messages": ""}
    return {"text": text}


def format_raw_text(example, columns, _tokenizer, use_loss_mask=False):
    texts = [str(example[col]) for col in columns if col in example and example[col]]
    text = " ".join(texts)
    if use_loss_mask:
        return {"text": text, "messages": ""}
    return {"text": text}


FORMATTERS = {
    "sharegpt": format_sharegpt,
    "prompt_answer": format_prompt_answer,
    "rombo_reasoning": format_rombo_reasoning,
    "chat_completion": format_chat_completion,
    "raw_text": format_raw_text,
}


def load_calibration(recipe_yaml, tokenizer, seed, shuffle, max_seq_length, use_loss_mask):
    with open(recipe_yaml, "r", encoding="utf-8") as f:
        recipe_file = yaml.safe_load(f)
    calibration_config = recipe_file.get("calibration_set", {})
    if max_seq_length is None:
        max_seq_length = calibration_config["max_seq_length"]
    if shuffle is None:
        shuffle = calibration_config.get("shuffle", True)
    if seed is None:
        seed = calibration_config.get("seed", 42)
    datasets_config = calibration_config.get("datasets", [])

    print(f"Loaded calibration recipe: {recipe_yaml}")
    print(f"  max_seq_length={max_seq_length} shuffle={shuffle} seed={seed}")

    all_parts = []
    for ds_config in datasets_config:
        dataset_name = ds_config["dataset"]
        split = ds_config.get("split", "train")
        subset = ds_config.get("subset")
        columns = ds_config.get("columns", [])
        formatter_name = ds_config.get("formatter", "raw_text")
        num_samples = ds_config.get("num_samples", 10)
        streaming = ds_config.get("streaming", False)
        dataset_seed = int(ds_config.get("seed", seed))
        load_kw = {}
        if subset:
            load_kw["name"] = subset
        print(
            f"  {dataset_name} split={split} subset={subset!r} "
            f"n={num_samples} fmt={formatter_name}"
        )
        try:
            if streaming:
                stream = load_dataset(
                    dataset_name, split=split, streaming=True, **load_kw
                )
                part = Dataset.from_list(list(stream.take(num_samples)))
            else:
                part = load_dataset(dataset_name, split=split, **load_kw)
                n = min(num_samples, len(part))
                part = part.shuffle(seed=dataset_seed).select(range(n))
            formatter_fn = FORMATTERS.get(formatter_name, format_raw_text)
            part = part.map(
                lambda x, c=columns, t=tokenizer, u=use_loss_mask, f=formatter_fn: f(
                    x, c, t, u
                ),
                remove_columns=part.column_names,
                num_proc=1,
            )
            part = part.filter(lambda x: len(x.get("text", "")) > 0)
            all_parts.append(part)
            print(f"    -> {len(part)} rows")
        except Exception as e:
            print(f"    -> WARNING skipped ({e})")
            continue

    if not all_parts:
        raise ValueError("No calibration datasets loaded.")
    ds = concatenate_datasets(all_parts)
    if shuffle:
        ds = ds.shuffle(seed=seed)

    def tokenize_with_mask(batch):
        result = tokenizer(
            batch["text"],
            padding=False,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False,
        )
        if use_loss_mask:
            loss_masks = []
            for i, messages_json in enumerate(batch["messages"]):
                seq_len = len(result["input_ids"][i])
                if not messages_json:
                    loss_masks.append([1] * seq_len)
                else:
                    mask = generate_assistant_mask(
                        json.loads(messages_json), tokenizer, max_seq_length
                    )
                    loss_masks.append(_align_mask(mask, seq_len))
            result["loss_mask"] = loss_masks
        return result

    ds = ds.map(
        tokenize_with_mask,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=1 if use_loss_mask else 4,
    )
    print(f"Tokenized {len(ds)} calibration samples")
    return ds, max_seq_length


def weight_args(num_bits: int, group_size: int, symmetric: bool) -> QuantizationArgs:
    return QuantizationArgs(
        num_bits=num_bits,
        type="int",
        symmetric=symmetric,
        strategy="group",
        group_size=group_size,
    )


def groups_from_bits(inv, bits_map, group_size: int, symmetric: bool):
    """Non-overlapping explicit module names. BF16 Linears go on the ignore list."""
    w4_names = [m.name for m in inv.linears if bits_map[m.name] == 4]
    w8_names = [m.name for m in inv.linears if bits_map[m.name] == 8]
    bf16_names = [m.name for m in inv.linears if bits_map[m.name] == 16]
    groups = {}
    if w8_names:
        groups["w8"] = QuantizationScheme(
            targets=w8_names,
            weights=weight_args(8, group_size, True),
            input_activations=None,
            output_activations=None,
        )
    if w4_names:
        groups["w4"] = QuantizationScheme(
            targets=w4_names if w8_names or bf16_names else ["Linear"],
            weights=weight_args(4, group_size, symmetric),
            input_activations=None,
            output_activations=None,
        )
    ignore = list(IGNORE_ALWAYS) + bf16_names
    return groups, ignore


def build_recipe(args, inv, bits_map):
    groups, ignore = groups_from_bits(
        inv, bits_map, args.group_size, not args.asymmetric
    )
    if args.algorithm == "autoround":
        from packaging.version import Version

        installed = importlib.metadata.version("auto-round")
        if Version(installed) < Version("0.13.0"):
            raise RuntimeError(
                "AutoRound >= 0.13.0 is required for W4A16 compressed-tensors "
                f"export; found {installed}."
            )
        print(f"AutoRound version: {installed}")
        from llmcompressor.modifiers.autoround import AutoRoundModifier

        return [
            AutoRoundModifier(
                ignore=ignore,
                config_groups=groups,
                iters=args.autoround_iters,
                batch_size=args.autoround_batch_size,
                lr=args.autoround_lr,
                device_ids=args.autoround_device_ids,
                enable_torch_compile=not args.disable_torch_compile,
                disable_opt_rtn=args.autoround_disable_opt_rtn,
            )
        ]

    gptq_kw = dict(
        ignore=ignore,
        config_groups=groups,
        dampening_frac=args.dampening_frac,
        actorder=args.actorder,
        block_size=args.block_size,
        offload_hessians=args.offload_hessians,
    )
    if args.algorithm == "gptq":
        return [GPTQModifier(**gptq_kw)]

    AWQModifier, AWQMapping, is_transform = _import_awq()
    mappings = mistral_awq_mappings(AWQMapping)
    awq_kw = {}
    if mappings is not None:
        awq_kw["mappings"] = mappings
    # 0.13 transform API uses duo_scaling="both"; older API used bool.
    awq_kw["duo_scaling"] = "both" if is_transform else True

    if args.algorithm == "awq":
        if is_transform:
            return [
                AWQModifier(**awq_kw),
                QuantizationModifier(ignore=ignore, config_groups=groups),
            ]
        return [
            AWQModifier(ignore=ignore, config_groups=groups, **awq_kw),
        ]
    # awq_gptq
    if is_transform:
        return [AWQModifier(**awq_kw), GPTQModifier(**gptq_kw)]
    return [
        AWQModifier(ignore=ignore, config_groups=groups, **awq_kw),
        GPTQModifier(**gptq_kw),
    ]


def load_policy_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def parse_args():
    parser = argparse.ArgumentParser(
        description="Behemoth mixed W4A16/W8A16 PTQ (llm-compressor)."
    )
    parser.add_argument("model_path", help="BF16 source model directory.")
    parser.add_argument("output_path", help="Destination for compressed checkpoint.")
    parser.add_argument("recipe_yaml", help="Calibration YAML (calibration_set).")
    parser.add_argument(
        "--policy-yaml",
        default=None,
        help="Optional precision policy (group_size, w8_regex, promote layers).",
    )
    parser.add_argument(
        "--algorithm",
        choices=("autoround", "awq", "gptq", "awq_gptq"),
        default=None,
    )
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=None)
    parser.add_argument("--asymmetric", action="store_true")
    parser.add_argument("--w8-regex", action="append", default=None)
    parser.add_argument("--bf16-regex", action="append", default=None)
    parser.add_argument("--promote-down-proj-layers", default=None)
    parser.add_argument("--dampening-frac", type=float, default=0.01)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--actorder", default="static")
    parser.add_argument("--offload-hessians", action="store_true")
    parser.add_argument("--autoround-iters", type=int, default=200)
    parser.add_argument("--autoround-batch-size", type=int, default=1)
    parser.add_argument("--autoround-lr", type=float, default=None)
    parser.add_argument("--autoround-device-ids", default="auto")
    parser.add_argument("--autoround-disable-opt-rtn", action="store_true")
    parser.add_argument("--disable-torch-compile", action="store_true")
    parser.add_argument("--use-loss-mask", action="store_true")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-disk-gib", type=float, default=70.0)
    parser.add_argument("--dry-run", action="store_true", help="Size estimate only.")
    parser.add_argument("--skip-sample-gen", action="store_true")
    parser.add_argument(
        "--sequential-targets",
        default=None,
        help="Optional oneshot sequential_targets (e.g. MistralMLP).",
    )
    return parser.parse_args()


def merge_policy(args, policy: dict):
    if args.algorithm is None:
        args.algorithm = policy.get("algorithm", "gptq")
    if args.group_size is None:
        args.group_size = int(policy.get("group_size", 32))
    if not args.asymmetric:
        args.asymmetric = bool(policy.get("asymmetric", False))
    if args.w8_regex is None:
        args.w8_regex = list(policy.get("w8_regex", []) or [])
    if args.bf16_regex is None:
        args.bf16_regex = list(policy.get("bf16_regex", []) or [])
    if args.promote_down_proj_layers is None:
        layers = policy.get("promote_down_proj_layers", [])
        if isinstance(layers, list):
            args.promote_down_proj_layers = ",".join(str(x) for x in layers)
        else:
            args.promote_down_proj_layers = str(layers or "")
    return args


def main() -> int:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args = parse_args()
    args = merge_policy(args, load_policy_yaml(args.policy_yaml))

    promote_layers = parse_layer_list(args.promote_down_proj_layers)

    inv = build_inventory(args.model_path)
    bits_map = assign_bits(
        inv,
        w8_regexes=args.w8_regex or [],
        promote_down_proj_layers=promote_layers,
        bf16_regexes=args.bf16_regex or [],
    )
    size = estimate(
        inv,
        bits_map,
        group_size=args.group_size,
        symmetric=not args.asymmetric,
    )
    total_gib = size["total_bytes"] / GIB
    print(
        f"Preflight: {total_gib:.2f} GiB  "
        f"(W8 params {size['params_w8']:,}, extra vs W4 "
        f"{size['extra_vs_uniform_w4_bytes'] / GIB:.2f} GiB)  "
        f"algorithm={args.algorithm} gs={args.group_size}"
    )
    if total_gib > args.max_disk_gib:
        print(
            f"REJECT: estimated {total_gib:.2f} GiB > {args.max_disk_gib:.1f} GiB budget"
        )
        return 2
    if args.dry_run:
        print("Dry run only; not quantizing.")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    ds, max_seq_length = load_calibration(
        args.recipe_yaml,
        tokenizer,
        seed=None,
        shuffle=None,
        max_seq_length=args.max_seq_length,
        use_loss_mask=args.use_loss_mask,
    )

    try:
        source_generation_config = GenerationConfig.from_pretrained(
            args.model_path, trust_remote_code=True
        )
    except Exception as e:
        source_generation_config = None
        print(f"WARNING: could not load GenerationConfig: {e}")

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype="auto",
        trust_remote_code=True,
        local_files_only=True,
    )

    recipe = build_recipe(args, inv, bits_map)
    print(f"Recipe modifiers: {[type(m).__name__ for m in recipe]}")
    oneshot_kwargs = dict(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=len(ds),
        tokenizer=tokenizer,
        use_loss_mask=args.use_loss_mask,
    )
    if args.algorithm == "autoround":
        # Official AutoRound examples disable this for slightly better recovery.
        oneshot_kwargs["shuffle_calibration_samples"] = False
    if args.use_loss_mask:
        oneshot_kwargs["pipeline"] = "sequential"
    if args.sequential_targets:
        oneshot_kwargs["sequential_targets"] = [
            t.strip() for t in args.sequential_targets.split(",") if t.strip()
        ]
    oneshot(**oneshot_kwargs)

    if not args.skip_sample_gen:
        print("\n========== SAMPLE GENERATION ==============")
        dispatch_model(model)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello my name is"}],
            tokenize=False,
            add_generation_prompt=True,
        ) if getattr(tokenizer, "chat_template", None) else "Hello my name is"
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(**inputs, max_new_tokens=64)
        print(tokenizer.decode(out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True))
        print("==========================================\n")

    if source_generation_config is not None:
        model.generation_config = source_generation_config

    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path, save_compressed=True)
    tokenizer.save_pretrained(args.output_path)
    copy_sidecars(args.model_path, args.output_path)
    print(f"Saved compressed model to {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
