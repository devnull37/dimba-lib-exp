#!/usr/bin/env python3
"""GRPO fine-tuning for DIMBA."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = (SCRIPT_DIR / ".." / ".." / "src").resolve()
sys.path.insert(0, str(SRC_DIR))

from dimba import DIMBA
from dimba.models.lora import inject_lora_to_model, save_lora_weights
from dimba.models.quantization import prepare_for_qlora, quantize_model_4bit
from dimba.tokenizers import BPETokenizer, SimpleCharacterTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_dtype(name: str) -> torch.dtype:
    m = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    if name not in m:
        raise ValueError(f"Unsupported dtype: {name}")
    return m[name]


def _strip(k: str) -> str:
    for p in ("module.", "model.", "ema_model."):
        while k.startswith(p):
            k = k[len(p) :]
    return k


def extract_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    keys = list(state.keys())
    if any(k.startswith("model.") for k in keys):
        keys = [k for k in keys if k.startswith("model.")]
    elif any(k.startswith("ema_model.") for k in keys):
        keys = [k for k in keys if k.startswith("ema_model.")]
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        out[_strip(k)] = state[k]
    return out


def infer_model_cfg(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    emb = state.get("token_embed.embedding.weight")
    if emb is None:
        raise ValueError("Missing token_embed.embedding.weight in checkpoint")
    vocab_size, d_model = int(emb.shape[0]), int(emb.shape[1])
    cfg: Dict[str, Any] = {"vocab_size": vocab_size, "d_model": d_model, "d_prompt": d_model}

    idxs = []
    for k in state:
        m = re.match(r"prompt_encoder\.mlp\.(\d+)\.weight", k)
        if m:
            idxs.append(int(m.group(1)))
    if idxs:
        w = state.get(f"prompt_encoder.mlp.{max(idxs)}.weight")
        if w is not None:
            cfg["d_prompt"] = int(w.shape[0])

    blocks = []
    for k in state:
        m = re.match(r"denoiser\.blocks\.(\d+)\.", k)
        if m:
            blocks.append(int(m.group(1)))
    cfg["num_denoiser_layers"] = max(blocks) + 1 if blocks else 6
    cfg["d_state"] = int(state["denoiser.blocks.0.mamba.A"].shape[-1]) if "denoiser.blocks.0.mamba.A" in state else 16
    if "denoiser.blocks.0.mamba.in_proj.weight" in state:
        d_inner = int(state["denoiser.blocks.0.mamba.in_proj.weight"].shape[0] // 2)
        cfg["expand"] = max(1, int(round(d_inner / float(d_model))))
    else:
        cfg["expand"] = 2
    cfg["d_conv"] = 4
    cfg["conditioning_type"] = "film" if any("gamma_proj" in k for k in state) else "additive"
    cfg["dropout"] = 0.1
    cfg["use_weight_tying"] = False
    cfg["use_simple_mamba"] = any(".mamba.B_proj." in k for k in state)
    return cfg


CFG_BASENAMES = ("train_config.yaml", "config.yaml", "hparams.yaml")
ARTIFACT_DIR_HINTS = ("checkpoints", "checkpoint", "ckpts", "artifacts", "artifact", "weights", "models")


def _looks_like_artifact_dir(path: Path) -> bool:
    name = path.name.lower()
    return name in ARTIFACT_DIR_HINTS or "checkpoint" in name or name.startswith("ckpt") or "artifact" in name


def _load_model_cfg_from_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    if not isinstance(y, dict):
        return {}
    cfg: Dict[str, Any] = {}
    if isinstance(y.get("model"), dict):
        cfg.update(y["model"])
    elif "vocab_size" in y:
        cfg.update(y)
    tok_cfg = y.get("tokenizer")
    if isinstance(tok_cfg, dict) and tok_cfg.get("vocab_size") is not None:
        cfg.setdefault("vocab_size", int(tok_cfg["vocab_size"]))
    return cfg


def discover_cfg(checkpoint: Path, explicit_cfg: Optional[str]) -> Optional[Path]:
    if explicit_cfg:
        p = Path(explicit_cfg)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Config file not found: {explicit_cfg}")
        return p

    candidates = [checkpoint.with_suffix(".yaml"), checkpoint.with_suffix(".yml")]

    search_dirs = [checkpoint.parent]
    if _looks_like_artifact_dir(checkpoint.parent):
        search_dirs.append(checkpoint.parent.parent)
    seen_dirs = set()
    for d in search_dirs:
        rd = d.resolve()
        if rd in seen_dirs:
            continue
        seen_dirs.add(rd)
        for name in CFG_BASENAMES:
            candidates.append(d / name)

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def resolve_checkpoint(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(path_str)
    if (p / "last.ckpt").exists():
        return p / "last.ckpt"
    files = list(p.glob("*.ckpt")) + list(p.glob("*.pt")) + list(p.glob("*.pth"))
    if not files:
        raise FileNotFoundError(f"No checkpoint file in {p}")
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]


def load_policy_model(checkpoint: Path, device: torch.device, explicit_cfg: Optional[str]) -> Tuple[DIMBA, Dict[str, Any]]:
    raw = torch.load(checkpoint, map_location="cpu")
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint must be a dict")
    state = extract_state(raw)
    cfg = infer_model_cfg(state)

    cfg_file = discover_cfg(checkpoint, explicit_cfg)
    if cfg_file is not None:
        # External YAML is treated as fallback only; checkpoint metadata/state stays authoritative.
        file_cfg = _load_model_cfg_from_yaml(cfg_file)
        for k, v in file_cfg.items():
            cfg.setdefault(k, v)

    top_cfg = raw.get("model_config", {})
    if isinstance(top_cfg, dict):
        cfg.update(top_cfg)

    hp = raw.get("hyper_parameters", {})
    if isinstance(hp, dict):
        hp_model_cfg = hp.get("model_config")
        if isinstance(hp_model_cfg, dict):
            cfg.update(hp_model_cfg)
        if raw.get("vocab_size") is not None:
            cfg["vocab_size"] = int(raw["vocab_size"])
        if hp.get("vocab_size") is not None:
            cfg["vocab_size"] = int(hp["vocab_size"])
    elif raw.get("vocab_size") is not None:
        cfg["vocab_size"] = int(raw["vocab_size"])

    valid = set(inspect.signature(DIMBA.__init__).parameters)
    valid.discard("self")
    cfg = {k: v for k, v in cfg.items() if k in valid}

    model = DIMBA(**cfg).to(device)
    model.load_state_dict(state, strict=False)
    return model, cfg


def load_tokenizer(checkpoint: Path, vocab_size: int, tokenizer_path: Optional[str]) -> Tuple[Any, Optional[Path]]:
    candidates = [Path(tokenizer_path)] if tokenizer_path else []
    candidates += [checkpoint.parent / "tokenizer.json", checkpoint.parent.parent / "tokenizer.json"]
    for p in candidates:
        if not p.exists():
            continue
        try:
            txt = p.read_text(encoding="utf-8")[:1024]
            if "char_to_id" in txt:
                tok = SimpleCharacterTokenizer(vocab_size=vocab_size)
                tok.load(str(p))
                return tok, p
            tok = BPETokenizer(vocab_size=vocab_size)
            tok.load(str(p))
            return tok, p
        except Exception:
            pass
    return SimpleCharacterTokenizer(vocab_size=vocab_size), None


def longest_common_prefix(a: str, b: str) -> str:
    i = 0
    n = min(len(a), len(b))
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]


def normalize_row(row: Dict[str, Any]) -> Optional[Tuple[str, str, str]]:
    p_keys = ("prompt", "input", "instruction", "question", "query")
    c_keys = ("chosen", "preferred", "response_chosen", "chosen_response", "winner")
    r_keys = ("rejected", "response_rejected", "rejected_response", "loser")
    p = next((str(row[k]) for k in p_keys if k in row and row[k] is not None), None)
    c = next((str(row[k]) for k in c_keys if k in row and row[k] is not None), None)
    r = next((str(row[k]) for k in r_keys if k in row and row[k] is not None), None)
    if c is None or r is None:
        return None
    if p is not None:
        return p, c, r
    pref = longest_common_prefix(c, r)
    return pref, c[len(pref) :], r[len(pref) :]


def _load_rows_with_repo_finetuning(dataset: str, split: str, max_samples: Optional[int]) -> Optional[List[Tuple[str, str, str]]]:
    try:
        from dimba.data.finetuning import load_and_format_finetuning_records
    except Exception:
        return None

    try:
        rows, _ = load_and_format_finetuning_records(
            source=dataset,
            split=split,
            max_examples=max_samples,
            strict=False,
        )
    except Exception:
        return None

    out: List[Tuple[str, str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = normalize_row(row)
        if item is None:
            continue
        prompt, chosen, rejected = item
        if chosen.strip() and rejected.strip():
            out.append((prompt, chosen, rejected))
        if max_samples is not None and len(out) >= max_samples:
            break
    return out or None


def load_rows(dataset: str, split: str, max_samples: Optional[int]) -> List[Tuple[str, str, str]]:
    repo_rows = _load_rows_with_repo_finetuning(dataset, split, max_samples)
    if repo_rows is not None:
        return repo_rows

    p = Path(dataset)
    raw_rows: List[Dict[str, Any]] = []
    if p.exists():
        if p.suffix.lower() == ".jsonl":
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    raw_rows.append(json.loads(line))
        elif p.suffix.lower() == ".json":
            obj = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                raw_rows = obj
            elif isinstance(obj, dict):
                raw_rows = obj.get(split, obj.get("train", []))
            else:
                raise ValueError("Unsupported json dataset structure")
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError(
                    "The 'datasets' package is required for this dataset format. "
                    "Install with: pip install datasets"
                ) from exc
            ds = load_dataset("json", data_files=str(p), split="train")
            raw_rows = [dict(x) for x in ds]
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for Hugging Face datasets. "
                "Install with: pip install datasets"
            ) from exc
        ds = load_dataset(dataset, split=split)
        raw_rows = [dict(x) for x in ds]

    out: List[Tuple[str, str, str]] = []
    for row in raw_rows:
        item = normalize_row(row)
        if item is None:
            continue
        prompt, chosen, rejected = item
        if chosen.strip() and rejected.strip():
            out.append((prompt, chosen, rejected))
        if max_samples is not None and len(out) >= max_samples:
            break
    if not out:
        raise ValueError("No valid preference rows found")
    return out


def parse_lora_target_modules(raw: str, model: nn.Module) -> List[str]:
    requested = [part.strip() for part in raw.split(",") if part.strip()]
    wants_all = (not requested) or any(part.lower() == "all-linear" for part in requested)
    linear_names = [name for name, module in model.named_modules() if name and isinstance(module, nn.Linear)]

    expanded: List[str] = []
    if wants_all:
        expanded.extend(linear_names)
    for part in requested:
        if part.lower() == "all-linear":
            continue
        matches = [name for name in linear_names if part in name]
        expanded.extend(matches if matches else [part])

    deduped: List[str] = []
    seen = set()
    for module_name in expanded:
        if module_name in seen:
            continue
        seen.add(module_name)
        deduped.append(module_name)

    if not deduped:
        raise ValueError("No target linear modules for LoRA")
    return deduped


def tok(tokenizer: Any, text: str, max_len: int, eos: Optional[int]) -> List[int]:
    ids = tokenizer.encode(text)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    ids = [int(x) for x in ids[:max_len]]
    if eos is not None and len(ids) < max_len and (not ids or ids[-1] != eos):
        ids.append(eos)
    return ids or ([eos] if eos is not None else [0])


class PrefDataset(Dataset):
    def __init__(self, rows: Sequence[Tuple[str, str, str]], tokenizer: Any, max_prompt_len: int, max_new_tokens: int, eos: Optional[int]) -> None:
        self.data = [(tok(tokenizer, p, max_prompt_len, eos), tok(tokenizer, c, max_new_tokens, eos), tok(tokenizer, r, max_new_tokens, eos)) for p, c, r in rows]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], List[int]]:
        return self.data[idx]


def collate_pref(batch: Sequence[Tuple[List[int], List[int], List[int]]], pad: int) -> Dict[str, Any]:
    plens = [len(x[0]) for x in batch]
    pmax = max(plens) if plens else 1
    p = torch.full((len(batch), pmax), pad, dtype=torch.long)
    c: List[List[int]] = []
    r: List[List[int]] = []
    for i, (pi, ci, ri) in enumerate(batch):
        p[i, : len(pi)] = torch.tensor(pi, dtype=torch.long)
        c.append(ci)
        r.append(ri)
    return {"prompt_ids": p, "prompt_lens": torch.tensor(plens, dtype=torch.long), "chosen_ids": c, "rejected_ids": r}


def strip_special(ids: Sequence[int], pad: int, eos: Optional[int]) -> List[int]:
    out: List[int] = []
    for x in ids:
        if x == pad:
            continue
        if eos is not None and x == eos:
            break
        out.append(int(x))
    return out


def token_f1(a: Sequence[int], b: Sequence[int]) -> float:
    if not a or not b:
        return 0.0
    ca, cb = Counter(a), Counter(b)
    ov = sum((ca & cb).values())
    if ov == 0:
        return 0.0
    p, r = ov / len(a), ov / len(b)
    return 2 * p * r / max(1e-8, p + r)


def bigram_prec(a: Sequence[int], b: Sequence[int]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ca = Counter(tuple(a[i : i + 2]) for i in range(len(a) - 1))
    cb = Counter(tuple(b[i : i + 2]) for i in range(len(b) - 1))
    return sum((ca & cb).values()) / max(1, sum(ca.values()))


def reward_fn(pred: Sequence[int], chosen: Sequence[int], rejected: Sequence[int], pad: int, eos: Optional[int]) -> float:
    p = strip_special(pred, pad, eos)
    c = strip_special(chosen, pad, eos)
    r = strip_special(rejected, pad, eos)
    if not p:
        return -0.25
    sc = 0.7 * token_f1(p, c) + 0.3 * bigram_prec(p, c)
    sr = 0.7 * token_f1(p, r) + 0.3 * bigram_prec(p, r)
    return sc - sr


def top_k_top_p(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        v = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < v, -float("inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        s_logits, s_idx = torch.sort(logits, descending=True, dim=-1)
        cp = torch.softmax(s_logits, dim=-1).cumsum(dim=-1)
        rm = cp > top_p
        rm[..., 0] = 0
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(dim=-1, index=s_idx, src=rm)
        logits = logits.masked_fill(mask, -float("inf"))
    return logits


@torch.no_grad()
def generate_quiet(model: DIMBA, prompt_ids: torch.Tensor, seq_len: int, num_steps: int, temperature: float, top_k: Optional[int], top_p: Optional[float], device: torch.device) -> torch.Tensor:
    model.eval()
    prompt_ids = prompt_ids.to(device)
    bsz = prompt_ids.shape[0]
    cond = model.encode_prompt(prompt_ids)
    if cond.shape[1] < seq_len:
        pad = torch.zeros(bsz, seq_len - cond.shape[1], cond.shape[2], device=device)
        cond = torch.cat([cond, pad], dim=1)
    else:
        cond = cond[:, :seq_len, :]
    cond = model.project_conditioning(cond)
    x = torch.randn(bsz, seq_len, model.d_latent, device=device)
    alphas = model.get_alphas_cumprod().to(device)
    ts = torch.linspace(model.num_diffusion_steps - 1, 0, max(1, int(num_steps)), dtype=torch.long, device=device)
    for i, t_cont in enumerate(ts):
        t = torch.full((bsz,), int(t_cont.item()), dtype=torch.long, device=device)
        xp = model.denoise_step(x, t, cond)
        if i < len(ts) - 1:
            t_prev = ts[i + 1].long()
            a_t, a_prev = alphas[t], alphas[t_prev]
            sigma = torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).view(-1, 1, 1)
            x = (xp + sigma * torch.randn_like(x)) * torch.sqrt(a_prev / a_t).view(-1, 1, 1)
        else:
            x = xp
    logits = model.output_head(model.decode_latent(x), embedding_weight=model.token_embed.get_weight()) / max(1e-6, temperature)
    probs = torch.softmax(top_k_top_p(logits, top_k, top_p), dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    s = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(s > 1e-6, probs / s, torch.ones_like(probs) / probs.shape[-1])
    return torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(bsz, seq_len)


def build_eval_inputs(prompt_ids: torch.Tensor, prompt_lens: torch.Tensor, generated: torch.Tensor, num_generations: int, max_new_tokens: int, max_seq_len: int, pad: int) -> Tuple[torch.Tensor, torch.Tensor, List[List[int]]]:
    seqs, masks, comps = [], [], []
    p_cpu, l_cpu, g_cpu = prompt_ids.cpu(), prompt_lens.cpu(), generated.cpu()
    for i in range(generated.shape[0]):
        b = i // num_generations
        plen = int(l_cpu[b].item())
        prompt = p_cpu[b, :plen].tolist()
        gen = g_cpu[i].tolist()
        comp = gen[min(plen, len(gen)) : min(plen + max_new_tokens, len(gen))]
        comps.append(comp)
        full = (prompt + comp)[:max_seq_len]
        m = [0] * min(len(prompt), len(full)) + [1] * max(0, len(full) - len(prompt))
        seqs.append(full)
        masks.append(m)
    mx = max((len(x) for x in seqs), default=1)
    inp = torch.full((len(seqs), mx), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), mx), dtype=torch.float32)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        if s:
            inp[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(m)] = torch.tensor(m, dtype=torch.float32)
    return inp, mask, comps


def model_logits(model: DIMBA, input_ids: torch.Tensor) -> torch.Tensor:
    t = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
    x_pred, _, _ = model(input_ids, t, return_latent_info=True)
    return model.output_head(x_pred, embedding_weight=model.token_embed.get_weight())


def grpo_loss(policy_logits: torch.Tensor, ref_logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor, adv: torch.Tensor, beta: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    p_lp = torch.log_softmax(policy_logits, dim=-1)
    r_lp = torch.log_softmax(ref_logits, dim=-1)
    tok_lp = torch.gather(p_lp, -1, input_ids.unsqueeze(-1)).squeeze(-1)
    den = mask.sum(dim=1).clamp_min(1.0)
    seq_lp = (tok_lp * mask).sum(dim=1) / den
    kl_tok = (p_lp.exp() * (p_lp - r_lp)).sum(dim=-1)
    seq_kl = (kl_tok * mask).sum(dim=1) / den
    obj = adv * seq_lp - beta * seq_kl
    loss = -obj.mean()
    return loss, {"loss": float(loss.item()), "logp": float(seq_lp.mean().item()), "kl": float(seq_kl.mean().item()), "adv_abs": float(adv.abs().mean().item())}


def save_state(
    output_dir: Path,
    name: str,
    model: DIMBA,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    step: int,
    epoch: int,
    model_cfg: Dict[str, Any],
    tokenizer: Any,
    save_lora_adapter: bool = False,
) -> Path:
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    path = output_dir / "checkpoints" / name
    torch.save({"policy_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": step, "epoch": epoch, "model_config": model_cfg, "args": vars(args)}, path)
    if save_lora_adapter:
        adapter_path = path.with_name(f"{path.stem}_lora.pt")
        save_lora_weights(model, adapter_path)
    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    try:
        tokenizer.save(str(output_dir / "tokenizer.json"))
    except Exception:
        pass
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO fine-tune for DIMBA")
    p.add_argument("--base-checkpoint", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--output-dir", type=str, default="checkpoints/grpo")
    p.add_argument("--dataset-split", type=str, default="train")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampling-steps", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--use-qlora", action="store_true")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-target-modules", type=str, default="all-linear")
    p.add_argument("--qlora-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--bnb-4bit-quant-type", type=str, default="nf4")
    p.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = resolve_checkpoint(args.base_checkpoint)
    policy, model_cfg = load_policy_model(ckpt, device, args.config)
    ref = copy.deepcopy(policy).to(device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False

    vocab = int(model_cfg.get("vocab_size", policy.vocab_size))
    tok_obj, tok_path = load_tokenizer(ckpt, vocab, args.tokenizer_path)
    print(f"[init] checkpoint={ckpt}")
    print(f"[init] tokenizer={tok_path if tok_path else 'SimpleCharacterTokenizer(fallback)'}")

    use_lora = bool(args.use_lora or args.use_qlora)
    if use_lora:
        target_modules = parse_lora_target_modules(args.lora_target_modules, policy)
        wrapped_modules = inject_lora_to_model(
            policy,
            target_modules=target_modules,
            r=int(args.lora_r),
            alpha=float(args.lora_alpha),
            dropout=float(args.lora_dropout),
        )
        if args.use_qlora:
            _ = parse_dtype(args.qlora_dtype)
            policy = quantize_model_4bit(policy)
            policy = prepare_for_qlora(policy)
        print(f"[init] {'Q-LoRA' if args.use_qlora else 'LoRA'} enabled. wrapped_linear_modules={len(wrapped_modules)}")
    else:
        for p in policy.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    total = sum(p.numel() for p in policy.parameters())
    print(f"[init] trainable={trainable:,}/{total:,}")

    pad = int(getattr(tok_obj, "pad_token_id", 0))
    eos = int(getattr(tok_obj, "eos_token_id", pad))
    rows = load_rows(args.dataset, args.dataset_split, args.max_samples)
    max_prompt = max(1, int(args.max_seq_len) - int(args.max_new_tokens))
    ds = PrefDataset(rows, tok_obj, max_prompt, int(args.max_new_tokens), eos)
    print(f"[data] samples={len(ds):,}")
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=bool(args.shuffle), num_workers=int(args.num_workers), collate_fn=lambda b: collate_pref(b, pad))

    params = [p for p in policy.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found")
    opt = torch.optim.AdamW(params, lr=float(args.learning_rate), weight_decay=float(args.weight_decay))

    step = 0
    stop = False
    for epoch in range(int(args.epochs)):
        e_loss, e_reward, e_steps = 0.0, 0.0, 0
        for batch in dl:
            if args.max_steps > 0 and step >= args.max_steps:
                stop = True
                break
            prompt_ids = batch["prompt_ids"].to(device)
            prompt_lens = batch["prompt_lens"].to(device)
            chosen, rejected = batch["chosen_ids"], batch["rejected_ids"]
            bsz = prompt_ids.shape[0]

            with torch.no_grad():
                reps = prompt_ids.repeat_interleave(args.num_generations, dim=0)
                seq_len = min(int(args.max_seq_len), int(prompt_ids.shape[1]) + int(args.max_new_tokens))
                gen = generate_quiet(policy, reps, seq_len, args.sampling_steps, args.temperature, args.top_k, args.top_p, device)
                eval_ids, comp_mask, comps = build_eval_inputs(prompt_ids, prompt_lens, gen, args.num_generations, args.max_new_tokens, args.max_seq_len, pad)
                rewards = torch.zeros(bsz, args.num_generations, dtype=torch.float32)
                for bi in range(bsz):
                    for gi in range(args.num_generations):
                        idx = bi * args.num_generations + gi
                        rewards[bi, gi] = reward_fn(comps[idx], chosen[bi], rejected[bi], pad, eos)
                adv = (rewards - rewards.mean(dim=1, keepdim=True)) / rewards.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
                adv = adv.reshape(-1).to(device)

            eval_ids = eval_ids.to(device)
            comp_mask = comp_mask.to(device)

            policy.train()
            opt.zero_grad(set_to_none=True)
            p_logits = model_logits(policy, eval_ids)
            with torch.no_grad():
                r_logits = model_logits(ref, eval_ids)
            loss, stats = grpo_loss(p_logits, r_logits, eval_ids, comp_mask, adv, args.beta)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            step += 1
            e_steps += 1
            e_loss += stats["loss"]
            e_reward += float(rewards.mean().item())
            if step % int(args.log_every) == 0:
                print(f"[train] step={step} loss={stats['loss']:.6f} reward={float(rewards.mean().item()):.4f} adv_abs={stats['adv_abs']:.4f} logp={stats['logp']:.4f} kl={stats['kl']:.6f}")
            if step % int(args.save_every) == 0:
                pth = save_state(
                    out_dir,
                    f"grpo_step_{step:07d}.pt",
                    policy,
                    opt,
                    args,
                    step,
                    epoch,
                    model_cfg,
                    tok_obj,
                    save_lora_adapter=use_lora,
                )
                print(f"[save] {pth}")
        if e_steps:
            print(f"[epoch] {epoch+1}/{args.epochs} avg_loss={e_loss/e_steps:.6f} avg_reward={e_reward/e_steps:.4f}")
        if stop:
            break

    final = save_state(
        out_dir,
        "grpo_final.pt",
        policy,
        opt,
        args,
        step,
        max(0, int(args.epochs) - 1),
        model_cfg,
        tok_obj,
        save_lora_adapter=use_lora,
    )
    print(f"[done] final_checkpoint={final}")


if __name__ == "__main__":
    main()
