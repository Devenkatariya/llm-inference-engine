"""
Core inference loop with configurable sampling strategies.
Supports greedy, top-k, top-p (nucleus), and temperature sampling.
"""

import time
import torch
from typing import Optional, List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils import get_logger, DEVICE

logger = get_logger(__name__)


@torch.inference_mode()
def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    return_full_text: bool = False,
) -> Dict[str, Any]:
    """
    Run single-prompt inference and return text + performance metrics.

    Returns dict with: text, tokens_generated, latency_ms, tokens_per_second,
                       prompt_tokens, total_tokens.
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(DEVICE)

    prompt_len = inputs["input_ids"].shape[1]

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature if do_sample else 1.0,
        top_p=top_p if do_sample else 1.0,
        top_k=top_k if do_sample else 0,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_end = time.perf_counter()

    latency_ms = (t_end - t_start) * 1000
    tokens_generated = output_ids.shape[1] - prompt_len

    # Decode — optionally strip the prompt from output
    if return_full_text:
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        new_tokens = output_ids[0][prompt_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)

    tps = tokens_generated / (latency_ms / 1000) if latency_ms > 0 else 0.0

    return {
        "text": decoded,
        "prompt_tokens": prompt_len,
        "tokens_generated": tokens_generated,
        "total_tokens": output_ids.shape[1],
        "latency_ms": round(latency_ms, 2),
        "tokens_per_second": round(tps, 2),
    }


@torch.inference_mode()
def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    Batched inference — pads to longest sequence in batch.
    Returns list of result dicts, one per prompt.
    """
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,
    ).to(DEVICE)

    prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_start = time.perf_counter()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t_end = time.perf_counter()

    total_latency_ms = (t_end - t_start) * 1000
    results = []

    for i, (out, plen) in enumerate(zip(output_ids, prompt_lens)):
        new_tok = out[plen:]
        text = tokenizer.decode(new_tok, skip_special_tokens=True)
        n_gen = len(new_tok)
        results.append({
            "text": text,
            "prompt_tokens": int(plen),
            "tokens_generated": n_gen,
            "latency_ms": round(total_latency_ms, 2),
            "tokens_per_second": round(n_gen / (total_latency_ms / 1000), 2),
        })

    return results
