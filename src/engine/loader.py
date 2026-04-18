"""
Model and tokenizer loading with automatic device placement,
dtype selection, and optional quantization config injection.

Compatibility note:
  Transformers 5.x changed how rope_scaling config is parsed for Phi-3.
  We patch the config rope_scaling dict before model init to ensure
  the 'type' key is always present, fixing the KeyError on older cached weights.
"""

import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoConfig
from src.utils import get_logger, detect_device, get_optimal_dtype

logger = get_logger(__name__)

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"


def _patch_phi3_rope_config(config) -> None:
    """
    Patch Phi-3 rope_scaling config for transformers 5.x compatibility.

    Older cached Phi-3 weights store rope_scaling as:
        {"short_factor": [...], "long_factor": [...]}   # missing 'type'

    Transformers 5.x expects:
        {"type": "longrope", "short_factor": [...], "long_factor": [...]}

    This patch adds the missing 'type' key so model init does not crash.
    """
    if not hasattr(config, "rope_scaling") or config.rope_scaling is None:
        return
    if isinstance(config.rope_scaling, dict) and "type" not in config.rope_scaling:
        config.rope_scaling["type"] = "longrope"
        logger.debug("Patched rope_scaling config: added missing 'type' = 'longrope'")


def load_quantization_config(precision: str) -> Optional[BitsAndBytesConfig]:
    """Build BitsAndBytesConfig for INT8 or INT4 (NF4) quantization."""
    if precision == "int8":
        logger.info("Loading with INT8 quantization (bitsandbytes).")
        return BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "int4":
        logger.info("Loading with INT4 quantization (bitsandbytes NF4).")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
    return None


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    precision: str = "fp16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Phi-3 Mini (or any HuggingFace CausalLM) with the specified precision.

    Args:
        model_name:         HuggingFace model ID or local path.
        precision:          fp32 | fp16 | bf16 | int8 | int4
        device_map:         "auto" lets accelerate handle multi-GPU placement.
        trust_remote_code:  Required for Phi-3.
        cache_dir:          Custom cache directory (useful in Kaggle /kaggle/working/).

    Returns:
        (model, tokenizer) tuple, model already on device.
    """
    device_info = detect_device()
    torch_dtype = get_optimal_dtype(device_info, precision)
    quant_config = load_quantization_config(precision)

    logger.info(f"Loading model: {model_name}")
    logger.info(f"Precision: {precision} | dtype: {torch_dtype} | backend: {device_info.backend}")

    # Step 1 — Load config and patch rope_scaling before model init
    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )
    _patch_phi3_rope_config(config)

    # Step 2 — Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
    )

    # Phi-3 specific: ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.debug("Set pad_token = eos_token for Phi-3.")

    # Step 3 — Load model with patched config passed directly
    model_kwargs = dict(
        pretrained_model_name_or_path=model_name,
        config=config,
        trust_remote_code=trust_remote_code,
        device_map=device_map,
        cache_dir=cache_dir,
    )

    if quant_config:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Model loaded. Parameters: {param_count:.2f}B")

    return model, tokenizer
