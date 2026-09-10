from pathlib import Path

import bitsandbytes as bnb
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "Qwen/Qwen3-0.6B"

ADAPTER_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-0.6b-tool-calling-v2-lora"
)


# --------------------------------------------------
# Quantization configuration
# --------------------------------------------------

def create_quantization_config():
    """
    Load the base model using 4-bit NF4 quantization.

    The LoRA adapter stays separate and is attached
    after loading the quantized base model.
    """

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


# --------------------------------------------------
# Count quantized layers
# --------------------------------------------------

def count_4bit_layers(model) -> int:
    """
    Count Linear4bit layers to verify that
    quantization was actually applied.
    """

    return sum(
        isinstance(module, bnb.nn.Linear4bit)
        for module in model.modules()
    )


# --------------------------------------------------
# Load quantized fine-tuned model
# --------------------------------------------------

def load_quantized_model():
    """
    Load:
        Qwen3-0.6B base model in 4-bit
        +
        our trained V2 LoRA adapter
    """

    print("=" * 60)
    print("LOADING QUANTIZED FINE-TUNED MODEL")
    print("=" * 60)

    print(f"Base model : {BASE_MODEL}")
    print(f"Adapter    : {ADAPTER_PATH}")

    # Load tokenizer
    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    # Create 4-bit configuration
    quantization_config = (
        create_quantization_config()
    )

    # Load base model directly in 4-bit
    print("Loading base model in 4-bit NF4...")

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quantization_config,
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    quantized_layers = count_4bit_layers(
        base_model
    )

    print(
        f"4-bit Linear layers: "
        f"{quantized_layers}"
    )

    # Attach trained LoRA adapter
    print("Loading V2 LoRA adapter...")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
        is_trainable=False,
    )

    model.eval()

    # Check that no parameters are trainable
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters during inference: "
        f"{trainable_parameters}"
    )

    return tokenizer, model


# --------------------------------------------------
# Main verification
# --------------------------------------------------

def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    tokenizer, model = (
        load_quantized_model()
    )

    print("\n" + "=" * 60)
    print("QUANTIZATION CHECK")
    print("=" * 60)

    quantized_layers = count_4bit_layers(
        model
    )

    print(
        f"Detected 4-bit layers : "
        f"{quantized_layers}"
    )

    print(
        f"Adapter loaded        : "
        f"{ADAPTER_PATH.name}"
    )

    print(
        f"Model training mode   : "
        f"{model.training}"
    )

    print("\n✅ Quantized model loaded successfully.")


if __name__ == "__main__":
    main()