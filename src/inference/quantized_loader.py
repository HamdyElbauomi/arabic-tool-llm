import json
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

MODEL_REGISTRY_PATH = (
    PROJECT_ROOT
    / "configs"
    / "model_registry.json"
)


# --------------------------------------------------
# Load active model configuration
# --------------------------------------------------

def load_active_model_config() -> dict:
    """
    Read the active model configuration
    from model_registry.json.

    This allows us to switch model versions
    without changing the inference code.
    """

    if not MODEL_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Model registry not found: "
            f"{MODEL_REGISTRY_PATH}"
        )

    with open(
        MODEL_REGISTRY_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        registry = json.load(file)

    if "active_model" not in registry:
        raise ValueError(
            "model_registry.json is missing "
            "'active_model'."
        )

    active_model = registry[
        "active_model"
    ]

    required_fields = {
        "version",
        "base_model",
        "adapter",
    }

    missing_fields = (
        required_fields
        - active_model.keys()
    )

    if missing_fields:
        raise ValueError(
            "Active model configuration is "
            f"missing fields: {missing_fields}"
        )

    return active_model


# Load active model metadata
ACTIVE_MODEL = (
    load_active_model_config()
)

MODEL_VERSION = ACTIVE_MODEL[
    "version"
]

BASE_MODEL = ACTIVE_MODEL[
    "base_model"
]

ADAPTER_PATH = (
    PROJECT_ROOT
    / ACTIVE_MODEL["adapter"]
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
        isinstance(
            module,
            bnb.nn.Linear4bit,
        )
        for module in model.modules()
    )


# --------------------------------------------------
# Load quantized fine-tuned model
# --------------------------------------------------

def load_quantized_model():
    """
    Load the active model version:

        Base model in 4-bit NF4
        +
        Active LoRA adapter

    The selected version comes from:

        configs/model_registry.json
    """

    print("=" * 60)
    print("LOADING QUANTIZED FINE-TUNED MODEL")
    print("=" * 60)

    print(
        f"Model version : "
        f"{MODEL_VERSION}"
    )

    print(
        f"Base model    : "
        f"{BASE_MODEL}"
    )

    print(
        f"Adapter       : "
        f"{ADAPTER_PATH}"
    )

    # --------------------------------------------------
    # Validate adapter path
    # --------------------------------------------------

    if not ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found: "
            f"{ADAPTER_PATH}"
        )

    # --------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            BASE_MODEL
        )
    )

    # --------------------------------------------------
    # Create 4-bit configuration
    # --------------------------------------------------

    quantization_config = (
        create_quantization_config()
    )

    # --------------------------------------------------
    # Load base model in 4-bit
    # --------------------------------------------------

    print(
        "Loading base model "
        "in 4-bit NF4..."
    )

    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=(
                quantization_config
            ),
            device_map="auto",
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
    )

    # --------------------------------------------------
    # Verify quantization
    # --------------------------------------------------

    quantized_layers = (
        count_4bit_layers(
            base_model
        )
    )

    print(
        f"4-bit Linear layers: "
        f"{quantized_layers}"
    )

    # --------------------------------------------------
    # Attach active LoRA adapter
    # --------------------------------------------------

    print(
        f"Loading {MODEL_VERSION} "
        f"LoRA adapter..."
    )

    model = (
        PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            is_trainable=False,
        )
    )

    model.eval()

    # --------------------------------------------------
    # Verify inference mode
    # --------------------------------------------------

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable parameters "
        "during inference: "
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

    print(
        "\n" + "=" * 60
    )

    print(
        "QUANTIZATION CHECK"
    )

    print(
        "=" * 60
    )

    quantized_layers = (
        count_4bit_layers(
            model
        )
    )

    print(
        "Detected 4-bit layers : "
        f"{quantized_layers}"
    )

    print(
        "Model version         : "
        f"{MODEL_VERSION}"
    )

    print(
        "Adapter loaded        : "
        f"{ADAPTER_PATH.name}"
    )

    print(
        "Model training mode   : "
        f"{model.training}"
    )

    print(
        "\n✅ Quantized model "
        "loaded successfully."
    )


if __name__ == "__main__":
    main()