import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "train.jsonl"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation.jsonl"
)

TOOLS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "tools.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "qwen3-0.6b-tool-calling-v2-lora"
)

MODEL_NAME = "Qwen/Qwen3-0.6B"


# --------------------------------------------------
# Load tools
# --------------------------------------------------

def load_tools() -> dict:
    """
    Load available tools from tools.json.
    """

    with open(
        TOOLS_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    return config["tools"]


# --------------------------------------------------
# Build system prompt
# --------------------------------------------------

def build_system_prompt(
    tools: dict,
) -> str:
    """
    Build the same tool-calling instructions
    used during inference.
    """

    tools_text = json.dumps(
        tools,
        ensure_ascii=False,
        indent=2,
    )

    return f"""
You are a bilingual Arabic-English tool-calling assistant.

Your job is to analyze the user's request and select exactly one
of the available tools.

Available tools:

{tools_text}

Rules:

1. Choose exactly one tool.
2. Extract all required arguments from the user message.
3. Never invent missing values.
4. Use "no_tool" when no tool is required.
5. Return only one valid JSON object.
6. Do not include explanations.
7. Do not include Markdown.

Required output format:

{{
    "tool": "tool_name",
    "arguments": {{}}
}}
""".strip()


# --------------------------------------------------
# Load JSONL
# --------------------------------------------------

def load_jsonl(
    path: Path,
) -> list[dict]:
    """
    Load a JSONL dataset.
    """

    examples = []

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            examples.append(
                json.loads(line)
            )

    return examples


# --------------------------------------------------
# Convert our dataset to chat format
# --------------------------------------------------

def convert_to_conversations(
    examples: list[dict],
    system_prompt: str,
) -> Dataset:
    """
    Convert our tool-calling dataset into the
    conversational format expected by SFTTrainer.
    """

    conversations = []

    for example in examples:

        expected_output = json.dumps(
            example["expected"],
            ensure_ascii=False,
        )

        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": example["input"],
                },
                {
                    "role": "assistant",
                    "content": expected_output,
                },
            ]
        }

        conversations.append(
            conversation
        )

    return Dataset.from_list(
        conversations
    )


# --------------------------------------------------
# QLoRA quantization
# --------------------------------------------------

def create_quantization_config():
    """
    Configure 4-bit NF4 quantization.

    The base model stays quantized while
    LoRA adapters are trained.
    """

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


# --------------------------------------------------
# LoRA configuration
# --------------------------------------------------

def create_lora_config():
    """
    Configure trainable LoRA adapters.
    """

    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.10,

        target_modules=[
            "q_proj",
            "v_proj",
        ],

        bias="none",
        task_type="CAUSAL_LM",
    )


# --------------------------------------------------
# Training configuration
# --------------------------------------------------

def create_training_config():
    """
    Configure supervised fine-tuning.
    """

    return SFTConfig(
        output_dir=str(OUTPUT_DIR),
        model_init_kwargs={
            "dtype": torch.float16,
        },

        # Small batch because LLM training
        # requires significant GPU memory.
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        # Simulates a larger effective batch.
        gradient_accumulation_steps=4,

        num_train_epochs=2,
        learning_rate=5e-5,

        warmup_ratio=0.10,
        weight_decay=0.01,

        # Our GPU does not support BF16 well,
        # so we use FP16.
        fp16=True,
        bf16=False,

        gradient_checkpointing=True,

        max_length=512,

        # Train only on assistant responses.
        assistant_only_loss=True,

        eval_strategy="epoch",
        save_strategy="epoch",

        logging_steps=1,

        save_total_limit=2,

        report_to="none",

        seed=42,
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("QWEN3 QLORA TRAINING PIPELINE")
    print("=" * 60)

    print(f"Base model: {MODEL_NAME}")

    # --------------------------------------------------
    # Load tool definitions
    # --------------------------------------------------

    tools = load_tools()

    system_prompt = build_system_prompt(
        tools
    )

    # --------------------------------------------------
    # Load datasets
    # --------------------------------------------------

    train_examples = load_jsonl(
        TRAIN_PATH
    )

    validation_examples = load_jsonl(
        VALIDATION_PATH
    )

    print(
        f"Train examples: {len(train_examples)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_examples)}"
    )

    # --------------------------------------------------
    # Convert to conversational dataset
    # --------------------------------------------------

    train_dataset = (
        convert_to_conversations(
            train_examples,
            system_prompt,
        )
    )

    validation_dataset = (
        convert_to_conversations(
            validation_examples,
            system_prompt,
        )
    )

    # --------------------------------------------------
    # QLoRA configuration
    # --------------------------------------------------

    quantization_config = (
        create_quantization_config()
    )

    lora_config = (
        create_lora_config()
    )

    training_config = (
        create_training_config()
    )

    # --------------------------------------------------
    # Create trainer
    # --------------------------------------------------

    trainer = SFTTrainer(
        model=MODEL_NAME,

        args=training_config,

        train_dataset=train_dataset,
        eval_dataset=validation_dataset,

        peft_config=lora_config,

        quantization_config=(
            quantization_config
        ),
    )
    # Keep trainable LoRA adapter weights in FP32.
    # The quantized base model remains frozen in 4-bit.
    for param in trainer.model.parameters():
        if param.requires_grad:
            param.data = param.data.float()

    trainable_dtypes = {
        param.dtype
        for param in trainer.model.parameters()
        if param.requires_grad
    }

    print(f"Trainable parameter dtypes: {trainable_dtypes}")

    # --------------------------------------------------
    # Show trainable parameters
    # --------------------------------------------------

    trainer.model.print_trainable_parameters()

    print("\n")
    print("=" * 60)
    print("STARTING QLORA FINE-TUNING")
    print("=" * 60)

    # Start actual fine-tuning
    train_result = trainer.train()

    print("\n")
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(f"Training loss: {train_result.training_loss:.4f}")

    # Save the trained LoRA adapter
    trainer.save_model(str(OUTPUT_DIR))

    # Save trainer state and training information
    trainer.save_state()

    print(f"\nLoRA adapter saved to:")
    print(OUTPUT_DIR)

    print("\n✅ First fine-tuning completed successfully.")


if __name__ == "__main__":
    main()