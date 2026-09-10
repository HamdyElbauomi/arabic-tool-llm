import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.jsonl"
)

TOOLS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "tools.json"
)

ADAPTER_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-0.6b-tool-calling-smoke-lora"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "post_training_comparison.json"
)

BASE_MODEL = "Qwen/Qwen3-0.6B"


# --------------------------------------------------
# Load tools
# --------------------------------------------------

def load_tools() -> dict:
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

def build_system_prompt(tools: dict) -> str:

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
# Load dataset
# --------------------------------------------------

def load_dataset() -> list[dict]:

    examples = []

    with open(
        TEST_PATH,
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
# Quantization config
# --------------------------------------------------

def create_quantization_config():

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )


# --------------------------------------------------
# Load base model
# --------------------------------------------------

def load_base_model():

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=create_quantization_config(),
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    return tokenizer, model


# --------------------------------------------------
# Load fine-tuned model
# --------------------------------------------------

def load_fine_tuned_model():

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=create_quantization_config(),
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_PATH,
    )

    model.eval()

    return tokenizer, model


# --------------------------------------------------
# Generate prediction
# --------------------------------------------------

def predict(
    message: str,
    tokenizer,
    model,
    system_prompt: str,
) -> str:

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": message,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = (
        inputs["input_ids"].shape[-1]
    )

    generated_tokens = output[
        0,
        input_length:,
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


# --------------------------------------------------
# Parse JSON
# --------------------------------------------------

def parse_prediction(
    response: str,
) -> dict | None:

    try:
        prediction = json.loads(response)

    except json.JSONDecodeError:
        return None

    if not isinstance(prediction, dict):
        return None

    if "tool" not in prediction:
        return None

    if "arguments" not in prediction:
        return None

    if not isinstance(
        prediction["arguments"],
        dict,
    ):
        return None

    return prediction


# --------------------------------------------------
# Evaluate one prediction
# --------------------------------------------------

def evaluate(
    expected: dict,
    prediction: dict | None,
) -> dict:

    if prediction is None:

        return {
            "json_valid": False,
            "tool_correct": False,
            "arguments_correct": False,
            "exact_match": False,
        }

    tool_correct = (
        prediction["tool"]
        == expected["tool"]
    )

    arguments_correct = (
        prediction["arguments"]
        == expected["arguments"]
    )

    return {
        "json_valid": True,
        "tool_correct": tool_correct,
        "arguments_correct": arguments_correct,
        "exact_match": (
            tool_correct
            and arguments_correct
        ),
    }


# --------------------------------------------------
# Calculate percentage
# --------------------------------------------------

def percentage(
    value: int,
    total: int,
) -> float:

    if total == 0:
        return 0.0

    return round(
        value / total * 100,
        2,
    )


# --------------------------------------------------
# Evaluate full model
# --------------------------------------------------

def evaluate_model(
    model_name: str,
    tokenizer,
    model,
    dataset: list[dict],
    system_prompt: str,
) -> dict:

    print("\n")
    print("=" * 60)
    print(f"EVALUATING: {model_name}")
    print("=" * 60)

    json_count = 0
    tool_count = 0
    argument_count = 0
    exact_count = 0

    predictions = []

    for index, example in enumerate(
        dataset,
        start=1,
    ):

        response = predict(
            example["input"],
            tokenizer,
            model,
            system_prompt,
        )

        prediction = parse_prediction(
            response
        )

        metrics = evaluate(
            expected=example["expected"],
            prediction=prediction,
        )

        json_count += int(
            metrics["json_valid"]
        )

        tool_count += int(
            metrics["tool_correct"]
        )

        argument_count += int(
            metrics["arguments_correct"]
        )

        exact_count += int(
            metrics["exact_match"]
        )

        print(
            f"\n[{index}/{len(dataset)}] "
            f"{example['id']}"
        )

        print(
            f"Input     : {example['input']}"
        )

        print(
            f"Expected  : {example['expected']}"
        )

        print(
            f"Predicted : {prediction}"
        )

        print(
            f"Exact     : "
            f"{metrics['exact_match']}"
        )

        predictions.append(
            {
                "id": example["id"],
                "input": example["input"],
                "expected": example["expected"],
                "prediction": prediction,
                "raw_response": response,
                "metrics": metrics,
            }
        )

    total = len(dataset)

    summary = {
        "json_validity": percentage(
            json_count,
            total,
        ),
        "tool_accuracy": percentage(
            tool_count,
            total,
        ),
        "argument_accuracy": percentage(
            argument_count,
            total,
        ),
        "exact_match": percentage(
            exact_count,
            total,
        ),
    }

    return {
        "summary": summary,
        "predictions": predictions,
    }


# --------------------------------------------------
# Free GPU memory
# --------------------------------------------------

def cleanup_model(
    tokenizer,
    model,
) -> None:

    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------
# Print comparison
# --------------------------------------------------

def print_comparison(
    base_results: dict,
    fine_tuned_results: dict,
) -> None:

    base = base_results["summary"]
    tuned = fine_tuned_results["summary"]

    print("\n")
    print("=" * 70)
    print("BEFORE VS AFTER FINE-TUNING")
    print("=" * 70)

    print(
        f"{'Metric':<22}"
        f"{'Base':>12}"
        f"{'Fine-Tuned':>16}"
        f"{'Change':>14}"
    )

    print("-" * 70)

    metrics = [
        ("JSON Validity", "json_validity"),
        ("Tool Accuracy", "tool_accuracy"),
        ("Argument Accuracy", "argument_accuracy"),
        ("Exact Match", "exact_match"),
    ]

    for label, key in metrics:

        before = base[key]
        after = tuned[key]

        change = round(
            after - before,
            2,
        )

        print(
            f"{label:<22}"
            f"{before:>11.2f}%"
            f"{after:>15.2f}%"
            f"{change:>+13.2f}%"
        )

    print("=" * 70)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("POST-TRAINING EVALUATION")
    print("=" * 60)

    dataset = load_dataset()

    print(
        f"Test examples: {len(dataset)}"
    )

    tools = load_tools()

    system_prompt = build_system_prompt(
        tools
    )

    # --------------------------------------------------
    # Evaluate base model
    # --------------------------------------------------

    print(
        "\nLoading Qwen3-0.6B base model..."
    )

    tokenizer, base_model = (
        load_base_model()
    )

    base_results = evaluate_model(
        model_name="Qwen3-0.6B Base",
        tokenizer=tokenizer,
        model=base_model,
        dataset=dataset,
        system_prompt=system_prompt,
    )

    cleanup_model(
        tokenizer,
        base_model,
    )

    # --------------------------------------------------
    # Evaluate fine-tuned model
    # --------------------------------------------------

    print(
        "\nLoading Qwen3-0.6B + LoRA..."
    )

    tokenizer, fine_tuned_model = (
        load_fine_tuned_model()
    )

    fine_tuned_results = evaluate_model(
        model_name="Qwen3-0.6B + LoRA",
        tokenizer=tokenizer,
        model=fine_tuned_model,
        dataset=dataset,
        system_prompt=system_prompt,
    )

    # --------------------------------------------------
    # Compare
    # --------------------------------------------------

    print_comparison(
        base_results,
        fine_tuned_results,
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output = {
        "base_model": BASE_MODEL,
        "adapter": str(ADAPTER_PATH),
        "test_examples": len(dataset),
        "base": base_results,
        "fine_tuned": fine_tuned_results,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nResults saved to:\n"
        f"{RESULTS_PATH}"
    )

    print(
        "\n✅ Post-training evaluation completed."
    )


if __name__ == "__main__":
    main()