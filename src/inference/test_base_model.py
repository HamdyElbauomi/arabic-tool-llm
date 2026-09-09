import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOOLS_PATH = PROJECT_ROOT / "configs" / "tools.json"

MODEL_NAME = "Qwen/Qwen3-4B"


def load_tools() -> dict:
    with open(TOOLS_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    return config["tools"]


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


def load_model():
    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading model in 4-bit...")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    model.eval()

    return tokenizer, model


def predict(
    user_message: str,
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
            "content": user_message,
        },
    ]

    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )

    model_inputs = model_inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    input_length = model_inputs["input_ids"].shape[-1]

    new_tokens = generated_ids[
        0,
        input_length:,
    ]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return response.strip()


def main():
    print("=" * 60)
    print("QWEN3-4B BASE MODEL TEST")
    print("=" * 60)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tools = load_tools()
    system_prompt = build_system_prompt(tools)

    tokenizer, model = load_model()

    test_messages = [
        "فين طلبي رقم 4821؟",
        "Cancel order 9921",
        "غير عنوان الطلب 5512 للمنصورة",
        "I want to return order 5504 because the product is damaged",
        "شكراً جداً",
    ]

    for number, message in enumerate(test_messages, start=1):

        print("\n" + "=" * 60)
        print(f"Example {number}")
        print(f"User: {message}")

        prediction = predict(
            user_message=message,
            tokenizer=tokenizer,
            model=model,
            system_prompt=system_prompt,
        )

        print(f"Model: {prediction}")

    print("\n" + "=" * 60)
    print("Base model test completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()