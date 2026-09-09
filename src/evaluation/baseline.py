import json
import sys
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allows importing from src/
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.test_base_model import (  # noqa: E402
    build_system_prompt,
    load_model,
    load_tools,
    predict,
)


TEST_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.jsonl"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "baseline_predictions.jsonl"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "baseline_summary.json"
)


# --------------------------------------------------
# Load test dataset
# --------------------------------------------------

def load_test_dataset() -> list[dict]:
    examples = []

    with open(
        TEST_DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    return examples


# --------------------------------------------------
# Parse model output
# --------------------------------------------------

def parse_prediction(
    response: str,
) -> dict | None:
    """
    Try to parse the model response as strict JSON.

    If parsing fails, return None.
    """

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

def evaluate_prediction(
    expected: dict,
    prediction: dict | None,
) -> dict:

    # Model did not return valid structured JSON
    if prediction is None:
        return {
            "json_valid": False,
            "tool_correct": False,
            "arguments_correct": False,
            "exact_match": False,
        }

    json_valid = True

    tool_correct = (
        prediction["tool"]
        == expected["tool"]
    )

    arguments_correct = (
        prediction["arguments"]
        == expected["arguments"]
    )

    exact_match = (
        tool_correct
        and arguments_correct
    )

    return {
        "json_valid": json_valid,
        "tool_correct": tool_correct,
        "arguments_correct": arguments_correct,
        "exact_match": exact_match,
    }


# --------------------------------------------------
# Calculate percentage
# --------------------------------------------------

def percentage(
    correct: int,
    total: int,
) -> float:

    if total == 0:
        return 0.0

    return round(
        correct / total * 100,
        2,
    )


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("QWEN3-4B BASELINE EVALUATION")
    print("=" * 60)

    examples = load_test_dataset()

    print(
        f"Test examples: {len(examples)}"
    )

    tools = load_tools()

    system_prompt = build_system_prompt(
        tools
    )

    tokenizer, model = load_model()

    results = []

    json_valid_count = 0
    tool_correct_count = 0
    arguments_correct_count = 0
    exact_match_count = 0

    # --------------------------------------------------
    # Run evaluation
    # --------------------------------------------------

    for index, example in enumerate(
        examples,
        start=1,
    ):

        print("\n" + "=" * 60)

        print(
            f"Example {index}/{len(examples)}"
        )

        print(
            f"ID: {example['id']}"
        )

        print(
            f"User: {example['input']}"
        )

        expected = example["expected"]

        raw_response = predict(
            user_message=example["input"],
            tokenizer=tokenizer,
            model=model,
            system_prompt=system_prompt,
        )

        prediction = parse_prediction(
            raw_response
        )

        metrics = evaluate_prediction(
            expected=expected,
            prediction=prediction,
        )

        json_valid_count += int(
            metrics["json_valid"]
        )

        tool_correct_count += int(
            metrics["tool_correct"]
        )

        arguments_correct_count += int(
            metrics["arguments_correct"]
        )

        exact_match_count += int(
            metrics["exact_match"]
        )

        print(
            f"Expected: {expected}"
        )

        print(
            f"Predicted: {prediction}"
        )

        print(
            f"Tool correct: "
            f"{metrics['tool_correct']}"
        )

        print(
            f"Arguments correct: "
            f"{metrics['arguments_correct']}"
        )

        result = {
            "id": example["id"],
            "language": example["language"],
            "input": example["input"],
            "expected": expected,
            "raw_response": raw_response,
            "prediction": prediction,
            "metrics": metrics,
        }

        results.append(result)

    # --------------------------------------------------
    # Calculate final metrics
    # --------------------------------------------------

    total = len(examples)

    summary = {
        "model": "Qwen/Qwen3-4B",
        "fine_tuned": False,
        "quantization": "4-bit NF4",
        "total_examples": total,
        "json_validity": percentage(
            json_valid_count,
            total,
        ),
        "tool_accuracy": percentage(
            tool_correct_count,
            total,
        ),
        "argument_accuracy": percentage(
            arguments_correct_count,
            total,
        ),
        "exact_match": percentage(
            exact_match_count,
            total,
        ),
    }

    # --------------------------------------------------
    # Save predictions
    # --------------------------------------------------

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        for result in results:

            file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    print("\n")
    print("=" * 60)
    print("BASELINE RESULTS")
    print("=" * 60)

    print(
        f"Examples          : {total}"
    )

    print(
        f"JSON Validity     : "
        f"{summary['json_validity']}%"
    )

    print(
        f"Tool Accuracy     : "
        f"{summary['tool_accuracy']}%"
    )

    print(
        f"Argument Accuracy : "
        f"{summary['argument_accuracy']}%"
    )

    print(
        f"Exact Match       : "
        f"{summary['exact_match']}%"
    )

    print("=" * 60)

    print(
        "\n✅ Baseline evaluation completed."
    )

    print(
        f"Predictions saved to:\n"
        f"{RESULTS_PATH}"
    )

    print(
        f"\nSummary saved to:\n"
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()