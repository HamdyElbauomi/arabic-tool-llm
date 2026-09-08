import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "examples_raw.jsonl"

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.jsonl"
VALIDATION_PATH = PROJECT_ROOT / "data" / "processed" / "validation.jsonl"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.jsonl"

RANDOM_SEED = 42


def load_dataset(path: Path) -> list[dict]:
    """
    Load JSONL dataset into memory.
    """
    examples = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            examples.append(json.loads(line))

    return examples


def save_jsonl(examples: list[dict], path: Path) -> None:
    """
    Save examples into a JSONL file.
    """
    with open(path, "w", encoding="utf-8") as file:
        for example in examples:
            file.write(
                json.dumps(example, ensure_ascii=False) + "\n"
            )


def split_dataset(examples: list[dict]):
    """
    Split dataset by tool.

    Each tool currently has 6 examples:
    - 4 train
    - 1 validation
    - 1 test
    """

    random.seed(RANDOM_SEED)

    grouped_examples = defaultdict(list)

    # Group examples by tool
    for example in examples:
        tool_name = example["expected"]["tool"]

        grouped_examples[tool_name].append(example)

    train_examples = []
    validation_examples = []
    test_examples = []

    for tool_name, tool_examples in grouped_examples.items():

        random.shuffle(tool_examples)

        if len(tool_examples) < 3:
            raise ValueError(
                f"Tool '{tool_name}' needs at least 3 examples "
                f"for train/validation/test splitting."
            )

        # For our current dataset:
        # 6 examples -> 4 train, 1 validation, 1 test

        test_example = tool_examples[0]
        validation_example = tool_examples[1]
        train_tool_examples = tool_examples[2:]

        test_examples.append(test_example)
        validation_examples.append(validation_example)
        train_examples.extend(train_tool_examples)

    # Shuffle final datasets
    random.shuffle(train_examples)
    random.shuffle(validation_examples)
    random.shuffle(test_examples)

    return (
        train_examples,
        validation_examples,
        test_examples,
    )


def print_distribution(name: str, examples: list[dict]) -> None:
    """
    Print number of examples for every tool.
    """

    distribution = defaultdict(int)

    for example in examples:
        tool_name = example["expected"]["tool"]
        distribution[tool_name] += 1

    print(f"\n{name}")
    print("-" * 40)

    for tool_name, count in sorted(distribution.items()):
        print(f"{tool_name:<28} {count}")

    print(f"{'TOTAL':<28} {len(examples)}")


def main() -> None:

    examples = load_dataset(INPUT_PATH)

    train, validation, test = split_dataset(examples)

    save_jsonl(train, TRAIN_PATH)
    save_jsonl(validation, VALIDATION_PATH)
    save_jsonl(test, TEST_PATH)

    print("=" * 50)
    print("DATASET SPLIT REPORT")
    print("=" * 50)

    print_distribution("TRAIN", train)
    print_distribution("VALIDATION", validation)
    print_distribution("TEST", test)

    print("\n✅ Dataset split completed successfully.")


if __name__ == "__main__":
    main()