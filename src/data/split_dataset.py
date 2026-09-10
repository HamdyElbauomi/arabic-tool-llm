import json
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "examples_v2.jsonl"

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


def split_dataset(
    examples: list[dict],
):
    """
    Stratified 80/10/10 split by tool.
    """

    random.seed(RANDOM_SEED)

    grouped_examples = defaultdict(list)

    for example in examples:
        tool_name = example["expected"]["tool"]

        grouped_examples[
            tool_name
        ].append(example)

    train_examples = []
    validation_examples = []
    test_examples = []

    for tool_name, tool_examples in (
        grouped_examples.items()
    ):

        random.shuffle(
            tool_examples
        )

        total = len(
            tool_examples
        )

        validation_count = max(
            1,
            round(total * 0.10),
        )

        test_count = max(
            1,
            round(total * 0.10),
        )

        if (
            validation_count
            + test_count
            >= total
        ):
            raise ValueError(
                f"Not enough examples "
                f"for '{tool_name}'."
            )

        test_examples.extend(
            tool_examples[
                :test_count
            ]
        )

        validation_examples.extend(
            tool_examples[
                test_count:
                test_count
                + validation_count
            ]
        )

        train_examples.extend(
            tool_examples[
                test_count
                + validation_count:
            ]
        )

    random.shuffle(
        train_examples
    )

    random.shuffle(
        validation_examples
    )

    random.shuffle(
        test_examples
    )

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