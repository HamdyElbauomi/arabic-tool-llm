import json
from pathlib import Path


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "examples_v2.jsonl"
TOOLS_CONFIG_PATH = PROJECT_ROOT / "configs" / "tools.json"


def load_tools() -> dict:
    """
    Load the available tools and their required arguments.
    """
    with open(TOOLS_CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    return config["tools"]


def validate_example(example: dict, tools: dict, line_number: int) -> list[str]:
    """
    Validate one dataset example.

    Returns a list of errors.
    An empty list means the example is valid.
    """
    errors = []

    # Check required top-level fields
    required_fields = ["id", "language", "input", "expected"]

    for field in required_fields:
        if field not in example:
            errors.append(
                f"Line {line_number}: missing field '{field}'"
            )

    # If expected is missing, we cannot continue validating it
    if "expected" not in example:
        return errors

    expected = example["expected"]

    # Validate expected structure
    if "tool" not in expected:
        errors.append(
            f"Line {line_number}: expected.tool is missing"
        )
        return errors

    if "arguments" not in expected:
        errors.append(
            f"Line {line_number}: expected.arguments is missing"
        )
        return errors

    tool_name = expected["tool"]
    arguments = expected["arguments"]

    # Check if tool exists
    if tool_name not in tools:
        errors.append(
            f"Line {line_number}: unknown tool '{tool_name}'"
        )
        return errors

    # Arguments must be a dictionary
    if not isinstance(arguments, dict):
        errors.append(
            f"Line {line_number}: arguments must be an object/dictionary"
        )
        return errors

    # Check required arguments for the selected tool
    required_arguments = tools[tool_name]["required_arguments"]

    for argument in required_arguments:
        if argument not in arguments:
            errors.append(
                f"Line {line_number}: tool '{tool_name}' "
                f"is missing required argument '{argument}'"
            )

    # Validate order_id
    if "order_id" in arguments:
        if not isinstance(arguments["order_id"], int):
            errors.append(
                f"Line {line_number}: order_id must be an integer"
            )

    # Check basic text fields
    if "input" in example:
        if not isinstance(example["input"], str) or not example["input"].strip():
            errors.append(
                f"Line {line_number}: input must be a non-empty string"
            )

    if "language" in example:
        allowed_languages = {"ar", "en", "mixed", "arabizi"}

        if example["language"] not in allowed_languages:
            errors.append(
                f"Line {line_number}: invalid language "
                f"'{example['language']}'"
            )

    return errors


def validate_dataset() -> None:
    """
    Validate all examples in the JSONL dataset.
    """

    tools = load_tools()

    total_examples = 0
    valid_examples = 0
    all_errors = []

    seen_ids = set()

    with open(DATASET_PATH, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            line = line.strip()

            # Ignore empty lines
            if not line:
                continue

            total_examples += 1

            # Validate JSON syntax
            try:
                example = json.loads(line)

            except json.JSONDecodeError as error:
                all_errors.append(
                    f"Line {line_number}: invalid JSON -> {error}"
                )
                continue

            # Check duplicate IDs
            example_id = example.get("id")

            if example_id:
                if example_id in seen_ids:
                    all_errors.append(
                        f"Line {line_number}: duplicate id '{example_id}'"
                    )
                    continue

                seen_ids.add(example_id)

            # Validate example content
            errors = validate_example(
                example=example,
                tools=tools,
                line_number=line_number,
            )

            if errors:
                all_errors.extend(errors)

            else:
                valid_examples += 1

    print("=" * 50)
    print("DATASET VALIDATION REPORT")
    print("=" * 50)

    print(f"Total examples : {total_examples}")
    print(f"Valid examples : {valid_examples}")
    print(f"Invalid        : {total_examples - valid_examples}")

    if all_errors:
        print("\nErrors:")

        for error in all_errors:
            print(f"- {error}")

        print("\n❌ Dataset validation failed.")

    else:
        print("\n✅ Dataset is valid.")


if __name__ == "__main__":
    validate_dataset()