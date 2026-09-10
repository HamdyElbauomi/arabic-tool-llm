import json
from pathlib import Path

import torch

from src.inference.quantized_loader import (
    load_quantized_model,
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOOLS_PATH = (
    PROJECT_ROOT
    / "configs"
    / "tools.json"
)


# --------------------------------------------------
# Tool-calling inference pipeline
# --------------------------------------------------

class ToolCallingPipeline:
    """
    Production-style inference pipeline.

    Responsibilities:
    1. Load tools.
    2. Build the system prompt.
    3. Load the quantized fine-tuned model.
    4. Generate a model response.
    5. Parse JSON.
    6. Validate tool name and arguments.
    """

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING TOOL-CALLING PIPELINE")
        print("=" * 60)

        self.tools = self._load_tools()

        self.system_prompt = (
            self._build_system_prompt()
        )

        (
            self.tokenizer,
            self.model,
        ) = load_quantized_model()

        print(
            "\n✅ Inference pipeline initialized."
        )

    # --------------------------------------------------
    # Load tool definitions
    # --------------------------------------------------

    def _load_tools(self) -> dict:

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

    def _build_system_prompt(self) -> str:

        tools_text = json.dumps(
            self.tools,
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
    # Generate raw response
    # --------------------------------------------------

    def _generate(
        self,
        user_message: str,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ]

        model_inputs = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )
        )

        model_inputs = model_inputs.to(
            self.model.device
        )

        with torch.inference_mode():

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )

        input_length = (
            model_inputs[
                "input_ids"
            ].shape[-1]
        )

        new_tokens = generated_ids[
            0,
            input_length:,
        ]

        response = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True,
        )

        return response.strip()

    # --------------------------------------------------
    # Parse JSON
    # --------------------------------------------------

    def _parse_json(
        self,
        raw_response: str,
    ) -> dict:

        try:

            result = json.loads(
                raw_response
            )

        except json.JSONDecodeError as error:

            raise ValueError(
                "Model returned invalid JSON:\n"
                f"{raw_response}"
            ) from error

        if not isinstance(result, dict):

            raise ValueError(
                "Model output must be "
                "a JSON object."
            )

        return result

    # --------------------------------------------------
    # Validate argument type
    # --------------------------------------------------

    @staticmethod
    def _validate_type(
        argument_name: str,
        value,
        parameter_type: str,
    ) -> None:

        if (
            parameter_type == "integer"
            and not isinstance(value, int)
        ):

            raise ValueError(
                f"Argument '{argument_name}' "
                f"must be an integer."
            )

        if (
            parameter_type == "string"
            and not isinstance(value, str)
        ):

            raise ValueError(
                f"Argument '{argument_name}' "
                f"must be a string."
            )

    # --------------------------------------------------
    # Validate complete tool call
    # --------------------------------------------------

    def _validate_tool_call(
        self,
        result: dict,
    ) -> dict:

        if "tool" not in result:

            raise ValueError(
                "Missing 'tool' field."
            )

        if "arguments" not in result:

            raise ValueError(
                "Missing 'arguments' field."
            )

        tool_name = result["tool"]
        arguments = result["arguments"]

        if tool_name not in self.tools:

            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        if not isinstance(arguments, dict):

            raise ValueError(
                "'arguments' must be "
                "a JSON object."
            )

        tool_config = self.tools[
            tool_name
        ]

        required_arguments = (
            tool_config[
                "required_arguments"
            ]
        )

        parameters = tool_config[
            "parameters"
        ]

        # ----------------------------------------------
        # Required arguments
        # ----------------------------------------------

        for argument_name in (
            required_arguments
        ):

            if argument_name not in arguments:

                raise ValueError(
                    f"Missing required argument "
                    f"'{argument_name}' "
                    f"for tool '{tool_name}'."
                )

        # ----------------------------------------------
        # Reject unknown arguments
        # ----------------------------------------------

        for argument_name in arguments:

            if argument_name not in parameters:

                raise ValueError(
                    f"Unexpected argument "
                    f"'{argument_name}' "
                    f"for tool '{tool_name}'."
                )

        # ----------------------------------------------
        # Validate simple parameter types
        # ----------------------------------------------

        for argument_name, value in (
            arguments.items()
        ):

            parameter_type = parameters[
                argument_name
            ]

            self._validate_type(
                argument_name,
                value,
                parameter_type,
            )

        # ----------------------------------------------
        # Validate structured return reason
        # ----------------------------------------------

        if tool_name == "return_order":

            allowed_reasons = {
                "damaged",
                "wrong_item",
                "wrong_size",
                "not_working",
                "other",
            }

            reason = arguments[
                "reason"
            ]

            if reason not in allowed_reasons:

                raise ValueError(
                    f"Invalid return reason: "
                    f"{reason}"
                )

        return {
            "tool": tool_name,
            "arguments": arguments,
        }

    # --------------------------------------------------
    # Public inference method
    # --------------------------------------------------

    def predict(
        self,
        user_message: str,
    ) -> dict:

        if not isinstance(
            user_message,
            str,
        ):

            raise TypeError(
                "user_message must be a string."
            )

        user_message = (
            user_message.strip()
        )

        if not user_message:

            raise ValueError(
                "user_message cannot be empty."
            )

        raw_response = self._generate(
            user_message
        )

        parsed_response = self._parse_json(
            raw_response
        )

        validated_response = (
            self._validate_tool_call(
                parsed_response
            )
        )

        return validated_response


# --------------------------------------------------
# Manual smoke test
# --------------------------------------------------

def main() -> None:

    pipeline = ToolCallingPipeline()

    test_messages = [
        "فين الطلب رقم 3147 وصل؟",
        "Please cancel order 4208",
        "غير عنوان الطلب 5510 وخليه القاهرة",
        "Return order 6621 because the item is damaged",
        "ممكن معلومات عن MacBook Air M5؟",
        "مساء الخير يا صديقي",
    ]

    print("\n")
    print("=" * 60)
    print("INFERENCE PIPELINE TEST")
    print("=" * 60)

    for index, message in enumerate(
        test_messages,
        start=1,
    ):

        print(
            f"\n[{index}] User: "
            f"{message}"
        )

        try:

            result = pipeline.predict(
                message
            )

            print(
                "Result: "
                + json.dumps(
                    result,
                    ensure_ascii=False,
                )
            )

        except Exception as error:

            print(
                f"Error: {error}"
            )

    print("\n" + "=" * 60)
    print(
        "✅ Inference pipeline test completed."
    )


if __name__ == "__main__":
    main()