import json

from src.inference.inference_pipeline import (
    ToolCallingPipeline,
)
from src.tools.tool_router import (
    route_tool_call,
)


# --------------------------------------------------
# Customer support agent
# --------------------------------------------------

class CustomerSupportAgent:
    """
    Connect the fine-tuned LLM to the tool router.

    Flow:
        User message
            ↓
        LLM inference
            ↓
        Validated tool call
            ↓
        Tool router
            ↓
        Python function execution
    """

    def __init__(self):

        print("=" * 60)
        print("INITIALIZING CUSTOMER SUPPORT AGENT")
        print("=" * 60)

        # Model is loaded only once.
        self.pipeline = ToolCallingPipeline()

        print(
            "\n✅ Customer support agent ready."
        )

    # --------------------------------------------------
    # Run one request
    # --------------------------------------------------

    def run(
        self,
        user_message: str,
    ) -> dict:

        # Step 1:
        # Ask the fine-tuned model which tool to use.
        tool_call = self.pipeline.predict(
            user_message
        )

        # Step 2:
        # Execute the selected tool.
        execution = route_tool_call(
            tool_call
        )

        # Step 3:
        # Return complete structured result.
        return {
            "user_message": user_message,
            "tool_call": tool_call,
            "tool_result": execution[
                "result"
            ],
        }


# --------------------------------------------------
# End-to-end smoke test
# --------------------------------------------------

def main() -> None:

    agent = CustomerSupportAgent()

    test_messages = [
        "فين الطلب رقم 3147 وصل؟",
        "Please cancel order 4208",
        (
            "Return order 6621 because "
            "the item is damaged"
        ),
        (
            "ممكن معلومات عن "
            "MacBook Air M5؟"
        ),
    ]

    print("\n")
    print("=" * 60)
    print("END-TO-END AGENT TEST")
    print("=" * 60)

    for index, message in enumerate(
        test_messages,
        start=1,
    ):

        print(
            f"\n[{index}] User: {message}"
        )

        try:

            result = agent.run(
                message
            )

            print(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                )
            )

        except Exception as error:

            print(
                f"Error: {error}"
            )

    print("\n" + "=" * 60)
    print(
        "✅ End-to-end agent test completed."
    )


if __name__ == "__main__":
    main()