from src.tools.customer_support_tools import (
    cancel_order,
    get_order_status,
    get_product_info,
    no_tool,
    return_order,
    update_shipping_address,
)


# --------------------------------------------------
# Explicit allowlist of executable tools
# --------------------------------------------------

TOOL_REGISTRY = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
    "update_shipping_address": (
        update_shipping_address
    ),
    "return_order": return_order,
    "get_product_info": get_product_info,
    "no_tool": no_tool,
}


# --------------------------------------------------
# Tool router
# --------------------------------------------------

def route_tool_call(
    tool_call: dict,
) -> dict:
    """
    Route a validated tool call to the correct
    Python function.
    """

    if not isinstance(
        tool_call,
        dict,
    ):
        raise TypeError(
            "tool_call must be a dictionary."
        )

    if "tool" not in tool_call:
        raise ValueError(
            "tool_call is missing 'tool'."
        )

    if "arguments" not in tool_call:
        raise ValueError(
            "tool_call is missing 'arguments'."
        )

    tool_name = tool_call["tool"]

    arguments = tool_call[
        "arguments"
    ]

    if not isinstance(
        arguments,
        dict,
    ):
        raise TypeError(
            "'arguments' must be a dictionary."
        )

    tool_function = TOOL_REGISTRY.get(
        tool_name
    )

    if tool_function is None:
        raise ValueError(
            f"Tool '{tool_name}' "
            f"is not registered."
        )

    # Execute the selected function
    tool_result = tool_function(
        **arguments
    )

    return {
        "tool": tool_name,
        "arguments": arguments,
        "result": tool_result,
    }


# --------------------------------------------------
# Router smoke test
# --------------------------------------------------

def main() -> None:

    test_calls = [
        {
            "tool": "get_order_status",
            "arguments": {
                "order_id": 3147,
            },
        },
        {
            "tool": "cancel_order",
            "arguments": {
                "order_id": 4208,
            },
        },
        {
            "tool": "update_shipping_address",
            "arguments": {
                "order_id": 5510,
                "new_address": "القاهرة",
            },
        },
        {
            "tool": "return_order",
            "arguments": {
                "order_id": 6621,
                "reason": "damaged",
            },
        },
        {
            "tool": "get_product_info",
            "arguments": {
                "product_name": (
                    "MacBook Air M5"
                ),
            },
        },
        {
            "tool": "no_tool",
            "arguments": {},
        },
    ]

    print("=" * 60)
    print("TOOL ROUTER TEST")
    print("=" * 60)

    for index, tool_call in enumerate(
        test_calls,
        start=1,
    ):

        print(
            f"\n[{index}] "
            f"{tool_call['tool']}"
        )

        result = route_tool_call(
            tool_call
        )

        print(
            f"Arguments: "
            f"{result['arguments']}"
        )

        print(
            f"Result   : "
            f"{result['result']}"
        )

    print("\n" + "=" * 60)
    print(
        "✅ Tool router test completed."
    )


if __name__ == "__main__":
    main()