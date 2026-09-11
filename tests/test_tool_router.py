import pytest

from src.tools.tool_router import (
    route_tool_call,
)


def test_get_order_status():

    result = route_tool_call(
        {
            "tool": "get_order_status",
            "arguments": {
                "order_id": 3147,
            },
        }
    )

    assert result["tool"] == (
        "get_order_status"
    )

    assert (
        result["result"]["success"]
        is True
    )

    assert (
        result["result"]["status"]
        == "out_for_delivery"
    )


def test_unknown_tool_is_rejected():

    with pytest.raises(
        ValueError,
    ):

        route_tool_call(
            {
                "tool": "delete_everything",
                "arguments": {},
            }
        )


def test_missing_arguments_field():

    with pytest.raises(
        ValueError,
    ):

        route_tool_call(
            {
                "tool": "get_order_status",
            }
        )