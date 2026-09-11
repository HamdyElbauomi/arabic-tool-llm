# --------------------------------------------------
# Mock backend data
# --------------------------------------------------

ORDERS = {
    3147: {
        "status": "out_for_delivery",
        "shipping_address": "Mansoura",
    },
    4208: {
        "status": "processing",
        "shipping_address": "Cairo",
    },
    5510: {
        "status": "processing",
        "shipping_address": "Alexandria",
    },
    6621: {
        "status": "delivered",
        "shipping_address": "Giza",
    },
}


PRODUCTS = {
    "MacBook Air M5": {
        "name": "MacBook Air M5",
        "available": True,
        "category": "Laptop",
    },
    "iPhone 17": {
        "name": "iPhone 17",
        "available": True,
        "category": "Smartphone",
    },
    "Samsung Galaxy S26": {
        "name": "Samsung Galaxy S26",
        "available": True,
        "category": "Smartphone",
    },
}


# --------------------------------------------------
# Tool: get_order_status
# --------------------------------------------------

def get_order_status(
    order_id: int,
) -> dict:

    order = ORDERS.get(order_id)

    if order is None:
        return {
            "success": False,
            "message": f"Order {order_id} not found.",
        }

    return {
        "success": True,
        "order_id": order_id,
        "status": order["status"],
    }


# --------------------------------------------------
# Tool: cancel_order
# --------------------------------------------------

def cancel_order(
    order_id: int,
) -> dict:

    order = ORDERS.get(order_id)

    if order is None:
        return {
            "success": False,
            "message": f"Order {order_id} not found.",
        }

    if order["status"] in {
        "delivered",
        "cancelled",
    }:
        return {
            "success": False,
            "message": (
                f"Order {order_id} cannot be cancelled "
                f"because its status is "
                f"'{order['status']}'."
            ),
        }

    order["status"] = "cancelled"

    return {
        "success": True,
        "order_id": order_id,
        "status": "cancelled",
        "message": "Order cancelled successfully.",
    }


# --------------------------------------------------
# Tool: update_shipping_address
# --------------------------------------------------

def update_shipping_address(
    order_id: int,
    new_address: str,
) -> dict:

    order = ORDERS.get(order_id)

    if order is None:
        return {
            "success": False,
            "message": f"Order {order_id} not found.",
        }

    if order["status"] in {
        "out_for_delivery",
        "delivered",
        "cancelled",
    }:
        return {
            "success": False,
            "message": (
                f"Shipping address cannot be changed "
                f"for order {order_id} because its "
                f"status is '{order['status']}'."
            ),
        }

    old_address = order[
        "shipping_address"
    ]

    order["shipping_address"] = (
        new_address
    )

    return {
        "success": True,
        "order_id": order_id,
        "old_address": old_address,
        "new_address": new_address,
        "message": (
            "Shipping address updated successfully."
        ),
    }


# --------------------------------------------------
# Tool: return_order
# --------------------------------------------------

def return_order(
    order_id: int,
    reason: str,
) -> dict:

    order = ORDERS.get(order_id)

    if order is None:
        return {
            "success": False,
            "message": f"Order {order_id} not found.",
        }

    allowed_reasons = {
        "damaged",
        "wrong_item",
        "wrong_size",
        "not_working",
        "other",
    }

    if reason not in allowed_reasons:
        return {
            "success": False,
            "message": (
                f"Invalid return reason: {reason}"
            ),
        }

    if order["status"] != "delivered":
        return {
            "success": False,
            "message": (
                f"Order {order_id} cannot be returned "
                f"because it has not been delivered."
            ),
        }

    return {
        "success": True,
        "order_id": order_id,
        "reason": reason,
        "message": (
            "Return request created successfully."
        ),
    }


# --------------------------------------------------
# Tool: get_product_info
# --------------------------------------------------

def get_product_info(
    product_name: str,
) -> dict:

    product = PRODUCTS.get(
        product_name
    )

    if product is None:
        return {
            "success": False,
            "message": (
                f"Product '{product_name}' "
                f"not found."
            ),
        }

    return {
        "success": True,
        "product": product,
    }


# --------------------------------------------------
# Tool: no_tool
# --------------------------------------------------

def no_tool() -> dict:

    return {
        "success": True,
        "message": (
            "No external tool is required."
        ),
    }