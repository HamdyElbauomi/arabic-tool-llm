import json
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "examples_v2.jsonl"
)

RANDOM_SEED = 42


ORDER_IDS = [
    1101 + i * 37
    for i in range(24)
]


STATUS_TEMPLATES = [
    ("ar", "فين طلبي رقم {order_id}؟"),
    ("ar", "الطلب {order_id} وصل لفين؟"),
    ("en", "What is the status of order {order_id}?"),
    ("en", "Track order {order_id} for me"),
    ("mixed", "ممكن check order {order_id}؟"),
    ("mixed", "order {order_id} وصل ولا لسه؟"),
    ("arabizi", "order {order_id} wesel fein?"),
    ("arabizi", "3ayz a3raf status order {order_id}"),
]


CANCEL_TEMPLATES = [
    ("ar", "عايز ألغي الطلب {order_id}"),
    ("ar", "مش عايز الأوردر {order_id}، الغيه"),
    ("en", "Cancel order {order_id}"),
    ("en", "I no longer want order {order_id}, cancel it"),
    ("mixed", "ممكن cancel order {order_id}؟"),
    ("mixed", "order {order_id} عايز ألغيه"),
    ("arabizi", "3ayz al8y order {order_id}"),
    ("arabizi", "cancel order {order_id} 3ashan msh 3ayzo"),
]


ADDRESS_TEMPLATES = [
    ("ar", "غير عنوان الطلب {order_id} وخليه {city}"),
    ("ar", "عدل عنوان أوردر {order_id} إلى {city}"),
    ("en", "Change the shipping address for order {order_id} to {city}"),
    ("en", "Update order {order_id} delivery address to {city}"),
    ("mixed", "عايز update address بتاع order {order_id} لـ {city}"),
    ("mixed", "order {order_id} غير الـ shipping address لـ {city}"),
    ("arabizi", "8ayar address order {order_id} le {city}"),
    ("arabizi", "update shipping address bta3 order {order_id} le {city}"),
]


RETURN_TEMPLATES = [
    ("ar", "عايز أرجع الطلب {order_id} عشان {reason_text}"),
    ("ar", "محتاج أعمل إرجاع للطلب {order_id} لأن {reason_text}"),
    ("en", "Return order {order_id} because {reason_text}"),
    ("en", "I need to return order {order_id} because {reason_text}"),
    ("mixed", "عايز return order {order_id} عشان {reason_text}"),
    ("mixed", "return للطلب {order_id} لأن {reason_text}"),
    ("arabizi", "3ayz araga3 order {order_id} 3ashan {reason_text}"),
    ("arabizi", "return order {order_id} 3ashan {reason_text}"),
]


REASON_TEXTS = {
    "damaged": {
        "ar": "المنتج تالف",
        "en": "the item is damaged",
        "mixed": "المنتج damaged",
        "arabizi": "el product talef",
    },
    "wrong_item": {
        "ar": "وصلني منتج غلط",
        "en": "I received the wrong item",
        "mixed": "وصلني wrong item",
        "arabizi": "wesely product ghalat",
    },
    "wrong_size": {
        "ar": "المقاس غلط",
        "en": "the size is wrong",
        "mixed": "الـ size غلط",
        "arabizi": "el size ghalat",
    },
    "not_working": {
        "ar": "المنتج مش شغال",
        "en": "the product is not working",
        "mixed": "المنتج not working",
        "arabizi": "el product msh sh8al",
    },
    "other": {
        "ar": "المنتج مش مناسب ليا",
        "en": "I changed my mind",
        "mixed": "المنتج مش مناسب",
        "arabizi": "msh monaseb leya",
    },
}


PRODUCT_TEMPLATES = [
    ("ar", "ممكن أعرف معلومات عن {product}?"),
    ("ar", "هل {product} موجود؟"),
    ("en", "Tell me about {product}"),
    ("en", "Do you have information about {product}?"),
    ("mixed", "عايز details عن {product}"),
    ("mixed", "{product} موجود عندكم؟"),
    ("arabizi", "3ayz info 3an {product}"),
    ("arabizi", "fe details 3an {product}?"),
]


PRODUCTS = [
    "iPhone 17",
    "Samsung Galaxy S26",
    "MacBook Air M5",
    "Sony WH-1000XM6",
    "PlayStation 6",
    "AirPods Pro 3",
    "Dell XPS 14",
    "Lenovo ThinkPad X1",
    "ASUS ROG Zephyrus",
    "Apple Watch Series 12",
    "Galaxy Watch 8",
    "iPad Pro M6",
    "Google Pixel 11",
    "Nintendo Switch 2",
    "Logitech MX Master 4",
    "Kindle Paperwhite",
    "GoPro Hero 14",
    "DJI Osmo Pocket 4",
    "Bose QuietComfort Ultra",
    "Steam Deck OLED",
    "Xiaomi 17 Pro",
    "OnePlus 14",
    "Huawei MateBook X Pro",
    "Meta Quest 4",
]


CITIES = [
    "القاهرة",
    "المنصورة",
    "الإسكندرية",
    "طنطا",
    "الجيزة",
    "الزقازيق",
    "Cairo",
    "Mansoura",
    "Alexandria",
    "Tanta",
    "Giza",
    "Zagazig",
] * 2


NO_TOOL_EXAMPLES = [
    ("ar", "مساء الخير"),
    ("ar", "صباح الخير"),
    ("ar", "شكراً جداً"),
    ("ar", "تمام شكراً"),
    ("ar", "سياسة الاسترجاع عندكم إيه؟"),
    ("ar", "بتوصلوا يوم الجمعة؟"),

    ("en", "Hello"),
    ("en", "Good morning"),
    ("en", "Thanks for your help"),
    ("en", "What is your return policy?"),
    ("en", "Do you deliver on weekends?"),
    ("en", "How can I contact support?"),

    ("mixed", "هاي، محتاج مساعدة"),
    ("mixed", "تمام thank you"),
    ("mixed", "hello مساء الخير"),
    ("mixed", "شكراً for your help"),
    ("mixed", "return policy عندكم إيه؟"),
    ("mixed", "support شغال امتى؟"),

    ("arabizi", "sabah el 5er"),
    ("arabizi", "masa2 el 5er"),
    ("arabizi", "shokran ya bro"),
    ("arabizi", "3andko return policy eh?"),
    ("arabizi", "btwslo yom el gom3a?"),
    ("arabizi", "ezay akalem support?"),
]


def build_dataset() -> list[dict]:

    examples = []
    counter = 1

    def add_example(
        language: str,
        text: str,
        expected: dict,
    ) -> None:

        nonlocal counter

        examples.append(
            {
                "id": f"v2_{counter:03d}",
                "language": language,
                "input": text,
                "expected": expected,
            }
        )

        counter += 1

    # --------------------------------------------------
    # Order status
    # --------------------------------------------------

    for index, order_id in enumerate(ORDER_IDS):

        language, template = (
            STATUS_TEMPLATES[
                index % len(STATUS_TEMPLATES)
            ]
        )

        add_example(
            language,
            template.format(
                order_id=order_id
            ),
            {
                "tool": "get_order_status",
                "arguments": {
                    "order_id": order_id
                },
            },
        )

    # --------------------------------------------------
    # Cancel order
    # --------------------------------------------------

    for index, order_id in enumerate(ORDER_IDS):

        order_id += 1

        language, template = (
            CANCEL_TEMPLATES[
                index % len(CANCEL_TEMPLATES)
            ]
        )

        add_example(
            language,
            template.format(
                order_id=order_id
            ),
            {
                "tool": "cancel_order",
                "arguments": {
                    "order_id": order_id
                },
            },
        )

    # --------------------------------------------------
    # Update address
    # --------------------------------------------------

    for index, order_id in enumerate(ORDER_IDS):

        order_id += 2
        city = CITIES[index]

        language, template = (
            ADDRESS_TEMPLATES[
                index % len(ADDRESS_TEMPLATES)
            ]
        )

        add_example(
            language,
            template.format(
                order_id=order_id,
                city=city,
            ),
            {
                "tool": "update_shipping_address",
                "arguments": {
                    "order_id": order_id,
                    "new_address": city,
                },
            },
        )

    # --------------------------------------------------
    # Return order
    # --------------------------------------------------

    reason_names = list(
        REASON_TEXTS.keys()
    )

    for index, order_id in enumerate(ORDER_IDS):

        order_id += 3

        language, template = (
            RETURN_TEMPLATES[
                index % len(RETURN_TEMPLATES)
            ]
        )

        reason = reason_names[
            index % len(reason_names)
        ]

        reason_text = (
            REASON_TEXTS[
                reason
            ][language]
        )

        add_example(
            language,
            template.format(
                order_id=order_id,
                reason_text=reason_text,
            ),
            {
                "tool": "return_order",
                "arguments": {
                    "order_id": order_id,
                    "reason": reason,
                },
            },
        )

    # --------------------------------------------------
    # Product info
    # --------------------------------------------------

    for index, product in enumerate(PRODUCTS):

        language, template = (
            PRODUCT_TEMPLATES[
                index % len(PRODUCT_TEMPLATES)
            ]
        )

        add_example(
            language,
            template.format(
                product=product
            ),
            {
                "tool": "get_product_info",
                "arguments": {
                    "product_name": product
                },
            },
        )

    # --------------------------------------------------
    # No tool
    # --------------------------------------------------

    for language, text in NO_TOOL_EXAMPLES:

        add_example(
            language,
            text,
            {
                "tool": "no_tool",
                "arguments": {},
            },
        )

    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    return examples


def save_dataset(
    examples: list[dict],
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        for example in examples:

            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_report(
    examples: list[dict],
) -> None:

    tool_counts = Counter(
        example["expected"]["tool"]
        for example in examples
    )

    language_counts = Counter(
        example["language"]
        for example in examples
    )

    print("=" * 55)
    print("DATASET V2 REPORT")
    print("=" * 55)

    print(
        f"Total examples: {len(examples)}"
    )

    print("\nExamples per tool:")

    for tool, count in sorted(
        tool_counts.items()
    ):
        print(
            f"{tool:<28} {count}"
        )

    print("\nExamples per language:")

    for language, count in sorted(
        language_counts.items()
    ):
        print(
            f"{language:<28} {count}"
        )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )


def main() -> None:

    examples = build_dataset()

    save_dataset(
        examples
    )

    print_report(
        examples
    )


if __name__ == "__main__":
    main()