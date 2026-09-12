import json
import math
import statistics
import time
from pathlib import Path

import psutil
import torch

from src.inference.inference_pipeline import (
    ToolCallingPipeline,
)
from src.inference.quantized_loader import (
    BASE_MODEL,
    MODEL_VERSION,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_PATH = (
    REPORT_DIR
    / "performance_benchmark.json"
)

REPEATS = 3

TEST_MESSAGES = [
    "فين الطلب رقم 3147 وصل؟",
    "Please cancel order 4208",
    "غير عنوان الطلب 5510 وخليه القاهرة",
    (
        "Return order 6621 because "
        "the item is damaged"
    ),
    "ممكن معلومات عن MacBook Air M5؟",
    "مساء الخير يا صديقي",
]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def bytes_to_mb(
    value: int,
) -> float:

    return value / (1024 ** 2)


def percentile(
    values: list[float],
    percentage: float,
) -> float:

    if not values:
        return 0.0

    ordered = sorted(values)

    index = math.ceil(
        percentage
        * len(ordered)
    ) - 1

    index = max(
        0,
        min(
            index,
            len(ordered) - 1,
        ),
    )

    return ordered[index]


def get_process_ram_mb() -> float:

    process = psutil.Process()

    return bytes_to_mb(
        process.memory_info().rss
    )


def get_peak_process_ram_mb() -> float:

    process = psutil.Process()

    memory_info = (
        process.memory_info()
    )

    peak_bytes = getattr(
        memory_info,
        "peak_wset",
        memory_info.rss,
    )

    return bytes_to_mb(
        peak_bytes
    )


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

def main() -> None:

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    gpu_name = (
        torch.cuda.get_device_name(0)
    )

    gpu_properties = (
        torch.cuda.get_device_properties(0)
    )

    gpu_total_mb = bytes_to_mb(
        gpu_properties.total_memory
    )

    ram_before_load_mb = (
        get_process_ram_mb()
    )

    print("=" * 70)
    print("INFERENCE PERFORMANCE BENCHMARK")
    print("=" * 70)

    print(
        f"GPU           : {gpu_name}"
    )

    print(
        f"Model version : {MODEL_VERSION}"
    )

    print(
        f"Base model    : {BASE_MODEL}"
    )

    print(
        f"Repeats       : {REPEATS}"
    )

    # --------------------------------------------------
    # Model load time
    # --------------------------------------------------

    print(
        "\nLoading inference pipeline..."
    )

    load_start = time.perf_counter()

    pipeline = ToolCallingPipeline()

    torch.cuda.synchronize()

    model_load_seconds = (
        time.perf_counter()
        - load_start
    )

    ram_after_load_mb = (
        get_process_ram_mb()
    )

    gpu_allocated_after_load_mb = (
        bytes_to_mb(
            torch.cuda.memory_allocated()
        )
    )

    gpu_reserved_after_load_mb = (
        bytes_to_mb(
            torch.cuda.memory_reserved()
        )
    )

    print(
        f"\nModel load time: "
        f"{model_load_seconds:.3f} sec"
    )

    # --------------------------------------------------
    # Warmup
    # --------------------------------------------------

    print("\nRunning warmup...")

    for message in TEST_MESSAGES[:2]:

        try:
            pipeline.predict(message)

        except Exception:
            pass

    torch.cuda.synchronize()

    # Reset peak stats AFTER warmup.
    torch.cuda.reset_peak_memory_stats()

    # --------------------------------------------------
    # Timed benchmark
    # --------------------------------------------------

    latencies = []
    generation_latencies = []

    total_generated_tokens = 0

    successful_requests = 0
    failed_requests = 0

    max_ram_observed_mb = (
        get_process_ram_mb()
    )

    detailed_results = []

    print("\nRunning timed benchmark...")

    for repeat in range(
        1,
        REPEATS + 1,
    ):

        for index, message in enumerate(
            TEST_MESSAGES,
            start=1,
        ):

            # ------------------------------------------
            # One real generation
            # ------------------------------------------

            torch.cuda.synchronize()

            request_start = (
                time.perf_counter()
            )

            generation_start = (
                time.perf_counter()
            )

            try:

                raw_response = (
                    pipeline._generate(
                        message
                    )
                )

                torch.cuda.synchronize()

                generation_seconds = (
                    time.perf_counter()
                    - generation_start
                )

                parsed_response = (
                    pipeline._parse_json(
                        raw_response
                    )
                )

                validated_response = (
                    pipeline._validate_tool_call(
                        parsed_response
                    )
                )

                request_seconds = (
                    time.perf_counter()
                    - request_start
                )

                generated_tokens = len(
                    pipeline.tokenizer.encode(
                        raw_response,
                        add_special_tokens=False,
                    )
                )

                total_generated_tokens += (
                    generated_tokens
                )

                successful_requests += 1

                latencies.append(
                    request_seconds
                )

                generation_latencies.append(
                    generation_seconds
                )

                status = "success"

                result = (
                    validated_response
                )

            except Exception as error:

                torch.cuda.synchronize()

                request_seconds = (
                    time.perf_counter()
                    - request_start
                )

                failed_requests += 1

                status = "failed"

                generated_tokens = 0

                result = {
                    "error": str(error)
                }

            current_ram_mb = (
                get_process_ram_mb()
            )

            max_ram_observed_mb = max(
                max_ram_observed_mb,
                current_ram_mb,
            )

            detailed_results.append(
                {
                    "repeat": repeat,
                    "example": index,
                    "message": message,
                    "status": status,
                    "latency_seconds":
                        round(
                            request_seconds,
                            4,
                        ),
                    "generated_tokens":
                        generated_tokens,
                    "result": result,
                }
            )

            print(
                f"Repeat {repeat} | "
                f"Example {index} | "
                f"{status} | "
                f"{request_seconds:.3f}s | "
                f"{generated_tokens} tokens"
            )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    total_timed_seconds = sum(
        latencies
    )

    average_latency = (
        statistics.mean(latencies)
        if latencies
        else 0.0
    )

    median_latency = (
        statistics.median(latencies)
        if latencies
        else 0.0
    )

    p95_latency = percentile(
        latencies,
        0.95,
    )

    min_latency = (
        min(latencies)
        if latencies
        else 0.0
    )

    max_latency = (
        max(latencies)
        if latencies
        else 0.0
    )

    throughput = (
        successful_requests
        / total_timed_seconds
        if total_timed_seconds > 0
        else 0.0
    )

    total_generation_seconds = sum(
        generation_latencies
    )

    effective_tokens_per_second = (
        total_generated_tokens
        / total_generation_seconds
        if total_generation_seconds > 0
        else 0.0
    )

    gpu_peak_allocated_mb = (
        bytes_to_mb(
            torch.cuda.max_memory_allocated()
        )
    )

    gpu_peak_reserved_mb = (
        bytes_to_mb(
            torch.cuda.max_memory_reserved()
        )
    )

    peak_ram_mb = max(
        max_ram_observed_mb,
        get_peak_process_ram_mb(),
    )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    report = {
        "environment": {
            "gpu": gpu_name,
            "gpu_total_mb":
                round(
                    gpu_total_mb,
                    2,
                ),
            "model_version":
                MODEL_VERSION,
            "base_model":
                BASE_MODEL,
            "quantization":
                "4-bit NF4",
        },

        "model_loading": {
            "load_time_seconds":
                round(
                    model_load_seconds,
                    4,
                ),

            "process_ram_before_load_mb":
                round(
                    ram_before_load_mb,
                    2,
                ),

            "process_ram_after_load_mb":
                round(
                    ram_after_load_mb,
                    2,
                ),

            "gpu_allocated_after_load_mb":
                round(
                    gpu_allocated_after_load_mb,
                    2,
                ),

            "gpu_reserved_after_load_mb":
                round(
                    gpu_reserved_after_load_mb,
                    2,
                ),
        },

        "inference": {
            "successful_requests":
                successful_requests,

            "failed_requests":
                failed_requests,

            "average_latency_seconds":
                round(
                    average_latency,
                    4,
                ),

            "p50_latency_seconds":
                round(
                    median_latency,
                    4,
                ),

            "p95_latency_seconds":
                round(
                    p95_latency,
                    4,
                ),

            "min_latency_seconds":
                round(
                    min_latency,
                    4,
                ),

            "max_latency_seconds":
                round(
                    max_latency,
                    4,
                ),

            "throughput_requests_per_second":
                round(
                    throughput,
                    4,
                ),

            "effective_output_tokens_per_second":
                round(
                    effective_tokens_per_second,
                    4,
                ),
        },

        "memory": {
            "gpu_peak_allocated_mb":
                round(
                    gpu_peak_allocated_mb,
                    2,
                ),

            "gpu_peak_reserved_mb":
                round(
                    gpu_peak_reserved_mb,
                    2,
                ),

            "process_peak_ram_mb":
                round(
                    peak_ram_mb,
                    2,
                ),
        },

        "runs": detailed_results,
    }

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------
    # Print summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    print(
        f"Model load time       : "
        f"{model_load_seconds:.3f} sec"
    )

    print(
        f"Average latency       : "
        f"{average_latency:.3f} sec"
    )

    print(
        f"P50 latency           : "
        f"{median_latency:.3f} sec"
    )

    print(
        f"P95 latency           : "
        f"{p95_latency:.3f} sec"
    )

    print(
        f"Throughput            : "
        f"{throughput:.3f} req/sec"
    )

    print(
        f"Effective tokens/sec  : "
        f"{effective_tokens_per_second:.3f}"
    )

    print(
        f"GPU after load        : "
        f"{gpu_allocated_after_load_mb:.2f} MB"
    )

    print(
        f"GPU peak allocated    : "
        f"{gpu_peak_allocated_mb:.2f} MB"
    )

    print(
        f"GPU peak reserved     : "
        f"{gpu_peak_reserved_mb:.2f} MB"
    )

    print(
        f"Process RAM after load: "
        f"{ram_after_load_mb:.2f} MB"
    )

    print(
        f"Process peak RAM      : "
        f"{peak_ram_mb:.2f} MB"
    )

    print(
        f"Successful requests   : "
        f"{successful_requests}"
    )

    print(
        f"Failed requests       : "
        f"{failed_requests}"
    )

    print(
        f"\nReport saved to:\n"
        f"{REPORT_PATH}"
    )

    print(
        "\n✅ Performance benchmark completed."
    )


if __name__ == "__main__":
    main()