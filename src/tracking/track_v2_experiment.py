from pathlib import Path

import mlflow


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

MLFLOW_DB = (
    PROJECT_ROOT / "mlflow.db"
)

TOOLS_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "tools.json"
)

MODEL_REGISTRY = (
    PROJECT_ROOT
    / "configs"
    / "model_registry.json"
)


# --------------------------------------------------
# MLflow setup
# --------------------------------------------------

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB.as_posix()}"
)

mlflow.set_experiment(
    "arabic-tool-calling-llm"
)


# --------------------------------------------------
# Log V2 experiment
# --------------------------------------------------

def main() -> None:

    with mlflow.start_run(
        run_name="qwen3-0.6b-qlora-v2"
    ):

        # ------------------------------------------
        # Model parameters
        # ------------------------------------------

        mlflow.log_params(
            {
                "base_model":
                    "Qwen/Qwen3-0.6B",

                "training_method":
                    "QLoRA",

                "quantization":
                    "4-bit NF4",

                "lora_r":
                    8,

                "lora_alpha":
                    16,

                "lora_dropout":
                    0.10,

                "target_modules":
                    "q_proj,v_proj",

                "epochs":
                    2,

                "learning_rate":
                    5e-5,

                "batch_size":
                    1,

                "gradient_accumulation_steps":
                    4,

                "max_length":
                    512,

                "train_examples":
                    120,

                "validation_examples":
                    12,

                "test_examples":
                    12,

                "dataset_version":
                    "v2",
            }
        )

        # ------------------------------------------
        # Training metrics
        # ------------------------------------------

        mlflow.log_metrics(
            {
                "train_loss":
                    0.2206,

                "eval_loss_epoch_1":
                    0.1569,

                "eval_loss_epoch_2":
                    0.1280,

                "training_runtime_seconds":
                    525.1,
            }
        )

        # ------------------------------------------
        # Evaluation metrics
        # ------------------------------------------

        mlflow.log_metrics(
            {
                "json_validity":
                    1.0,

                "tool_accuracy":
                    0.9167,

                "argument_accuracy":
                    0.8333,

                "exact_match":
                    0.8333,
            }
        )

        # ------------------------------------------
        # Useful configuration artifacts
        # ------------------------------------------

        if TOOLS_CONFIG.exists():

            mlflow.log_artifact(
                str(TOOLS_CONFIG),
                artifact_path="configs",
            )

        if MODEL_REGISTRY.exists():

            mlflow.log_artifact(
                str(MODEL_REGISTRY),
                artifact_path="registry",
            )

        # ------------------------------------------
        # Tags
        # ------------------------------------------

        mlflow.set_tags(
            {
                "project":
                    "arabic-tool-llm",

                "language":
                    "Arabic-English-Mixed-Arabizi",

                "task":
                    "tool-calling",

                "model_version":
                    "v2",

                "status":
                    "production-candidate",
            }
        )

        run_id = (
            mlflow.active_run()
            .info.run_id
        )

        print("=" * 60)
        print("EXPERIMENT TRACKED SUCCESSFULLY")
        print("=" * 60)

        print(
            f"Run ID: {run_id}"
        )

        print(
            "Experiment: "
            "arabic-tool-calling-llm"
        )

        print(
            "Model version: v2"
        )

        print(
            "Exact match: 83.33%"
        )

        print(
            "\n✅ MLflow tracking completed."
        )


if __name__ == "__main__":
    main()