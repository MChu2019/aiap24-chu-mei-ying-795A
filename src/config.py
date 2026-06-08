import argparse
import json
from copy import deepcopy
from pathlib import Path


def deep_update(base, updates):
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AIAP delivery machine learning pipeline.")
    parser.add_argument("--config", default="config/default_config.json", help="Path to JSON config file.")
    parser.add_argument("--db-path", default=None, help="Override SQLite database path.")
    parser.add_argument("--task", choices=["classification", "regression"], default=None, help="ML task type.")
    parser.add_argument("--model", choices=["logistic_regression", "ridge_regression"], default=None, help="Model algorithm.")
    parser.add_argument("--test-size", type=float, default=None, help="Holdout test fraction.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Gradient descent learning rate.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs.")
    parser.add_argument("--l2", type=float, default=None, help="L2 regularisation strength.")
    parser.add_argument("--no-tuning", action="store_true", help="Disable hyperparameter search.")
    return parser.parse_args()


def build_config(args):
    config = load_config(args.config)
    overrides = {}

    if args.db_path:
        overrides.setdefault("data", {})["db_path"] = args.db_path
    if args.task:
        overrides.setdefault("task", {})["type"] = args.task
        overrides["task"]["target"] = "low_rating" if args.task == "classification" else "rating"
    if args.model:
        overrides.setdefault("model", {})["name"] = args.model
    if args.test_size is not None:
        overrides.setdefault("split", {})["test_size"] = args.test_size
    if args.no_tuning:
        overrides.setdefault("model", {}).setdefault("tuning", {})["enabled"] = False

    param_overrides = {}
    if args.learning_rate is not None:
        param_overrides["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        param_overrides["epochs"] = args.epochs
    if args.l2 is not None:
        param_overrides["l2"] = args.l2
    if param_overrides:
        overrides.setdefault("model", {}).setdefault("params", {}).update(param_overrides)

    final_config = deep_update(config, overrides)
    Path(final_config["outputs"]["dir"]).mkdir(parents=True, exist_ok=True)
    return final_config
