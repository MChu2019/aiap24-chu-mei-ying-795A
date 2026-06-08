from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import SQLiteDeliveryLoader
from src.evaluation import classification_metrics, regression_metrics, save_json, save_predictions
from src.features import FeaturePreprocessor, build_training_frame
from src.models import build_model, param_grid


def train_test_split_indices(n_rows, test_size, random_state):
    rng = np.random.default_rng(random_state)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    test_n = max(1, int(n_rows * test_size))
    return indices[test_n:], indices[:test_n]


class DeliveryMLPipeline:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path(config["outputs"]["dir"])

    def load_data(self):
        data_cfg = self.config["data"]
        loader = SQLiteDeliveryLoader(
            data_cfg["db_path"],
            data_cfg.get("delivery_table", "deliveries"),
            data_cfg.get("feedback_table", "feedback"),
        )
        return loader.load()

    def prepare_data(self, deliveries, feedback):
        frame = build_training_frame(deliveries, feedback, self.config)
        split_cfg = self.config["split"]
        train_idx, test_idx = train_test_split_indices(
            len(frame),
            split_cfg.get("test_size", 0.2),
            split_cfg.get("random_state", 42),
        )

        train_df = frame.iloc[train_idx].reset_index(drop=True)
        test_df = frame.iloc[test_idx].reset_index(drop=True)

        feature_cfg = self.config["features"]
        preprocessor = FeaturePreprocessor(feature_cfg["numeric"], feature_cfg["categorical"])
        x_train = preprocessor.fit_transform(train_df)
        x_test = preprocessor.transform(test_df)
        y_train = train_df["target"].to_numpy()
        y_test = test_df["target"].to_numpy()
        return train_df, test_df, x_train, x_test, y_train, y_test, preprocessor

    def tune_and_train(self, x_train, y_train, x_test, y_test):
        model_cfg = self.config["model"]
        task = self.config["task"]["type"]
        candidates = param_grid(model_cfg.get("tuning", {}), model_cfg.get("params", {}))
        results = []

        for params in candidates:
            model = build_model(model_cfg["name"], params)
            model.fit(x_train, y_train)
            if task == "classification":
                y_score = model.predict_proba(x_test)[:, 1]
                y_pred = (y_score >= 0.5).astype(int)
                metrics = classification_metrics(y_test, y_pred, y_score)
                score = metrics["f1"]
            else:
                y_pred = model.predict(x_test)
                metrics = regression_metrics(y_test, y_pred)
                score = -metrics["rmse"]

            results.append({"params": params, "metrics": metrics, "score": score, "model": model})

        best = max(results, key=lambda item: item["score"])
        serialisable = [{k: v for k, v in item.items() if k != "model"} for item in results]
        return best["model"], best["params"], best["metrics"], serialisable

    def run(self):
        deliveries, feedback = self.load_data()
        train_df, test_df, x_train, x_test, y_train, y_test, preprocessor = self.prepare_data(deliveries, feedback)
        model, best_params, metrics, tuning_results = self.tune_and_train(x_train, y_train, x_test, y_test)

        task = self.config["task"]["type"]
        if task == "classification":
            y_score = model.predict_proba(x_test)[:, 1]
            y_pred = (y_score >= 0.5).astype(int)
        else:
            y_score = None
            y_pred = model.predict(x_test)

        run_summary = {
            "task": task,
            "model": self.config["model"]["name"],
            "best_params": best_params,
            "metrics": metrics,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(x_train.shape[1]),
            "features": preprocessor.feature_names,
        }

        save_json(run_summary, self.output_dir / "metrics.json")
        save_json(tuning_results, self.output_dir / "tuning_results.json")

        if self.config["outputs"].get("save_predictions", True):
            save_predictions(
                test_df["delivery_id"],
                y_test,
                y_pred,
                self.output_dir / "predictions.csv",
                y_score=y_score,
            )

        pd.DataFrame(
            {
                "feature": preprocessor.feature_names,
                "coefficient": getattr(model, "weights", np.zeros(len(preprocessor.feature_names))),
            }
        ).sort_values("coefficient", key=lambda s: s.abs(), ascending=False).to_csv(
            self.output_dir / "feature_coefficients.csv", index=False
        )

        return run_summary
