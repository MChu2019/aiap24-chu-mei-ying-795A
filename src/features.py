import numpy as np
import pandas as pd


def clean_deliveries(deliveries):
    df = deliveries.copy()

    df["branch_clean"] = (
        df["branch"].astype("string").str.strip().str.title().replace({"Noth": "North", "Cnetral": "Central"})
    )
    df["parcel_category_clean"] = (
        df["parcel_category"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"over-sized": "oversized", "refrig.": "refrigerated"})
    )
    df["vehicle_type_clean"] = df["vehicle_type"].astype("string").str.strip().str.lower()
    df["payment_method_clean"] = df["payment_method"].astype("string").str.strip().str.lower()
    df["delivery_priority_clean"] = df["delivery_priority"].astype("string").str.strip().str.title()

    for column in ["booking_datetime", "pickup_datetime", "promised_delivery_datetime", "delivery_datetime"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    for column in [
        "distance_km",
        "parcel_weight_kg",
        "parcel_value_sgd",
        "num_stops_on_route",
        "driver_experience_months",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["pickup_wait_hours"] = (df["pickup_datetime"] - df["booking_datetime"]).dt.total_seconds() / 3600
    df["promised_delivery_hours"] = (
        df["promised_delivery_datetime"] - df["booking_datetime"]
    ).dt.total_seconds() / 3600
    df["actual_delivery_hours"] = (df["delivery_datetime"] - df["booking_datetime"]).dt.total_seconds() / 3600
    df["lateness_hours"] = (df["delivery_datetime"] - df["promised_delivery_datetime"]).dt.total_seconds() / 3600
    df["booking_day_name"] = df["booking_datetime"].dt.day_name()

    return df


def clean_feedback(feedback):
    df = feedback.copy()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["feedback_datetime"] = pd.to_datetime(df["feedback_datetime"], errors="coerce")
    return df


def build_training_frame(deliveries, feedback, config):
    deliveries_clean = clean_deliveries(deliveries)
    feedback_clean = clean_feedback(feedback)
    joined = deliveries_clean.merge(feedback_clean, on="delivery_id", how="left", suffixes=("", "_feedback"))

    task = config["task"]["type"]
    if task == "classification":
        threshold = config["task"].get("low_rating_threshold", 3)
        joined = joined[joined["rating"].notna()].copy()
        joined["target"] = (joined["rating"] <= threshold).astype(int)
    elif task == "regression":
        joined = joined[joined["rating"].notna()].copy()
        joined["target"] = joined["rating"].astype(float)
    else:
        raise ValueError(f"Unsupported task type: {task}")

    return joined.reset_index(drop=True)


class FeaturePreprocessor:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.numeric_medians = {}
        self.numeric_means = {}
        self.numeric_stds = {}
        self.category_levels = {}
        self.feature_names = None

    def fit(self, df):
        for column in self.numeric_features:
            values = pd.to_numeric(df[column], errors="coerce")
            self.numeric_medians[column] = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(self.numeric_medians[column])
            self.numeric_means[column] = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.numeric_stds[column] = std if std > 0 else 1.0

        for column in self.categorical_features:
            levels = (
                df[column]
                .astype("string")
                .fillna("missing")
                .str.strip()
                .replace("", "missing")
                .dropna()
                .unique()
                .tolist()
            )
            self.category_levels[column] = sorted(str(level) for level in levels)

        self.feature_names = self.numeric_features.copy()
        for column in self.categorical_features:
            self.feature_names.extend([f"{column}__{level}" for level in self.category_levels[column]])
        return self

    def transform(self, df):
        arrays = []

        for column in self.numeric_features:
            values = pd.to_numeric(df[column], errors="coerce").fillna(self.numeric_medians[column])
            values = (values - self.numeric_means[column]) / self.numeric_stds[column]
            arrays.append(values.to_numpy(dtype=float).reshape(-1, 1))

        for column in self.categorical_features:
            values = df[column].astype("string").fillna("missing").str.strip().replace("", "missing")
            for level in self.category_levels[column]:
                arrays.append((values == level).astype(float).to_numpy().reshape(-1, 1))

        return np.hstack(arrays) if arrays else np.empty((len(df), 0))

    def fit_transform(self, df):
        return self.fit(df).transform(df)
