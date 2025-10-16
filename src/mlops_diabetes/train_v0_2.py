from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_ORDER, load_data
from .metrics import rmse, to_metrics_dict

MODEL_VERSION = "v0_2"


def train(artifacts_dir: Path) -> dict:
    X_train, y_train, X_test, y_test = load_data()

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0, random_state=42)),
        ]
    )
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = to_metrics_dict(rmse(y_test.to_numpy(), y_pred))

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / f"model_{MODEL_VERSION}.joblib"
    joblib.dump({"model": pipe, "feature_order": FEATURE_ORDER, "model_version": MODEL_VERSION}, model_path)

    metrics_path = artifacts_dir / f"metrics_{MODEL_VERSION}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f)

    return {
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    args = parser.parse_args()

    out = train(Path(args.artifacts_dir))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
