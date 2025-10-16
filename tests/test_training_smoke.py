from pathlib import Path

from mlops_diabetes.train_v0_1 import train


def test_train_smoke(tmp_path: Path):
    out = train(tmp_path)
    assert Path(out["model_path"]).exists()
    assert Path(out["metrics_path"]).exists()
    assert "rmse" in out["metrics"]
