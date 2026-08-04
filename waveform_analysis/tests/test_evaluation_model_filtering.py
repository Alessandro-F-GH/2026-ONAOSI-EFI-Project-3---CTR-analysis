from pathlib import Path
from types import SimpleNamespace

from ml_pipeline.evaluation import TrainedModel, discover_models


def test_discover_models_filters_model_type(monkeypatch, tmp_path):
    search = tmp_path / "train"
    search.mkdir()
    (search / "a").mkdir()
    (search / "b").mkdir()
    (search / "a" / "training_summary.json").write_text("{}")
    (search / "b" / "training_summary.json").write_text("{}")

    models = {
        "a": TrainedModel("old_encoder", "constructive_identity_encoder", Path("a.pt"), 2.0, tmp_path, "none"),
        "b": TrainedModel("svr", "linear_svr", Path("b.pt"), 1.0, tmp_path, "differentiate"),
    }

    def fake_model(path):
        return models[path.parent.name]

    monkeypatch.setattr("ml_pipeline.evaluation._model_from_summary", fake_model)
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    found = discover_models({"models": [], "model_search_dir": search, "model_types": ["linear_svr"]}, logger)
    assert [model.model_name for model in found] == ["svr"]
