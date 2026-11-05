import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wpf_benchmark.models import get_model


ROOT = Path(__file__).parents[1]


def test_toto_is_registered_without_loading_optional_weights():
    model = get_model("toto")()
    assert model.params["model_id"] == "Datadog/Toto-Open-Base-1.0"
    assert model.params["device"] == "auto"
    assert model.params["num_samples"] is None


def test_toto_validates_the_target_and_horizon():
    model = get_model("toto")()
    model.validate_context(SimpleNamespace(target_input_index=0, L=144, H=36))

    with pytest.raises(ValueError, match="target_input"):
        model.validate_context(SimpleNamespace(target_input_index=None, L=144, H=36))


def test_toto_config_is_optional_and_explicit():
    config = json.loads((ROOT / "configs/models/toto.json").read_text(encoding="utf-8"))
    assert config["model"] == "toto"
    assert config["params"]["model_id"] == "Datadog/Toto-Open-Base-1.0"
    assert config["params"]["num_samples"] is None
