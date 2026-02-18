import json
from preprocess_data import parse_mixture, parse_mixture_json


def test_parse_mixture():
    mix = parse_mixture("hh-rlhf:0.5,ultrafeedback:0.25")
    assert mix[0][0] == "hh-rlhf"
    assert mix[1][0] == "ultrafeedback"


def test_parse_mixture_json(tmp_path):
    p = tmp_path / "mix.json"
    p.write_text(json.dumps({"hh-rlhf": 0.5, "ultrafeedback": 0.5}))
    mix = parse_mixture_json(str(p))
    names = sorted([m[0] for m in mix])
    assert names == ["hh-rlhf", "ultrafeedback"]
