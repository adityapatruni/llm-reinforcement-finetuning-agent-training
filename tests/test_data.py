import json
from data import load_jsonl


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_jsonl_labeled(tmp_path):
    p = tmp_path / "data.jsonl"
    write_jsonl(p, [{"prompt": "p", "response": "r", "label": 3}])
    data = load_jsonl(str(p), num_actions=5)
    assert len(data) == 1
    assert "text" in data[0]
    assert data[0]["label"] == 3.0


def test_load_jsonl_pairwise(tmp_path):
    p = tmp_path / "data.jsonl"
    write_jsonl(
        p,
        [
            {
                "prompt": "p",
                "chosen": "good",
                "rejected": "bad",
            }
        ],
    )
    data = load_jsonl(str(p), num_actions=5)
    assert len(data) == 2
    labels = sorted([d["label"] for d in data])
    assert labels == [0.0, 4.0]
