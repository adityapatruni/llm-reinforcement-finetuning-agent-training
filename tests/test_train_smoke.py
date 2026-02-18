import json
import os
import sys
import pytest


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.mark.skipif(os.environ.get("RUN_SLOW") != "1", reason="Set RUN_SLOW=1 to run")
def test_train_smoke(tmp_path):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"

    write_jsonl(
        train_path,
        [
            {"prompt": "p", "response": "r", "label": 2},
            {"prompt": "p2", "response": "r2", "label": 3},
        ],
    )
    write_jsonl(
        eval_path,
        [
            {"prompt": "p", "response": "r", "label": 2},
        ],
    )

    sys.argv = [
        "train_feedback_agent.py",
        "--data",
        str(train_path),
        "--eval",
        str(eval_path),
        "--model",
        "distilroberta-base",
        "--num-actions",
        "5",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--max-len",
        "16",
    ]

    from train_feedback_agent import train

    train()
