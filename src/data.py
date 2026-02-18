import json
from typing import List, Dict, Any


def _normalize_label(label, num_actions: int):
    if isinstance(label, float):
        return max(0.0, min(float(label), float(num_actions - 1)))
    return float(max(0, min(int(label), num_actions - 1)))


def load_jsonl(path: str, num_actions: int) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "prompt" in ex and "response" in ex and "label" in ex:
                data.append(
                    {
                        "text": f"Prompt: {ex['prompt']}\nResponse: {ex['response']}",
                        "label": _normalize_label(ex["label"], num_actions),
                    }
                )
            elif "prompt" in ex and "chosen" in ex and "rejected" in ex:
                hi = num_actions - 1
                lo = 0
                data.append(
                    {
                        "text": f"Prompt: {ex['prompt']}\nResponse: {ex['chosen']}",
                        "label": float(hi),
                    }
                )
                data.append(
                    {
                        "text": f"Prompt: {ex['prompt']}\nResponse: {ex['rejected']}",
                        "label": float(lo),
                    }
                )
            else:
                raise ValueError("Unsupported JSONL schema. Expected labeled or pairwise format.")
    return data
