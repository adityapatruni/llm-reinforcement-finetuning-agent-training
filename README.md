# llm_rft_agent

Minimal reinforcement fine-tuning (RFT) feedback agent for LLM responses.

This project trains a *feedback policy* that scores LLM responses given a prompt. The training is an offline, bandit-style REINFORCE objective on labeled data (e.g., ratings or pairwise preferences). The resulting feedback agent can later be plugged into RFT or RLHF pipelines.

## Project layout
- `src/train_feedback_agent.py` – main training script
- `src/model.py` – feedback policy model
- `src/data.py` – dataset loaders
- `src/utils.py` – helpers
- `src/preprocess_data.py` – dataset download + preprocessing
- `src/ppo_rft.py` – minimal PPO loop using the feedback model
- `docs/TECHNICAL_DOCS.md` – detailed documentation (design, algorithms, diagrams, usage)

## Data format
JSONL with one example per line. Required fields:

```
{"prompt": "...", "response": "...", "label": 3}
```

`label` can be integer in `[0, num_actions-1]` or float in `[0, num_actions-1]`.

Optional pairwise format:

```
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

Pairwise data is converted into two labeled examples (chosen gets higher label, rejected gets lower label).

## Quickstart

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/preprocess_data.py \
  --dataset hh-rlhf \
  --out-train data/train.jsonl \
  --out-eval data/eval.jsonl

python src/train_feedback_agent.py \
  --data data/train.jsonl \
  --eval data/eval.jsonl \
  --model distilroberta-base \
  --num-actions 5 \
  --epochs 3 \
  --batch-size 8
```

## Mixture mode
Combine multiple datasets with explicit weights:

```
python src/preprocess_data.py \
  --mixture hh-rlhf:0.5,ultrafeedback:0.25,pku-saferlhf:0.25 \
  --out-train data/train.jsonl \
  --out-eval data/eval.jsonl \
  --add-source
```

Or via JSON:

```
cat > data/mixture.json <<'JSON'
{"hh-rlhf": 0.5, "ultrafeedback": 0.25, "pku-saferlhf": 0.25}
JSON

python src/preprocess_data.py \
  --mixture-json data/mixture.json \
  --out-train data/train.jsonl \
  --out-eval data/eval.jsonl \
  --add-source
```

## Calibration and HH-RLHF prompt parsing
```
python src/preprocess_data.py \
  --dataset hh-rlhf \
  --hh-parse-prompt \
  --out-train data/train.jsonl \
  --out-eval data/eval.jsonl
```

```
python src/preprocess_data.py \
  --mixture hh-rlhf:0.5,ultrafeedback:0.5 \
  --calibrate \
  --out-train data/train.jsonl \
  --out-eval data/eval.jsonl
```

## PPO RFT loop (minimal)
```
python src/ppo_rft.py \
  --prompts data/prompts.jsonl \
  --policy-model gpt2 \
  --ref-model gpt2 \
  --feedback-model distilroberta-base \
  --feedback-ckpt checkpoints/epoch_1.pt \
  --num-actions 5 \
  --epochs 1 \
  --batch-size 2
```

## Tests
```
pytest
RUN_SLOW=1 pytest -k train_smoke
```

## From-scratch model
You can train a small Transformer encoder from scratch (tokenizer still uses a pretrained vocab).

```
python src/train_feedback_agent.py \
  --data path/to/train.jsonl \
  --eval path/to/eval.jsonl \
  --from-scratch \
  --model distilroberta-base \
  --d-model 384 --n-heads 6 --n-layers 6 --d-ff 1536 \
  --num-actions 5 --epochs 5 --batch-size 8
```

## Notes
- This is offline RL (contextual bandit) with REINFORCE. If you have supervised labels, you can also enable `--supervised-loss-weight`.
- The agent outputs a discrete feedback action (e.g., 0–4). You can map actions to rewards for downstream RFT.
