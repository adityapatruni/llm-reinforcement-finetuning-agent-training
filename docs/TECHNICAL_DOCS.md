# LLM RFT Feedback Agent: Technical Documentation

## 1. Overview
This project builds a **reinforcement fine-tuning (RFT) feedback agent** that evaluates LLM responses. It is a contextual-bandit policy that outputs a discrete feedback action (e.g., 0–4) for a given `(prompt, response)` pair. The model is trained with a **REINFORCE** objective on offline labels or pairwise preferences.

The feedback agent can be used in two ways:
1. **Reward/feedback model** for a separate LLM policy during RFT or RLHF.
2. **Standalone evaluator** for scoring or filtering responses.

## 2. System Architecture

**High-level pipeline**
```
        ┌──────────────────────┐
        │ JSONL Training Data  │
        │ (prompt/response/    │
        │  label or pairwise)  │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Tokenizer            │
        │ (HF AutoTokenizer)   │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Feedback Policy      │
        │  - Pretrained encoder│
        │  - or Scratch encoder│
        └──────────┬───────────┘
                   │ logits
                   ▼
        ┌──────────────────────┐
        │ Policy Sampling      │
        │ (Categorical)        │
        └──────────┬───────────┘
                   │ action, logp
                   ▼
        ┌──────────────────────┐
        │ REINFORCE Loss        │
        │ - baseline            │
        │ - optional CE loss    │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Optimizer (AdamW)    │
        └──────────────────────┘
```

**Component diagram**
```
┌──────────────────────────────┐
│ src/train_feedback_agent.py  │
│  - orchestration             │
│  - RL loop + eval            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ src/model.py                 │
│  - FeedbackPolicy            │
│  - ScratchEncoder            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ src/data.py                  │
│  - JSONL loading             │
│  - pairwise -> labels        │
└──────────────────────────────┘

┌──────────────────────────────┐
│ src/utils.py                 │
│  - REINFORCE helpers         │
│  - reward normalization      │
└──────────────────────────────┘
```

## 3. Implementation Details

### 3.1 Data processing
**File**: `src/data.py`

Supported schemas:
- Labeled: `{"prompt": ..., "response": ..., "label": 3}`
- Pairwise: `{"prompt": ..., "chosen": ..., "rejected": ...}`

Pairwise data is converted to labeled examples by assigning:
- `chosen -> num_actions - 1`
- `rejected -> 0`

This allows a uniform REINFORCE objective over discrete actions.

### 3.2 Data-prep pipeline design
**File**: `src/preprocess_data.py`

This pipeline downloads open-source preference datasets via Hugging Face `datasets` and normalizes them to a common JSONL schema used by training.

**Pipeline stages**
1. **Ingest**: `load_dataset(...)` pulls the dataset.
2. **Split**: Uses `validation`/`test` if available, otherwise `train_test_split`.
3. **Normalize**: Convert each dataset record into one of two schemas:
   - Pairwise: `{prompt, chosen, rejected}`
   - Labeled: `{prompt, response, label}`
4. **Filter**: Drop samples with missing fields.
5. **Optional sampling**: `--max-samples`, `--max-eval-samples`.
6. **Write JSONL**: `data/train.jsonl`, `data/eval.jsonl`.

**Datasets supported**
- `Anthropic/hh-rlhf` (pairwise `chosen/rejected`)
- `stanfordnlp/SHP` (pairwise derived from `human_ref_A/B` and scores)
- `zhengr/UltraFeedback` (labeled using rating aggregation)
- `PKU-Alignment/PKU-SafeRLHF` (pairwise with `better_response_id`)

**Normalization rules**
- Pairwise datasets are kept as `{prompt, chosen, rejected}` and later converted to labels by `src/data.py`.
- UltraFeedback ratings are averaged across annotations and scaled to `[0, num_actions-1]`.
 - HH-RLHF supports prompt/response parsing via `--hh-parse-prompt`.

**Mixture mode**
`src/preprocess_data.py` supports a unified mixture mode to combine datasets with weights:
```
python src/preprocess_data.py \
  --mixture hh-rlhf:0.5,ultrafeedback:0.25,pku-saferlhf:0.25 \
  --out-train data/train.jsonl \
  --out-eval data/eval.jsonl \
  --add-source
```
JSON alternative:
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
Implementation details:
- Each dataset is loaded and normalized independently.
- A weighted sampler draws from each dataset with replacement.
- `--max-samples` and `--max-eval-samples` cap the total samples.
- `--add-source` adds a `source` field for analysis and debugging.

**Calibration**
When `--calibrate` is enabled in mixture mode:
- Labeled examples are grouped by `source`.
- Per-source mean/std are computed.
- Labels are rescaled to match the global mean/std and clipped to `[0, num_actions-1]`.
This reduces reward-scale mismatches between datasets.

### 3.2 Model design
**File**: `src/model.py`

Two models share the same interface:

1. **FeedbackPolicy (pretrained)**
- Uses `transformers.AutoModel`.
- Mean-pools token embeddings.
- Linear layer outputs action logits.

2. **ScratchEncoder**
- Token + positional embeddings.
- `nn.TransformerEncoder` stack.
- Mean pooling and linear policy head.

Both are small enough to train on a single GPU and return a vector of size `num_actions`.

### 3.3 RL algorithm (REINFORCE)
**File**: `src/train_feedback_agent.py`

Objective:
```
J(θ) = E[ (R - b) * log πθ(a | x) ]
```
Where:
- `R` is normalized reward from labels.
- `b` is a moving average baseline for variance reduction.
- `πθ` is the policy distribution over actions.

This is implemented as:
- Sample action `a` from `Categorical(logits)`.
- Compute `loss_rl = -(R - b) * logp(a)`.
- Optionally add supervised CE loss for stability:
```
loss = loss_rl + λ * CE(logits, label)
```

### 3.4 Design patterns
- **Strategy**: `build_model(...)` chooses pretrained or scratch model based on a flag.
- **Separation of concerns**: data loading, model definitions, and training logic are isolated.
- **Configuration via CLI**: all hyperparameters are exposed as arguments to keep the code reproducible.

## 4. Technologies and Dependencies
- **PyTorch**: model definition and training.
- **Hugging Face Transformers**: pretrained encoders and tokenizers.
- **NumPy**: evaluation utilities.
- **scikit-learn**: optional for future metrics (not used directly).

Dependencies are declared in `requirements.txt`.

## 5. User Reference

### 5.1 Install
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Prepare data
Create `train.jsonl` and `eval.jsonl`:
```
{"prompt": "...", "response": "...", "label": 3}
{"prompt": "...", "chosen": "good response", "rejected": "bad response"}
```

Or use the preprocessing script:
```
python src/preprocess_data.py \
  --dataset hh-rlhf \
  --out-train data/train.jsonl \
  --out-eval data/eval.jsonl
```

### 5.3 Train with pretrained encoder
```
python src/train_feedback_agent.py \
  --data train.jsonl \
  --eval eval.jsonl \
  --model distilroberta-base \
  --num-actions 5 \
  --epochs 3 \
  --batch-size 8
```

### 5.4 Train from scratch
```
python src/train_feedback_agent.py \
  --data train.jsonl \
  --eval eval.jsonl \
  --from-scratch \
  --model distilroberta-base \
  --d-model 384 \
  --n-heads 6 \
  --n-layers 6 \
  --d-ff 1536 \
  --num-actions 5
```

### 5.5 Recommended hyperparameters
- Start with `--batch-size 4` or `8`.
- Use `--max-len 256` for faster throughput.
- Enable `--supervised-loss-weight 0.2` when labels are high-quality.

### 5.6 PPO RFT loop (minimal)
`src/ppo_rft.py` provides a small PPO-style loop that:
1. Generates responses from a policy LLM.
2. Scores each response with the feedback model (expected reward).
3. Optimizes the policy using PPO clipping and a KL penalty to a reference model.

Example:
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

## 6. AMD Radeon RX 9070 RT Guidance
PyTorch on AMD GPUs uses ROCm. Practical guidance:
- Use a **ROCm-enabled PyTorch build** compatible with your ROCm version.
- If the GPU is not supported by ROCm, fall back to CPU or use cloud GPUs.
- If you hit memory limits, reduce `--batch-size` and `--max-len`.

## 7. GCP Free-Tier Fallback (Action List)
If local ROCm support is not viable:
1. Use **Colab free tier** for quick experiments.
2. Use **GCP free-tier CPU VM** for preprocessing and small tests.
3. Use **short paid GPU sessions** only for training runs; keep batch sizes small.
4. Save checkpoints to Google Drive or Google Cloud Storage.

## 8. Extending the Project
- Replace accuracy with correlation or preference accuracy metrics.
- Add a pairwise ranking loss for preference data.
- Integrate with an RFT loop (e.g., PPO) to train an LLM policy directly.

## 9. Tests
Basic unit tests are in `tests/`. Run:
```
pytest
RUN_SLOW=1 pytest -k train_smoke
```

## 10. Future Feature Additions (Not Implemented)
- **Production-grade PPO**: value head, advantage estimation (GAE), reward normalization, adaptive KL target.
- **Rich evaluation**: preference accuracy, rank correlation, calibration metrics, dataset-by-dataset diagnostics.
- **Dataset version pinning**: cache and pin specific dataset revisions for reproducibility.
- **Reward model ensembles**: multiple feedback models with aggregation for stability.
- **Safety filters**: guardrails for unsafe content in both training and inference.
