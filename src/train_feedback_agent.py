import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import load_jsonl
from model import build_model
from utils import set_seed, policy_sample, normalize_rewards, moving_average_update


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def collate_fn(batch, tokenizer, max_len):
    texts = [x["text"] for x in batch]
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    return Batch(enc["input_ids"], enc["attention_mask"], labels)


def eval_loop(model, dataloader, device, num_actions):
    model.eval()
    correct = 0
    total = 0
    rewards = []
    preds = []
    labels_all = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)
            logits = model(input_ids, attention_mask)
            pred = logits.argmax(dim=-1).float()
            labels_round = labels.round()
            correct += (pred == labels_round).sum().item()
            total += labels.size(0)
            rewards.append(normalize_rewards(labels, num_actions).cpu().numpy())
            preds.append(pred.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    acc = correct / max(1, total)
    rewards = np.concatenate(rewards) if rewards else np.array([])
    preds = np.concatenate(preds) if preds else np.array([])
    labels_all = np.concatenate(labels_all) if labels_all else np.array([])
    return acc, rewards, preds, labels_all


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--model", default="distilroberta-base")
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--supervised-loss-weight", type=float, default=0.0)
    parser.add_argument("--baseline-beta", type=float, default=0.9)
    parser.add_argument("--save", default="./checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    train_data = load_jsonl(args.data, args.num_actions)
    eval_data = load_jsonl(args.eval, args.num_actions)

    train_dl = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len),
    )
    eval_dl = DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_len),
    )

    model = build_model(
        model_name=args.model,
        num_actions=args.num_actions,
        dropout=args.dropout,
        from_scratch=args.from_scratch,
        vocab_size=len(tokenizer),
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()

    baseline = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dl:
            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            logits = model(input_ids, attention_mask)

            actions, logp = policy_sample(logits, args.temperature)
            rewards = normalize_rewards(labels, args.num_actions)
            rewards = rewards.to(device)
            baseline = moving_average_update(baseline, rewards.mean().item(), args.baseline_beta)
            adv = rewards - (baseline if baseline is not None else 0.0)

            loss_rl = -(adv.detach() * logp).mean()

            loss = loss_rl
            if args.supervised_loss_weight > 0:
                labels_int = labels.round().long().clamp(0, args.num_actions - 1)
                loss_ce = ce_loss(logits, labels_int)
                loss = loss + args.supervised_loss_weight * loss_ce

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()

        acc, rewards_eval, preds, labels_all = eval_loop(model, eval_dl, device, args.num_actions)
        ckpt = os.path.join(args.save, f"epoch_{epoch+1}.pt")
        torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)
        print(
            f"epoch={epoch+1} loss={total_loss/ max(1, len(train_dl)):.4f} eval_acc={acc:.4f}"
        )


if __name__ == "__main__":
    train()
