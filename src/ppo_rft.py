import argparse
import json
import math
import os
import random
from typing import List, Dict

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM

from model import build_model
from utils import set_seed


def load_prompts(path: str) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            prompt = ex.get("prompt") or ex.get("text") or ex.get("instruction")
            if prompt:
                prompts.append(prompt)
    return prompts


def logprob_of_generated(model, input_ids, attention_mask, gen_len, with_grad: bool):
    # Compute log-prob of generated tokens only
    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        logp = torch.log_softmax(logits, dim=-1)
        tok_logp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Mask to only generated tokens (last gen_len tokens)
    seq_len = input_ids.size(1)
    start = seq_len - gen_len
    if gen_len <= 0:
        return tok_logp.sum(dim=1) * 0.0
    mask = torch.zeros_like(tok_logp)
    mask[:, start - 1 : seq_len - 1] = 1.0
    logp_sum = (tok_logp * mask).sum(dim=1)
    return logp_sum


def compute_reward(feedback_model, fb_tokenizer, prompts, responses, device, num_actions):
    texts = [f"Prompt: {p}\nResponse: {r}" for p, r in zip(prompts, responses)]
    enc = fb_tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = feedback_model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=-1)
        actions = torch.arange(num_actions, device=device, dtype=torch.float)
        # Expected rating in [0, num_actions-1]
        rating = (probs * actions).sum(dim=-1)
        reward = rating / max(1.0, float(num_actions - 1))
    return reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--policy-model", default="gpt2")
    parser.add_argument("--ref-model", default="gpt2")
    parser.add_argument("--feedback-model", default="distilroberta-base")
    parser.add_argument("--feedback-ckpt", required=True)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", default="./ppo_checkpoints")
    args = parser.parse_args()

    os.makedirs(args.save, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_tok = AutoTokenizer.from_pretrained(args.policy_model)
    if policy_tok.pad_token is None:
        policy_tok.pad_token = policy_tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(args.policy_model).to(device)
    ref = AutoModelForCausalLM.from_pretrained(args.ref_model).to(device)
    ref.eval()

    fb_tok = AutoTokenizer.from_pretrained(args.feedback_model)
    if fb_tok.pad_token is None:
        fb_tok.pad_token = fb_tok.eos_token if fb_tok.eos_token else fb_tok.unk_token
    feedback = build_model(
        model_name=args.feedback_model,
        num_actions=args.num_actions,
        dropout=0.1,
        from_scratch=False,
        vocab_size=len(fb_tok),
        max_len=512,
        d_model=384,
        n_heads=6,
        n_layers=6,
        d_ff=1536,
    ).to(device)
    ckpt = torch.load(args.feedback_ckpt, map_location=device)
    feedback.load_state_dict(ckpt["model"], strict=False)
    feedback.eval()

    prompts = load_prompts(args.prompts)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        random.shuffle(prompts)
        total_loss = 0.0

        # Snapshot old policy for PPO ratio
        old_policy = AutoModelForCausalLM.from_pretrained(args.policy_model).to(device)
        old_policy.load_state_dict(policy.state_dict())
        old_policy.eval()

        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i : i + args.batch_size]
            if not batch_prompts:
                continue

            enc = policy_tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            with torch.no_grad():
                gen_ids = policy.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=policy_tok.eos_token_id,
                )

            gen_len = gen_ids.size(1) - enc["input_ids"].size(1)
            responses = policy_tok.batch_decode(
                gen_ids[:, enc["input_ids"].size(1) :], skip_special_tokens=True
            )

            # Compute rewards using feedback model
            rewards = compute_reward(
                feedback, fb_tok, batch_prompts, responses, device, args.num_actions
            )
            rewards = rewards.detach()
            adv = rewards - rewards.mean()

            # Compute logprobs for generated tokens
            gen_attn = torch.ones_like(gen_ids)
            logp_new = logprob_of_generated(policy, gen_ids, gen_attn, gen_len, with_grad=True)
            logp_old = logprob_of_generated(old_policy, gen_ids, gen_attn, gen_len, with_grad=False)
            ratio = torch.exp(logp_new - logp_old)

            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps) * adv
            loss_ppo = -torch.min(unclipped, clipped).mean()

            # KL penalty against reference model
            logp_ref = logprob_of_generated(ref, gen_ids, gen_attn, gen_len, with_grad=False)
            kl = (logp_new - logp_ref).mean()
            loss = loss_ppo + args.kl_coef * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            total_loss += loss.item()

        ckpt_path = os.path.join(args.save, f"epoch_{epoch+1}.pt")
        torch.save({"model": policy.state_dict(), "args": vars(args)}, ckpt_path)
        print(f"epoch={epoch+1} loss={total_loss / max(1, len(prompts)):.4f}")


if __name__ == "__main__":
    main()
