import argparse
import json
import os
import random
from typing import Iterable, Dict, Any, List, Tuple

from datasets import load_dataset


def write_jsonl(path: str, records: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_dataset(ds, eval_size: float, seed: int):
    if "validation" in ds:
        return ds["train"], ds["validation"]
    if "test" in ds:
        return ds["train"], ds["test"]
    tmp = ds["train"].train_test_split(test_size=eval_size, seed=seed)
    return tmp["train"], tmp["test"]


def limit_samples(records: List[Dict[str, Any]], max_samples: int, seed: int):
    if max_samples is None or max_samples <= 0:
        return records
    rng = random.Random(seed)
    if len(records) <= max_samples:
        return records
    return rng.sample(records, max_samples)


def prep_hh_rlhf(args):
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=args.hh_data_dir)
    train_split, eval_split = split_dataset(ds, args.eval_size, args.seed)

    def _split_dialog(text: str) -> Tuple[str, str]:
        marker = "\n\nAssistant:"
        if marker not in text:
            return "", text
        parts = text.split(marker)
        prompt = marker.join(parts[:-1]).strip()
        response = parts[-1].strip()
        return prompt, response

    def convert(split):
        out = []
        for ex in split:
            chosen = ex.get("chosen")
            rejected = ex.get("rejected")
            if not chosen or not rejected:
                continue
            if args.hh_parse_prompt:
                p_c, r_c = _split_dialog(chosen)
                p_r, r_r = _split_dialog(rejected)
                # prefer prompt from chosen if available
                prompt = p_c if p_c else p_r
                out.append({"prompt": prompt, "chosen": r_c, "rejected": r_r})
            else:
                out.append({"prompt": "", "chosen": chosen, "rejected": rejected})
        return out

    train = limit_samples(convert(train_split), args.max_samples, args.seed)
    eval_ = limit_samples(convert(eval_split), args.max_eval_samples, args.seed + 1)
    return train, eval_


def prep_shp(args):
    ds = load_dataset("stanfordnlp/SHP")
    train_split, eval_split = split_dataset(ds, args.eval_size, args.seed)

    def convert(split):
        out = []
        for ex in split:
            prompt = ex.get("history", "")
            a = ex.get("human_ref_A")
            b = ex.get("human_ref_B")
            if not prompt or not a or not b:
                continue
            score_a = ex.get("score_A", 0)
            score_b = ex.get("score_B", 0)
            if score_a >= score_b:
                chosen, rejected = a, b
            else:
                chosen, rejected = b, a
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        return out

    train = limit_samples(convert(train_split), args.max_samples, args.seed)
    eval_ = limit_samples(convert(eval_split), args.max_eval_samples, args.seed + 1)
    return train, eval_


def _extract_ultra_ratings(annotations: Dict[str, Any]) -> List[float]:
    ratings = []
    if not isinstance(annotations, dict):
        return ratings
    for _, entries in annotations.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            rating = entry.get("Rating")
            if rating is None:
                continue
            try:
                ratings.append(float(rating))
            except Exception:
                continue
    return ratings


def prep_ultrafeedback(args):
    ds = load_dataset("zhengr/UltraFeedback")
    train_split, eval_split = split_dataset(ds, args.eval_size, args.seed)

    def convert(split):
        out = []
        for ex in split:
            prompt = ex.get("instruction")
            completions = ex.get("completions")
            if not prompt or not completions:
                continue
            for comp in completions:
                response = comp.get("response")
                annotations = comp.get("annotations")
                if not response:
                    continue
                ratings = _extract_ultra_ratings(annotations)
                if not ratings:
                    continue
                score = sum(ratings) / len(ratings)
                score_norm = (score - args.ultra_min_rating) / max(
                    1e-6, (args.ultra_max_rating - args.ultra_min_rating)
                )
                score_norm = max(0.0, min(1.0, score_norm))
                label = score_norm * float(args.num_actions - 1)
                out.append({"prompt": prompt, "response": response, "label": label})
        return out

    train = limit_samples(convert(train_split), args.max_samples, args.seed)
    eval_ = limit_samples(convert(eval_split), args.max_eval_samples, args.seed + 1)
    return train, eval_


def prep_pku_saferlhf(args):
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", name=args.pku_name)
    train_split, eval_split = split_dataset(ds, args.eval_size, args.seed)

    def convert(split):
        out = []
        for ex in split:
            prompt = ex.get("prompt")
            r0 = ex.get("response_0")
            r1 = ex.get("response_1")
            better = ex.get("better_response_id")
            if prompt is None or r0 is None or r1 is None or better is None:
                continue
            if int(better) == 0:
                chosen, rejected = r0, r1
            else:
                chosen, rejected = r1, r0
            out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        return out

    train = limit_samples(convert(train_split), args.max_samples, args.seed)
    eval_ = limit_samples(convert(eval_split), args.max_eval_samples, args.seed + 1)
    return train, eval_


def get_dataset(name: str, args) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if name == "hh-rlhf":
        return prep_hh_rlhf(args)
    if name == "shp":
        return prep_shp(args)
    if name == "ultrafeedback":
        return prep_ultrafeedback(args)
    if name == "pku-saferlhf":
        return prep_pku_saferlhf(args)
    raise ValueError(f"Unknown dataset: {name}")


def parse_mixture(spec: str) -> List[Tuple[str, float]]:
    parts = []
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"Invalid mixture entry: {raw}")
        name, weight = raw.split(":", 1)
        name = name.strip()
        weight = float(weight.strip())
        if weight <= 0:
            continue
        parts.append((name, weight))
    if not parts:
        raise ValueError("Mixture spec produced no valid entries.")
    return parts


def parse_mixture_json(path: str) -> List[Tuple[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("Mixture JSON must be an object mapping dataset -> weight.")
    parts = []
    for name, weight in obj.items():
        if weight is None:
            continue
        w = float(weight)
        if w <= 0:
            continue
        parts.append((name, w))
    if not parts:
        raise ValueError("Mixture JSON produced no valid entries.")
    return parts


def weighted_sample(
    pools: List[List[Dict[str, Any]]],
    weights: List[float],
    total: int,
    seed: int,
    add_source: bool,
    sources: List[str],
):
    rng = random.Random(seed)
    out = []
    # Build cumulative weights for sampling
    total_w = sum(weights)
    cum = []
    acc = 0.0
    for w in weights:
        acc += w / total_w
        cum.append(acc)

    for _ in range(total):
        r = rng.random()
        idx = 0
        for i, c in enumerate(cum):
            if r <= c:
                idx = i
                break
        pool = pools[idx]
        if not pool:
            continue
        ex = pool[rng.randrange(len(pool))]
        if add_source:
            ex = dict(ex)
            ex["source"] = sources[idx]
        out.append(ex)
    return out


def calibrate_labels(records: List[Dict[str, Any]], num_actions: int):
    # Calibrate labeled samples per source to match global mean/std.
    labeled = [r for r in records if "label" in r and "source" in r]
    if not labeled:
        return records

    # Compute per-source stats
    by_src: Dict[str, List[float]] = {}
    for r in labeled:
        by_src.setdefault(r["source"], []).append(float(r["label"]))

    src_stats = {}
    for src, vals in by_src.items():
        mean = sum(vals) / max(1, len(vals))
        var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals))
        std = max(1e-6, var ** 0.5)
        src_stats[src] = (mean, std)

    all_vals = [float(r["label"]) for r in labeled]
    g_mean = sum(all_vals) / max(1, len(all_vals))
    g_var = sum((v - g_mean) ** 2 for v in all_vals) / max(1, len(all_vals))
    g_std = max(1e-6, g_var ** 0.5)

    for r in labeled:
        mean, std = src_stats[r["source"]]
        v = float(r["label"])
        v = (v - mean) / std * g_std + g_mean
        v = max(0.0, min(float(num_actions - 1), v))
        r["label"] = v
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["hh-rlhf", "shp", "ultrafeedback", "pku-saferlhf"],
    )
    parser.add_argument(
        "--mixture",
        help="Comma-separated dataset:weight list, e.g. hh-rlhf:0.5,ultrafeedback:0.5",
    )
    parser.add_argument(
        "--mixture-json",
        help="Path to JSON mapping dataset -> weight, e.g. {\"hh-rlhf\":0.5,\"ultrafeedback\":0.5}",
    )
    parser.add_argument("--out-train", required=True)
    parser.add_argument("--out-eval", required=True)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--eval-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--hh-data-dir", default="helpful-base")
    parser.add_argument("--hh-parse-prompt", action="store_true")
    parser.add_argument("--pku-name", default="default")
    parser.add_argument("--ultra-min-rating", type=float, default=1.0)
    parser.add_argument("--ultra-max-rating", type=float, default=5.0)
    parser.add_argument("--add-source", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    if args.mixture and args.mixture_json:
        raise ValueError("Use only one of --mixture or --mixture-json")

    if args.mixture or args.mixture_json:
        mix = parse_mixture(args.mixture) if args.mixture else parse_mixture_json(args.mixture_json)
        names = [n for n, _ in mix]
        weights = [w for _, w in mix]
        datasets = [get_dataset(n, args) for n in names]
        train_pools = [d[0] for d in datasets]
        eval_pools = [d[1] for d in datasets]

        total_train = args.max_samples if args.max_samples > 0 else sum(
            len(p) for p in train_pools
        )
        total_eval = args.max_eval_samples if args.max_eval_samples > 0 else sum(
            len(p) for p in eval_pools
        )

        add_source = args.add_source or args.calibrate
        train = weighted_sample(
            train_pools, weights, total_train, args.seed, add_source, names
        )
        eval_ = weighted_sample(
            eval_pools,
            weights,
            total_eval,
            args.seed + 1,
            add_source,
            names,
        )

        if args.calibrate:
            train = calibrate_labels(train, args.num_actions)
            eval_ = calibrate_labels(eval_, args.num_actions)

        if add_source and not args.add_source:
            for r in train:
                r.pop("source", None)
            for r in eval_:
                r.pop("source", None)
    else:
        if not args.dataset:
            raise ValueError("Must provide --dataset or --mixture")
        train, eval_ = get_dataset(args.dataset, args)

    write_jsonl(args.out_train, train)
    write_jsonl(args.out_eval, eval_)

    print(f"Wrote train={len(train)} eval={len(eval_)}")


if __name__ == "__main__":
    main()
