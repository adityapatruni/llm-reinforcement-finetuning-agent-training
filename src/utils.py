import math
import torch
from torch.distributions import Categorical


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def policy_sample(logits, temperature: float):
    if temperature <= 0:
        return logits.argmax(dim=-1), torch.zeros(logits.size(0), device=logits.device)
    dist = Categorical(logits=logits / temperature)
    actions = dist.sample()
    logp = dist.log_prob(actions)
    return actions, logp


def normalize_rewards(labels, num_actions: int):
    # map [0, num_actions-1] -> [0, 1]
    return labels / max(1.0, float(num_actions - 1))


def moving_average_update(prev, value, beta: float = 0.9):
    if prev is None:
        return value
    return beta * prev + (1 - beta) * value
