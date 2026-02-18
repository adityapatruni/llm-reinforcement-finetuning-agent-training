import torch
from model import ScratchEncoder


def test_scratch_encoder_forward():
    model = ScratchEncoder(
        vocab_size=100,
        num_actions=5,
        max_len=32,
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
    )
    input_ids = torch.randint(0, 100, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (2, 5)
