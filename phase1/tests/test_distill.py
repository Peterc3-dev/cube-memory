"""Smoke test for distill().

Builds two copies of a stub model (the same one used in test_swap.py),
swaps 25% of the student's FFN, runs distill() for a handful of steps
on a fake dataloader, and verifies:
  1. KL loss trends down (allowing some noise — softer threshold).
  2. Teacher params are unchanged after training.
"""
from __future__ import annotations

import copy
import logging
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(__file__.rsplit("/", 2)[0]))

from distill import distill
from swap_ffn import swap_ffn_modules


# ----------------------------------------------------------------------
# Stub model with a logits-shaped output to drive KL distillation.
# ----------------------------------------------------------------------

class StubBlock(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.mlp = nn.Linear(d_in, d_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class StubInner(nn.Module):
    def __init__(self, d_in: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([StubBlock(d_in) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class StubConfig:
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class StubOutput:
    def __init__(self, logits: torch.Tensor):
        self.logits = logits


class StubCausalLM(nn.Module):
    """Mimics the call signature of an HF causal LM."""

    def __init__(self, vocab_size: int, d_in: int, n_layers: int):
        super().__init__()
        self.config = StubConfig(d_in, vocab_size)
        self.embed = nn.Embedding(vocab_size, d_in)
        self.model = StubInner(d_in, n_layers)
        self.lm_head = nn.Linear(d_in, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kw) -> StubOutput:
        h = self.embed(input_ids)
        h = self.model(h)
        return StubOutput(self.lm_head(h))


# ----------------------------------------------------------------------
# Fake dataloader: yields random input_ids of shape (2, 16).
# ----------------------------------------------------------------------

class FakeLoader:
    def __init__(self, n_batches: int, vocab_size: int, batch: int = 2, seq: int = 16, seed: int = 0):
        self.n = n_batches
        self.vocab = vocab_size
        self.b = batch
        self.s = seq
        self.seed = seed

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.n):
            ids = torch.randint(0, self.vocab, (self.b, self.s), generator=g)
            mask = torch.ones_like(ids)
            yield {"input_ids": ids, "attention_mask": mask}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    torch.manual_seed(0)

    VOCAB, D, N = 64, 32, 8
    teacher = StubCausalLM(VOCAB, D, N)
    student = copy.deepcopy(teacher)

    # Swap 25% of student FFN -> 2 layers.
    cube_kwargs = {"d_codebook": 32, "d_value": 16, "m": 8, "p": 3, "n_slots": 64}
    student, replaced = swap_ffn_modules(student, fraction=0.25, cube_kwargs=cube_kwargs)
    assert len(replaced) == 2, f"expected 2 swapped layers, got {replaced}"

    # Snapshot teacher params for the no-mutation check.
    teacher_before = {n: p.detach().clone() for n, p in teacher.named_parameters()}

    loader = FakeLoader(n_batches=200, vocab_size=VOCAB, batch=2, seq=16, seed=0)

    metrics = distill(
        teacher,
        student,
        loader,
        lr_new=1e-3,
        lr_old=1e-4,
        steps=10,
        grad_accum=1,
        kl_temperature=2.0,
        log_every=2,
        eval_every=0,         # disable eval
        eval_fn=None,
        device="cpu",
        dtype=torch.float32,  # autocast disabled on cpu anyway
    )

    history = metrics["train_loss_history"]
    assert len(history) == 10, f"expected 10 logged losses, got {len(history)}"
    assert all(torch.isfinite(torch.tensor(l)).item() for l in history), \
        f"non-finite loss in history: {history}"

    # 2. Soft monotonic-descent check. We allow some noise on a 10-step
    # smoke test: compare the average of the first 3 vs the last 3.
    early = sum(history[:3]) / 3
    late = sum(history[-3:]) / 3
    assert late < early, f"loss did not decrease: early={early:.4f}, late={late:.4f}"
    print(f"PASS  loss descent {early:.4f} -> {late:.4f} over 10 steps")

    # 3. Teacher params unchanged.
    max_drift = 0.0
    for name, p in teacher.named_parameters():
        ref = teacher_before[name]
        d = (p.detach() - ref).abs().max().item()
        max_drift = max(max_drift, d)
    assert max_drift == 0.0, f"teacher drifted by {max_drift:.3e}"
    print(f"PASS  teacher max param drift = {max_drift:.3e}")

    print(f"PASS  swapped indices = {metrics['swapped_indices']}")
    print(f"PASS  final loss = {metrics['final_train_loss']:.4f}")
    print("ALL PASS")


if __name__ == "__main__":
    main()
