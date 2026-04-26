#!/usr/bin/env python3
"""Unit test for per_layer_trainer.

Generates fake CACT chunks (5 files, 200 tokens, 5120 dim each) with a
fixed input -> output mapping (linear + small noise) so the trainer
should observably reduce MSE within 100 steps.

Asserts:
  - training loss at step 100 < training loss at step 0
  - saved safetensors file exists and is reloadable
  - cleans up its scratch dir on success
"""
from __future__ import annotations

import shutil
import struct
import sys
import tempfile
from pathlib import Path

import torch

# Make the phase1 dir importable.
HERE = Path(__file__).resolve().parent
PHASE1 = HERE.parent
sys.path.insert(0, str(PHASE1))

from per_layer_trainer import CACT_MAGIC, CACT_VERSION, HEADER_STRUCT, train  # noqa: E402


def _write_cact_chunk(path: Path, tensor_bf16: torch.Tensor) -> None:
    """Write a CACT chunk holding the given (n_tokens, n_embd) bf16 tensor."""
    assert tensor_bf16.dtype == torch.bfloat16
    assert tensor_bf16.dim() == 2
    n_tokens, n_embd = tensor_bf16.shape
    header = HEADER_STRUCT.pack(CACT_MAGIC, CACT_VERSION, n_tokens, n_embd)
    raw = tensor_bf16.contiguous().view(torch.uint16).cpu().numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(raw)


def main() -> int:
    scratch = Path(tempfile.mkdtemp(prefix="test-trainer-"))
    print(f"scratch dir: {scratch}")

    try:
        torch.manual_seed(0)

        # Use a tiny d_in for speed (the trainer's structure doesn't depend
        # on d_in=5120 — that's a Qwen3.6-27B production setting). The
        # CACT format preserves embedded n_embd, so we just have to keep
        # the test --d-in argument in sync.
        d_in = 64
        n_per_chunk = 200
        n_chunks = 5

        layer = 7  # arbitrary
        act_dir = scratch / "activations"
        in_dir = act_dir / f"layer_{layer}_in"
        out_dir = act_dir / f"layer_{layer}_out"

        # Synthetic mapping: out = tanh(W @ in) + small noise.
        # Linear-with-nonlinearity gives the trainer something learnable
        # but not trivial. Bf16 round-trips for sanity.
        W = torch.randn(d_in, d_in) * (1.0 / d_in ** 0.5)
        for i in range(n_chunks):
            x = torch.randn(n_per_chunk, d_in)
            y = torch.tanh(x @ W.T) + 0.01 * torch.randn(n_per_chunk, d_in)
            _write_cact_chunk(in_dir / f"chunk_{i:04d}.bin", x.to(torch.bfloat16))
            _write_cact_chunk(out_dir / f"chunk_{i:04d}.bin", y.to(torch.bfloat16))

        out_path = scratch / "trained" / f"layer_{layer}.safetensors"

        # Run trainer for 100 steps. Tiny CubeMemoryLayer to keep it fast.
        metrics = train(
            activations_dir=act_dir,
            layer=layer,
            output=out_path,
            steps=100,
            batch_size=16,
            lr=1e-2,         # bumped for short-horizon test
            seed=42,
            d_in=d_in,
            d_codebook=32,
            m=8,
            p=2,
            n_slots=128,
            d_value=32,
            top_k=4,
            val_split=0.1,
            log_every=20,
            val_every=50,
            device="cpu",
        )

        initial = metrics["initial_train_mse"]
        final = metrics["final_train_mse"]
        print(f"initial_train_mse = {initial:.6f}")
        print(f"final_train_mse   = {final:.6f}")

        assert final < initial, (
            f"loss did not decrease: initial={initial:.6f}, final={final:.6f}"
        )
        print(f"PASS  loss descended {initial:.6f} -> {final:.6f} in 100 steps")

        assert out_path.exists(), f"safetensors file missing: {out_path}"
        from safetensors.torch import load_file
        loaded = load_file(str(out_path))
        assert len(loaded) > 0, "loaded state_dict is empty"
        print(f"PASS  safetensors reloadable, {len(loaded)} tensors")

        return 0
    finally:
        # Cleanup scratch dir on every exit path.
        try:
            shutil.rmtree(scratch)
            print(f"cleaned up {scratch}")
        except OSError as e:
            print(f"warning: cleanup failed for {scratch}: {e}")


if __name__ == "__main__":
    sys.exit(main())
