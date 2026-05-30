# Cube Memory

Research code investigating **Vector Symbolic Architecture (VSA) memory layers as a
replacement for transformer feed-forward (FFN) blocks**. The idea: instead of looking
up values by learned nearest-neighbor similarity, retrieve them by *orientation* —
algebraic unbind against a structured FHRR codebook ("rotate to face θ, read the
snapshot"). The goal was a matmul-free, bandwidth-cheap stand-in for the dense FFN
that dominates per-token weight reads.

## Status

**Negative result — the core hypothesis does not work.** This repo is the experimental
record behind a paper titled *"Two Negative Results for Vector Symbolic Architectures:
FFN Replacement and Compositional Image Generation."* The FFN-replacement direction
(Case Study 1) fails because of a **rank bottleneck**: a VSA cleanup→bind→retrieve
pipeline has effective rank bounded by top-k (typically ~4), while real FFN mappings are
~89% linear with effective rank >2048. A 164K-parameter rank-16 linear projection
captures more variance than a 35M-parameter VSA memory layer, and scaling top-k does not
close the gap. The companion direction (Rubik Gen — compositional image generation via
token binding, `rubik-gen/`) fails for separate reasons documented in
`IDEAS_FUTURE_PAPER_2_RUBIK_GEN.md` and folds in as Case Study 2.

The experiments are complete and the figures are generated; the work is kept here as a
documented negative result and portfolio artifact rather than a usable library. Do not
expect a shippable FFN replacement here.

## What's in the repo

- `SPEC.md`, `PAPER_OUTLINE.md`, `LOCAL_DISTILL_PLAN.md`, `RISKS.md` — design, plan,
  and the bandwidth motivation.
- `phase0/` — FHRR primitives (`fhrr.py`) and a recall sanity test.
- `phase1/` — the bulk of the work: `cube_memory_layer*.py` (the VSA memory layer, v1–v3),
  SVD codebook extraction, FFN-swap harness, per-layer training, reviewer ablation
  experiments, and `tests/`.
- `rubik-gen/` — the compositional image-generation experiments (Case Study 2) with
  results JSON and figures.
- `shaders/` — Rust-GPU (rust-gpu → SPIR-V) compute kernels for the layer, with a
  CPU parity test. Intended for a Phase 2 Vulkan integration that was not reached.
- `paper/` — LaTeX source, bibliography, figures, and a compiled `main.pdf`.
- `reviewer_results/`, `*_STATUS.md`, `EVAL_RESULTS.md`, `V3_ANALYSIS.md` — measured
  numbers, status logs, and analysis.

Most analysis numbers and design rationale live in the Markdown status files; start with
`PAPER_OUTLINE.md` for the overall story and `phase1/PLAN.md` for the experiment plan.

## Running the experiments

The experiments are Python scripts plus a Rust shader crate; there is no single entry
point or packaged module. Each script is meant to be run on its own.

### Python (phase0 / phase1 / rubik-gen)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy   # plus transformers/datasets for the scripts that pull a model
```

Then run an individual script, e.g.:

```bash
python phase0/recall_test.py
python phase1/reviewer_exp1_svd_spectrum.py
```

Scripts that load a model (e.g. activation extraction, distillation) expect a local Qwen3
GGUF / HF checkpoint and a llama.cpp Vulkan build; paths are set inside the scripts and
will need editing for your environment.

### Rust shaders

```bash
cd shaders

# CPU reference unit tests — pure Rust, run on the stable toolchain,
# no GPU or prebuilt SPIR-V needed. These pin the FHRR/Cube-Memory
# reference algebra that the GPU parity tests compare against.
cargo test -p cube-memory-host --lib

# Full GPU/CPU parity tests — require a Vulkan adapter AND the shader
# binary built first via the rust-gpu nightly toolchain:
cargo run  -p cube-memory-shader-builder --release
cargo test -p cube-memory-host --release   # runs tests/parity.rs
```

See `shaders/README.md` for the rust-gpu toolchain details (the pinned
nightly and components needed to build the SPIR-V).

## Limitations

- This is research/experiment code, not a library — no stable API and no packaging.
  CI (`.github/workflows/ci.yml`) covers only the deterministic, hardware-free
  surface: the Rust host crate's CPU reference path (build + clippy + unit tests)
  and a Python syntax check / ruff lint. The rust-gpu shader build and the GPU
  parity tests are out of CI scope (nightly toolchain + Vulkan adapter required).
- Hardware-specific: numbers were measured on an AMD Radeon 890M (gfx1150) Vulkan build of
  llama.cpp; the bandwidth and t/s figures are local measurements, not general benchmarks.
- The headline conclusion is negative; the layer does not match a linear baseline.

## License

MIT — see [LICENSE](LICENSE).
