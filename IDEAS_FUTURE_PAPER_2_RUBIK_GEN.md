# Future Paper 2 — Rubik's-Cube Permutation-Group Generative Adaptation

**Status**: stashed 2026-04-26. Pick up after Paper 1 (cube-memory as
FFN replacement on Qwen3.6-27B) ships and the VSA primitives are
battle-tested at production scale.

**Why stashed, not abandoned**: the core insight is genuinely novel
and the math is sound, but the proposal has three structural gaps
that handwave away ~80% of the work. Better to surface those now,
ship Paper 1 first, and return with eyes open.

---

## The thesis (in one paragraph)

Treat image/video generation as **permutation of a small basis set
in hypervector space**, analogous to a Rubik's cube. Faces are bound
bundles of tile hypervectors. Tiles are latent patches. Rotations are
permutation operators on the bundle (cheap O(D) unitary ops).
Snapshots are `cube_memory_retrieve` against current state, decoded
to pixels. Video is a trajectory through the rotation group;
consecutive frames share most of the bound state, so temporal
coherence is — claimed — a structural property, not a learned one.

## Hardware fit (real)

- 5×5 cube basis at D=8192 = ~5 MB total state. Fits in cache on
  Strix Point. Diffusion checkpoints are 1000× bigger.
- Permutation ops are matmul-free, near-free on iGPU.
- Hierarchical depth scales expressive capacity multiplicatively
  while keeping per-layer cost flat.
- Maps cleanly onto the Arbiter OS NPU sentinel + iGPU decode split
  (tri-processor thesis from the consortium plan).

## What's right (the publishable kernel)

1. **VSA permutation operators are O(D)** — verified primitive, not a
   claim. Already-built bind/unbind kernels apply directly.
2. **Hierarchical bundling is a known capacity-extender** — Plate's
   bound `D / (k log N)` makes a 5×5 face (25 items) right at the
   edge of flat capacity at D=8192. Tree-structured bundling (5
   row-bundles of 5 each) keeps every level at safe margin.
3. **"Temporal coherence as structural property of permutation-group-
   indexed state" is a fresh framing** — nobody has published this
   for video gen. Diffusion video burns compute fighting the
   incoherence problem; AR video burns sequence length. A third axis
   exists.
4. **Permutation-Group-Indexed Compositional Generation** sits in a
   genuinely empty cell of the design space:
   - Diffusion = denoising in latent/pixel space
   - Autoregressive = next-token in token space
   - PGICG = permutation in algebraic-bundle space

---

## What's hand-waved (the proposal punts on these)

### Gap 1 — Tile basis acquisition is the WHOLE problem

The proposal says: *"You need to learn the 54 base latents from real
data. That's a training run, not free."* In reality this is the
entire latent-representation-learning problem — the same one
VAE/VQ-VAE/RQ-VAE spend their core engineering on.

Where do 25 tiles per face × 6 faces = 150 latents come from such
that they meaningfully decompose images? You need an encoder that:
- Produces a dictionary of tiles with semantic locality
- Has tile-positional prior consistent with the cube's 6×N×N grid
- Is invertible enough that the decoder can reach acceptable PSNR

This is at least 80% of the work. The proposal treats it as a setup
step.

### Gap 2 — Conditioning between layers is non-trivial

The hierarchical stack proposal:
```
Layer 1 (3×3) → H1
Layer 2 (5×5) conditioned on H1 → H2
Layer 3 (7×7) conditioned on H2 → H3
```

The conditioning is described as *"H_prev → tile selection logits"*
in one bullet. This is the hardest part of any hierarchical model
— what hierarchical VQ-VAE, DALL-E 1's discrete VAE, VAR, and
MaskGIT all spend their parameters on. Not trivially solvable by
structural decree.

### Gap 3 — Temporal coherence is NOT free

*"Permutation-neighbors in cube space = visually-neighboring frames"*
is asserted as a structural property. False. It only holds if the
tile bank is trained such that permutation-locality implies
visual-locality. That's a *training-objective constraint*, not an
architectural inheritance.

If the tile bank is trained as a generic VAE codebook with no
permutation-locality bias, two adjacent permutations can decode to
wildly different images. The claim of "free temporal coherence" needs
a loss term and an empirical validation, not just an assertion.

### Gap 4 — Demo bar is too low to validate the thesis

- **PSNR > 20 dB on MNIST** is reachable by a 2-layer MLP
  autoencoder. Doesn't prove cube-memory is doing useful work.
- **MNIST → CIFAR-32 transfer** is the "we generalize" demo for any
  new arch. Proves nothing about the bandwidth/quality tradeoff vs
  diffusion at meaningful resolutions.
- **Real validation** needs ≥256×256 with FID and CLIP-score numbers
  vs a tiny diffusion baseline (e.g. SD-Turbo, Tiny-DiT). The
  proposal stops at MNIST.

### Gap 5 — Storage comparison is asymmetric

*"5 MB cube state vs 4 GB diffusion checkpoint"* sounds great until
you remember the 4 GB encodes photorealism that 25 latent tiles per
face cannot. Storage isn't the metric users notice; output quality
is. The right comparison is bandwidth-per-frame at iso-quality, and
that requires actually getting to iso-quality first.

### Gap 6 — VSA's track record is in the wrong domain

VSA bind has 30 years of validation at **associative recall**
(content-addressable retrieval, symbolic reasoning, working memory).
Effectively zero track record at **photorealistic synthesis**.
Recall ≠ generation. This doesn't kill the idea — generative VSA is
genuinely uncharted — but it does mean the existence proof matters
much more than the marketing.

---

## What it would actually take to do this right

1. **Pre-train a perm-locality-aware VQ-VAE** on the target image
   distribution. ~2 weeks engineering + 1-3 days compute on
   raz-gpd4 if the model stays small.
2. **Define the conditioning operator** (cross-attention from H_prev
   to current layer's tile-selection logits, probably). Pick from
   the hierarchical-VAE literature; don't reinvent.
3. **Train end-to-end** with both reconstruction loss AND a
   permutation-locality loss (tiles whose permutation-distance is
   small should have small visual distance after decoding).
4. **Bench at ≥256×256**. FID vs SD-Turbo. CLIP-score vs same-budget
   tiny diffusion.
5. **Demo the temporal-coherence claim** with a real user test:
   permutation sequence → decoded video, vs diffusion-video baseline,
   vs autoregressive-video baseline, all at same compute budget.

Total realistic budget: **3-6 months of focused work** to a
publishable result. Not 2 weeks to a MNIST demo.

---

## Why we're not pivoting the active loop into this

The current loop has clear measurable progress:
- Cube-memory tiled shaders shipped (commits ccc3b9f + a6af0ff)
- Qwen3.6-27B on disk
- llama-dump-activations tool built and smoke-tested
- per_layer_trainer + load_layer round-trip working
- NPU operational + ollama-coprocessor MCP extended with proofread tool
- tok/s baseline sweep in flight

Pivoting to image-gen now means:
- Forking infrastructure (image dataloader, FID eval, decoder, VAE)
- Losing the immediate measurable target (tok/s on Qwen3.6-27B)
- Probably never converging back to the Qwen3.6 work

Paper 1 (FFN replacement) earns its keep on bandwidth-bound LLM TG.
After it ships and the primitives are battle-tested at production
scale, **then** the image-gen adaptation has a stronger foundation —
including real-world capacity-curve data for the VSA bundling +
real-world tooling for the dispatch path.

---

## The original session prompt (verbatim, for future restart)

When picking this back up, drop the prompt below into a fresh Claude
Code session. Note the gaps above before executing — the wedge order
is reasonable but Wedge 3's training loop will hit Gap 1 (tile basis
acquisition) almost immediately, so prepend a **Wedge 0: train a
perm-locality-aware tile encoder** before Wedge 3 makes sense.

---

```markdown
# Session: Cube Memory — Permutation Kernel & Hierarchical Generative Stack

## Context

I'm extending the Cube Memory project (rust-gpu/wgpu/ggml,
structured-VSA memory layers for GPU inference) with a new
application domain: **bandwidth-efficient compositional image/video
generation** via permutation-group-indexed VSA retrieval. This is a
research wedge adjacent to the existing FFN-replacement work, not a
replacement for it.

### Existing state (don't rebuild)

- 6 rust-gpu kernels green on iGPU with byte-level CPU parity:
  bind, unbind, unitize, superpose, cube_memory_cleanup,
  cube_memory_retrieve
- Tiled multi-WG cube_memory_{cleanup,retrieve}_{score,finalize}
  shipped in mainline cube-memory-op branch with subgroup_add
  reduction (commits ccc3b9f + a6af0ff). 3.5x cleanup / 1.9x
  retrieve vs single-WG v0 baseline at the bench's largest shapes.
- ggml-vulkan two-pass dispatch wired with prealloc_x scratch +
  sync_buffers between passes. supports_op gates m/n_slots ≤
  device->maxComputeWorkGroupCount[0], top_k ∈ [1, 8], d_key ≤ 4096.
- llama-dump-activations CLI tool built and smoke-tested
- per_layer_trainer + load_layer + safetensors round-trip working
- Hardware: AMD Ryzen AI HX 370, Radeon 890M iGPU (gfx1150),
  16 GB system RAM + 16 GB VRAM (BIOS UMA), CachyOS

### The new thesis (mental model)

Treat image/video generation as **permutation of a small basis set
in hypervector space**, analogous to a Rubik's cube:
- Face = bound bundle of tile hypervectors
- Tile = latent patch encoded as a hypervector
- Rotation = permutation operator on the bundle (cheap O(D))
- Snapshot = cube_memory_retrieve against current state, decoded
- Video = trajectory through the rotation group

5×5 cube chosen as starting basis: 25 tiles/face, 150 cells, past
toy threshold but below VSA capacity ceiling. D=8192 hypervectors,
FP32.

### Why this matters on this hardware

- 890M is bandwidth-bound. 5×5 cube basis at D=8192 ≈ 5 MB total
  state. Fits in cache.
- Permutation ops are matmul-free → near-free on iGPU
- Hierarchical depth scales expressive capacity multiplicatively
  while keeping per-layer cost flat
- Maps cleanly onto the Arbiter OS NPU sentinel + iGPU decode split

---

## CRITICAL — read the gap analysis at the top of this file FIRST.

The wedge order below is reasonable but Wedge 3's training loop will
hit Gap 1 (tile basis acquisition) almost immediately. Prepend a
WEDGE 0 (perm-locality-aware tile encoder pre-training) before
Wedge 3 makes sense. Don't proceed without addressing Gaps 1-3.

---

## Build order — three wedges (after Wedge 0)

### WEDGE 0 (NEW — required, was missing from original): tile encoder

Pre-train a small VQ-VAE or RQ-VAE on the target image distribution
with a permutation-locality auxiliary loss. Codebook size = 150
(matches 6 faces × 25 tiles/face for 5×5 cube). Trainable on
raz-gpd4 in ~1-3 days at 64×64 resolution.

### WEDGE 1: Permutation kernel

Create shaders/src/permute.rs matching the existing rust-gpu shader
pattern (see bind.rs, superpose.rs for style/structure):

```rust
#[spirv(compute(threads(64)))]
pub fn permute(
    #[spirv(global_invocation_id)] gid: UVec3,
    #[spirv(storage_buffer, descriptor_set=0, binding=0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set=0, binding=1)] perm: &[u32],
    #[spirv(storage_buffer, descriptor_set=0, binding=2)] output: &mut [f32],
) {
    let i = gid.x as usize;
    if i >= output.len() { return; }
    output[i] = input[perm[i] as usize];
}
```

Tasks:
1. Add permute.rs to the shader workspace, wire into the existing
   wgpu host harness
2. Generate all 24 permutation index tables for a 5×5 cube
3. Add CPU reference using np.take-equivalent
4. Add 24 parity tests — assert byte-level match between iGPU output
   and CPU reference on random D=8192 input vectors

Wedge 1a: Permutation index generator
Write tools/gen_perms.py that emits the 24 index tables for a 5×5
cube. Verify: applying CW four times = identity; CW + CCW = identity.

### WEDGE 2: Hierarchical bundle composer

Create cube_memory/src/hierarchical.rs. Implement tree-structured
VSA bundling — 5 items per superposition level vs 25 in flat.

```rust
pub fn bundle_face(
    tiles: &[HV; 25],
    row_keys: &[HV; 5],
    col_keys: &[HV; 5],
) -> HV {
    let mut row_bundles = [HV::zero(); 5];
    for r in 0..5 {
        let mut row = HV::zero();
        for c in 0..5 {
            let pos_key = bind(&row_keys[r], &col_keys[c]);
            row = superpose(&row, &bind(&tiles[r*5 + c], &pos_key));
        }
        row_bundles[r] = unitize(&row);
    }
    let mut face = HV::zero();
    for r in 0..5 { face = superpose(&face, &row_bundles[r]); }
    unitize(&face)
}
```

Done when: 25/25 cleanup hits on flat AND post-permutation retrieval,
cleanup margin logged and ≥ 0.3.

### WEDGE 3: Three-layer hierarchical stack on MNIST

Layer 1: 6 faces × 9 tiles  (3×3) → coarse state H1 (D=8192)
Layer 2: 6 faces × 25 tiles (5×5) conditioned on H1 → H2
Layer 3: 6 faces × 49 tiles (7×7) conditioned on H2 → H3
Decoder: small ConvT or 2-layer MLP head: H3 → 28×28 pixels

Done when: hold-out PSNR > 20 dB, recognizable digits, training
runs end-to-end on the 890M iGPU without OOM.

NOTE per gap analysis: PSNR > 20 dB on MNIST is reachable by a
2-layer MLP autoencoder — necessary but NOT sufficient validation.
Add a permutation-locality test: decoded frames from adjacent
permutations should have visual-distance below a threshold.

---

## Validation milestones (track in MILESTONES.md)

1. Permute kernel green
2. 5×5 face round-trips 25/25 with margin ≥ 0.3
3. 3-layer stack reconstructs MNIST, PSNR > 20 dB
4. **NEW per gap analysis: adjacent-permutation visual-distance < threshold**
5. MNIST → CIFAR-32 transfer reconstructs
6. **REVISED per gap analysis: 256×256 FID vs SD-Turbo at same bandwidth**
7. Permutation sequence → smooth image trajectory (the killer demo)

---

## Constraints & guardrails

- Don't break existing tests (all shader + ggml runtime tests stay green)
- Use existing primitives (no new bind/unbind/cleanup variants)
- Fail loud on capacity (cleanup margin < 0.2 = halt)
- No premature optimization (decoder stays MLP until plateau)
- FP32 throughout
```

---

*This file's job is to be a complete pickup point. When the moment
comes, the original Claude.ai mobile session prompt + the gap
analysis + the hardware state above should be enough to restart
without re-deriving anything.*
