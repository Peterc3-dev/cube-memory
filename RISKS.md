# Cube Memory — known risks ledger

Captured from the Phase 1→2 transition adversarial sweep
(2026-04-26). Each item is something the random-input parity tests
do not exercise but production will. Severity tags reflect "how
likely is this to bite real users."

## High — must guard before Phase 2 deployment

- **`top_k > n_slots` is silently clamped on GPU but unrestricted in
  PyTorch.**  Layer constructor should validate
  `top_k <= n_slots and top_k <= MAX_TOP_K`. Add to
  `phase1/cube_memory_layer.py::__init__`.

- **`top_k = 0` crashes softmax** in the PyTorch retrieve and produces
  uniform 1/k weights with garbage indices on GPU. Constructor should
  refuse `top_k < 1`.

## Medium — test before betting on it

- **Cleanup tie-break diverges between CPU `argmax` (first-wins) and
  GPU `>` comparison (last-wins).** Random unit phasors don't tie at
  fp32 precision; deterministic codebooks with duplicate entries do.
  Either standardize the comparison (`>=` vs `>`) on both sides or
  document that the layer is not bit-deterministic across backends.

- **STE gradient through cleanup vanishes at exact codebook
  boundaries.** Gradient on `role_proj` is zero whenever the query
  exactly aligns with a snapped phasor. Practically rare with random
  init, but the test set never explicitly covers it. Add a unit test
  that initializes a query at a codebook entry and asserts the
  forward+backward path still moves the loss.

- **Distillation test fixture's FakeLoader resets seed every epoch,
  so all 10 steps see identical batches.** Hides shuffling bugs that
  would matter at real distillation scale. Fix by using a per-epoch
  RNG reseed or asserting batch variety.

- **GGUF buffer layout for `slot_keys` and `slot_values` is
  unspecified.** PyTorch is row-major `(n_slots, 2*d_codebook)`; the
  ggml-vulkan backend's tensor stride conventions need to match
  exactly. **Test before claiming Phase 2 is integrated.** If a
  silent transpose creeps in, the dot products compute on the wrong
  axis and retrieve returns plausible-looking garbage.

- **Unitize on `Vec2::ZERO` returns `Vec2::ZERO`** (the eps floor
  prevents division-by-zero but does not produce a unit vector).
  Algebraically wrong but doesn't crash. Mitigate by making the
  cleanup path map `Vec2::ZERO` to a canonical phasor `(1, 0)`.

## Low — flag if scaling up

- **Cleanup with d ≤ 4** has insufficient noise margin to be
  reliable. Document a `d >= 64` floor in the layer.
- **`n_slots = 1` with `top_k > 1`** picks the same slot k times and
  softmax-weights it equally. Mathematically correct, semantically
  pointless. Validate `n_slots >= top_k` at init.
- **Naive dot product in the GPU retrieve loop** could accumulate
  fp32 error at very large `d_key`. Probably fine to d_key ≈ 4096;
  re-check before going beyond.

## Phase 2 implications

Three of the items above gate Phase 2 acceptance:
1. The `top_k`/`n_slots` invariants must be enforced in the GGUF
   loader (refuse to load a model that violates them).
2. The buffer-layout assumption must be verified by a round-trip
   parity test: PyTorch forward → export to GGUF → load in ggml →
   ggml forward → compare to PyTorch output. Same input, same
   weights, same output to fp32 tolerance.
3. The shader push-constant order must match the GGUF tensor order
   exactly. Today both are written by us; in a future where ggml
   reorders tensors during graph optimization, that assumption breaks.
   Add a runtime assert.
