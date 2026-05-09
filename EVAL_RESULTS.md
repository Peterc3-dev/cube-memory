# Cube Memory — BFCL Evaluation Results

## Configuration
- Model: Qwen3-8B-Q4_K_M (4.7 GB GGUF, Q4_K quantization)
- Server: llama-server (custom Vulkan build, ~/bin/llama-server)
- Backend: Vulkan (AMD Radeon 890M, gfx1150)
- Context: 4096 tokens
- System prompt: `/no_think` to suppress verbose reasoning
- Temperature: 0.1

## Pilot Run (50 cases per category)

| Category | Score   | Target | Status |
|----------|---------|--------|--------|
| Simple   | 92.0%   | 80%    | PASS   |
| Multiple | 94.0%   | 60%    | PASS   |

### Simple Failures (4/50)
- 3x `x^2` vs `x**2` notation mismatch (correct answer, formatting difference)
- 1x `operating_hours: 11` vs `23` (12h vs 24h format)

### Multiple Failures (3/50)
- 1x wrong function selected (database.create_backup instead of database.modify_columns)
- 1x `3x+2` vs `3x + 2` whitespace mismatch
- 1x `3x^2` vs `3x**2` notation mismatch

## Full Run (400 + 200 cases)

| Category | Score   | Target | Status | Margin |
|----------|---------|--------|--------|--------|
| Simple   | 95.8%   | 80%    | PASS   | +15.8  |
| Multiple | 93.5%   | 60%    | PASS   | +33.5  |

### Simple Failures (17/400)
- 3x `x^2` vs `x**2` notation (correct answer, Python vs math notation)
- 3x semantic near-misses (e.g., `human cell` vs `human`, `banana` vs `bananas`)
- 3x value interpretation (e.g., 12h vs 24h time, `renewable` vs `solar`)
- 2x complex nested structures (array-of-objects conditions)
- 6x other minor mismatches

### Multiple Failures (13/200)
- 2x wrong function chosen (database.create_backup, random_forest_regression)
- 2x `x^2` vs `x**2` notation
- 4x parameter value mismatches (e.g., `quarterly` vs `annually`, extra instruments)
- 1x missing required param (n_rolls)
- 1x no tool call generated
- 3x other minor mismatches

## Performance
- Avg time per case: 3.1s simple, 3.8s multiple (with /no_think)
- Total eval time: 1241s simple + 755s multiple = ~33 minutes
- Token generation: ~18 t/s on Vulkan
- Prompt processing: ~238 t/s on Vulkan
- Zero server errors in full run

## Key Design Decisions
1. `/no_think` system prompt eliminates reasoning overhead (155 -> 45 tokens per call)
2. BFCL type conversion: dict->object, float->number, tuple->array, any->string
3. Nested ground-truth dict matching for complex parameter values
