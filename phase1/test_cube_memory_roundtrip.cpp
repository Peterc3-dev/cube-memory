/*
 * test_cube_memory_roundtrip.cpp — standalone C++ acceptance test
 *
 * Loads a GGUF file exported by export_to_gguf.py, replays the full
 * CubeMemoryLayer forward pass in plain C++ (no ggml graph), and
 * compares against the gold output stored in the file.
 *
 * Validates: tensor layout, algorithm parity, GGUF readability.
 *
 * Build:
 *   g++ -O2 -std=c++17 -o test-cube-memory-roundtrip \
 *       test_cube_memory_roundtrip.cpp -lm
 *
 * Run:
 *   ./test-cube-memory-roundtrip cube_memory_roundtrip.gguf
 *
 * Generate the GGUF first:
 *   python3 export_to_gguf.py cube_memory_roundtrip.gguf
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <cassert>

// =====================================================================
// Minimal GGUF parser — just enough for f32 tensors and uint32 KV
// =====================================================================

static const uint32_t GGUF_MAGIC   = 0x46554747; // "GGUF" as little-endian uint32
static const uint32_t GGUF_VERSION = 3;

// GGUF value types we care about
enum gguf_type : uint32_t {
    GGUF_TYPE_UINT8    = 0,
    GGUF_TYPE_INT8     = 1,
    GGUF_TYPE_UINT16   = 2,
    GGUF_TYPE_INT16    = 3,
    GGUF_TYPE_UINT32   = 4,
    GGUF_TYPE_INT32    = 5,
    GGUF_TYPE_FLOAT32  = 6,
    GGUF_TYPE_BOOL     = 7,
    GGUF_TYPE_STRING   = 8,
    GGUF_TYPE_ARRAY    = 9,
    GGUF_TYPE_UINT64   = 10,
    GGUF_TYPE_INT64    = 11,
    GGUF_TYPE_FLOAT64  = 12,
};

// GGML tensor types we care about
enum ggml_type : uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
};

struct gguf_str {
    uint64_t len;
    std::string s;
};

struct gguf_tensor_info {
    std::string name;
    uint32_t    n_dims;
    uint64_t    ne[4]; // dimensions
    uint32_t    type;
    uint64_t    offset; // relative to start of tensor data block
};

class GGUFReader {
public:
    FILE * fp = nullptr;
    uint32_t version = 0;
    uint64_t n_tensors = 0;
    uint64_t n_kv = 0;

    // Parsed metadata
    std::unordered_map<std::string, uint32_t> kv_uint32;
    std::unordered_map<std::string, std::string> kv_string;

    // Parsed tensor info
    std::vector<gguf_tensor_info> tensors;

    // Offset where tensor data blob starts
    uint64_t data_offset = 0;

    bool open(const char * fname) {
        fp = fopen(fname, "rb");
        if (!fp) {
            fprintf(stderr, "ERROR: cannot open '%s'\n", fname);
            return false;
        }
        return true;
    }

    void close() {
        if (fp) { fclose(fp); fp = nullptr; }
    }

    bool parse() {
        // Header
        uint32_t magic;
        fread_exact(&magic, 4);
        if (magic != GGUF_MAGIC) {
            fprintf(stderr, "ERROR: bad magic 0x%08x (expected GGUF)\n", magic);
            return false;
        }
        fread_exact(&version, 4);
        if (version < 2 || version > 3) {
            fprintf(stderr, "ERROR: unsupported GGUF version %u\n", version);
            return false;
        }
        fread_exact(&n_tensors, 8);
        fread_exact(&n_kv, 8);

        // KV pairs
        for (uint64_t i = 0; i < n_kv; i++) {
            if (!read_kv()) return false;
        }

        // Tensor infos
        tensors.resize(n_tensors);
        for (uint64_t i = 0; i < n_tensors; i++) {
            if (!read_tensor_info(tensors[i])) return false;
        }

        // Alignment: tensor data starts at the next 32-byte boundary
        // after all the header/kv/tensor-info data
        uint64_t pos = (uint64_t)ftell(fp);
        uint32_t alignment = 32; // default GGUF alignment
        // Check if alignment is in KV
        if (kv_uint32.count("general.alignment")) {
            alignment = kv_uint32["general.alignment"];
        }
        data_offset = (pos + alignment - 1) / alignment * alignment;

        return true;
    }

    // Load tensor data as f32 vector
    std::vector<float> load_tensor(const std::string & name) {
        for (auto & ti : tensors) {
            if (ti.name == name) {
                if (ti.type != GGML_TYPE_F32) {
                    fprintf(stderr, "ERROR: tensor '%s' is type %u, expected f32\n",
                            name.c_str(), ti.type);
                    return {};
                }
                uint64_t n_elems = 1;
                for (uint32_t d = 0; d < ti.n_dims; d++) {
                    n_elems *= ti.ne[d];
                }
                std::vector<float> data(n_elems);
                fseek(fp, (long)(data_offset + ti.offset), SEEK_SET);
                fread_exact(data.data(), n_elems * sizeof(float));
                return data;
            }
        }
        fprintf(stderr, "ERROR: tensor '%s' not found\n", name.c_str());
        return {};
    }

    // Get tensor info by name
    const gguf_tensor_info * find_tensor(const std::string & name) const {
        for (auto & ti : tensors) {
            if (ti.name == name) return &ti;
        }
        return nullptr;
    }

    uint32_t get_u32(const std::string & key) const {
        auto it = kv_uint32.find(key);
        if (it == kv_uint32.end()) {
            fprintf(stderr, "ERROR: KV key '%s' not found\n", key.c_str());
            exit(1);
        }
        return it->second;
    }

private:
    void fread_exact(void * buf, size_t n) {
        size_t r = fread(buf, 1, n, fp);
        if (r != n) {
            fprintf(stderr, "ERROR: short read (%zu of %zu bytes)\n", r, n);
            exit(1);
        }
    }

    gguf_str read_string() {
        gguf_str s;
        fread_exact(&s.len, 8);
        s.s.resize(s.len);
        if (s.len > 0) {
            fread_exact(s.s.data(), s.len);
        }
        return s;
    }

    void skip_value(uint32_t vtype) {
        switch (vtype) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_BOOL:
                { uint8_t v; fread_exact(&v, 1); break; }
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
                { uint16_t v; fread_exact(&v, 2); break; }
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
                { uint32_t v; fread_exact(&v, 4); break; }
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
                { uint64_t v; fread_exact(&v, 8); break; }
            case GGUF_TYPE_STRING:
                read_string();
                break;
            case GGUF_TYPE_ARRAY: {
                uint32_t atype;
                uint64_t alen;
                fread_exact(&atype, 4);
                fread_exact(&alen, 8);
                for (uint64_t i = 0; i < alen; i++) {
                    skip_value(atype);
                }
                break;
            }
            default:
                fprintf(stderr, "ERROR: unknown GGUF value type %u\n", vtype);
                exit(1);
        }
    }

    bool read_kv() {
        gguf_str key = read_string();
        uint32_t vtype;
        fread_exact(&vtype, 4);

        if (vtype == GGUF_TYPE_UINT32) {
            uint32_t val;
            fread_exact(&val, 4);
            kv_uint32[key.s] = val;
        } else if (vtype == GGUF_TYPE_STRING) {
            gguf_str val = read_string();
            kv_string[key.s] = val.s;
        } else {
            // Skip values we don't need but still consume bytes
            skip_value(vtype);
        }
        return true;
    }

    bool read_tensor_info(gguf_tensor_info & ti) {
        gguf_str name = read_string();
        ti.name = name.s;

        fread_exact(&ti.n_dims, 4);
        if (ti.n_dims > 4) {
            fprintf(stderr, "ERROR: tensor '%s' has %u dims (max 4)\n",
                    ti.name.c_str(), ti.n_dims);
            return false;
        }
        memset(ti.ne, 0, sizeof(ti.ne));
        for (uint32_t d = 0; d < ti.n_dims; d++) {
            fread_exact(&ti.ne[d], 8);
        }
        fread_exact(&ti.type, 4);
        fread_exact(&ti.offset, 8);
        return true;
    }
};

// =====================================================================
// Helper: print first N elements of a float vector
// =====================================================================
static void print_vec(const char * label, const float * v, int n, int show = 4) {
    printf("  %-30s [", label);
    for (int i = 0; i < std::min(n, show); i++) {
        if (i) printf(", ");
        printf("%12.8f", v[i]);
    }
    if (n > show) printf(", ...");
    printf("]  (n=%d)\n", n);
}

// =====================================================================
// Forward pass implementation
// =====================================================================

// Mat-vec: y = A @ x  where A is (rows, cols) in row-major, x is (cols,)
static void matvec(const float * A, const float * x, float * y,
                   int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        double sum = 0.0;
        for (int c = 0; c < cols; c++) {
            sum += (double)A[r * cols + c] * (double)x[c];
        }
        y[r] = (float)sum;
    }
}

// Dot product
static float dot(const float * a, const float * b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += (double)a[i] * (double)b[i];
    }
    return (float)sum;
}

// Softmax over k elements in-place
static void softmax_inplace(float * x, int k) {
    float mx = *std::max_element(x, x + k);
    double sum = 0.0;
    for (int i = 0; i < k; i++) {
        x[i] = expf(x[i] - mx);
        sum += x[i];
    }
    if (sum < 1e-8) sum = 1e-8;
    for (int i = 0; i < k; i++) {
        x[i] = (float)(x[i] / sum);
    }
}

int main(int argc, char ** argv) {
    const char * fname = (argc > 1) ? argv[1] : "cube_memory_roundtrip.gguf";

    printf("=== Cube Memory Round-Trip Acceptance Test ===\n");
    printf("Loading: %s\n\n", fname);

    // ── 1. Load GGUF ────────────────────────────────────────────
    GGUFReader reader;
    if (!reader.open(fname)) return 1;
    if (!reader.parse()) return 1;

    // Read hyperparameters
    uint32_t d_in       = reader.get_u32("cube_memory.d_in");
    uint32_t d_codebook = reader.get_u32("cube_memory.d_codebook");
    uint32_t d_value    = reader.get_u32("cube_memory.d_value");
    uint32_t m          = reader.get_u32("cube_memory.m");
    uint32_t p          = reader.get_u32("cube_memory.p");
    uint32_t n_slots    = reader.get_u32("cube_memory.n_slots");
    uint32_t top_k      = reader.get_u32("cube_memory.top_k");

    uint32_t d_key = 2 * d_codebook; // interleaved re/im dimension

    printf("Hyperparameters:\n");
    printf("  d_in=%u  d_codebook=%u  d_value=%u  m=%u  p=%u  n_slots=%u  top_k=%u\n",
           d_in, d_codebook, d_value, m, p, n_slots, top_k);
    printf("  d_key (2*d_codebook) = %u\n\n", d_key);

    // Load tensors
    auto input     = reader.load_tensor("cube_memory.input");
    auto role_proj = reader.load_tensor("cube_memory.role_proj.weight");
    auto slot_keys = reader.load_tensor("cube_memory.slot_keys");
    auto slot_vals = reader.load_tensor("cube_memory.slot_values");
    auto out_proj  = reader.load_tensor("cube_memory.out_proj.weight");
    auto gold      = reader.load_tensor("cube_memory.gold_output");

    std::vector<std::vector<float>> codebooks(p);
    for (uint32_t ax = 0; ax < p; ax++) {
        char name[64];
        snprintf(name, sizeof(name), "cube_memory.codebook_%u", ax);
        codebooks[ax] = reader.load_tensor(name);
        if (codebooks[ax].size() != (size_t)(m * d_key)) {
            fprintf(stderr, "ERROR: codebook_%u has %zu elems, expected %u\n",
                    ax, codebooks[ax].size(), m * d_key);
            return 1;
        }
    }

    reader.close();

    // Validate tensor sizes
    if (input.size()     != d_in)                { fprintf(stderr, "bad input size\n"); return 1; }
    if (role_proj.size() != (size_t)(p * d_codebook * d_in)) { fprintf(stderr, "bad role_proj size\n"); return 1; }
    if (slot_keys.size() != (size_t)(n_slots * d_key))       { fprintf(stderr, "bad slot_keys size\n"); return 1; }
    if (slot_vals.size() != (size_t)(n_slots * d_value))     { fprintf(stderr, "bad slot_values size\n"); return 1; }
    if (out_proj.size()  != (size_t)(d_in * d_value))        { fprintf(stderr, "bad out_proj size\n"); return 1; }
    if (gold.size()      != d_in)                { fprintf(stderr, "bad gold size\n"); return 1; }

    if (top_k > n_slots) {
        fprintf(stderr, "ERROR: top_k %u > n_slots %u\n", top_k, n_slots);
        return 1;
    }

    printf("Tensor sizes validated OK.\n\n");

    // ── 2. Forward pass ─────────────────────────────────────────

    printf("--- Step 1: Role projection (role_proj @ h) ---\n");
    // role_proj is (p*d_codebook, d_in) in row-major
    // h is (d_in,)
    // result q is (p*d_codebook,) reals
    std::vector<float> q(p * d_codebook);
    matvec(role_proj.data(), input.data(), q.data(), p * d_codebook, d_in);
    print_vec("input h", input.data(), d_in);
    print_vec("q (role_proj @ h)", q.data(), p * d_codebook);
    printf("\n");

    printf("--- Step 2: Per-axis cleanup (phasor + Hermitian argmax) ---\n");
    // For each axis: convert q slice to phasor, find nearest codebook entry
    // Codebook is stored as interleaved (re,im): [re0,im0,re1,im1,...]
    // q slice is d_codebook reals -> phasor is d_key interleaved reals

    // cleaned[ax] holds the winning codebook row (d_key interleaved reals)
    std::vector<std::vector<float>> cleaned(p);
    for (uint32_t ax = 0; ax < p; ax++) {
        cleaned[ax].resize(d_key);

        // q_axis = q[ax*d_codebook : (ax+1)*d_codebook]  (d_codebook reals)
        const float * q_axis = q.data() + ax * d_codebook;

        // Convert to phasor: interleaved [re0,im0,re1,im1,...]
        std::vector<float> q_phasor(d_key);
        for (uint32_t j = 0; j < d_codebook; j++) {
            q_phasor[2 * j]     = cosf(q_axis[j]);
            q_phasor[2 * j + 1] = sinf(q_axis[j]);
        }

        printf("  axis %u: q_axis[:4] = [", ax);
        for (int i = 0; i < std::min((int)d_codebook, 4); i++) {
            if (i) printf(", ");
            printf("%.6f", q_axis[i]);
        }
        printf("]\n");

        printf("  axis %u: q_phasor[:4] = [", ax);
        for (int i = 0; i < std::min((int)d_key, 4); i++) {
            if (i) printf(", ");
            printf("%.6f", q_phasor[i]);
        }
        printf("]\n");

        // Find nearest codebook entry via Hermitian inner product:
        //   sim[r] = Re(q^H . cb_row) / d_codebook
        //          = sum_j(q_re[j]*cb_re[j] + q_im[j]*cb_im[j]) / d_codebook
        // Since both are interleaved, this is just a dot product / d_codebook
        const float * cb = codebooks[ax].data();
        float best_sim = -1e30f;
        int best_idx = 0;
        for (uint32_t r = 0; r < m; r++) {
            float sim = dot(q_phasor.data(), cb + r * d_key, d_key) / d_codebook;
            if (sim > best_sim) {
                best_sim = sim;
                best_idx = r;
            }
        }

        // Copy winning row
        memcpy(cleaned[ax].data(), cb + best_idx * d_key, d_key * sizeof(float));

        printf("  axis %u: best_idx=%d  best_sim=%.8f\n", ax, best_idx, best_sim);
        printf("  axis %u: cleaned[:4] = [", ax);
        for (int i = 0; i < std::min((int)d_key, 4); i++) {
            if (i) printf(", ");
            printf("%.6f", cleaned[ax][i]);
        }
        printf("]\n\n");
    }

    printf("--- Step 3: FHRR bind across axes (elementwise complex multiply) ---\n");
    // addr starts as cleaned[0], then *= cleaned[1], *= cleaned[2], ...
    // Interleaved layout: [re0,im0,re1,im1,...]
    // Complex mul: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    std::vector<float> addr(d_key);
    memcpy(addr.data(), cleaned[0].data(), d_key * sizeof(float));

    for (uint32_t ax = 1; ax < p; ax++) {
        for (uint32_t j = 0; j < d_codebook; j++) {
            float a_re = addr[2 * j];
            float a_im = addr[2 * j + 1];
            float b_re = cleaned[ax][2 * j];
            float b_im = cleaned[ax][2 * j + 1];
            addr[2 * j]     = a_re * b_re - a_im * b_im;
            addr[2 * j + 1] = a_re * b_im + a_im * b_re;
        }
    }
    print_vec("addr after bind", addr.data(), d_key);
    printf("\n");

    printf("--- Step 4: Unitize (normalize each complex element to unit modulus) ---\n");
    for (uint32_t j = 0; j < d_codebook; j++) {
        float re = addr[2 * j];
        float im = addr[2 * j + 1];
        float mag = sqrtf(re * re + im * im);
        if (mag < 1e-8f) mag = 1e-8f;
        addr[2 * j]     = re / mag;
        addr[2 * j + 1] = im / mag;
    }
    print_vec("addr after unitize", addr.data(), d_key);
    printf("\n");

    printf("--- Step 5: Convert to q_real (split re||im for slot key matching) ---\n");
    // PyTorch _addr_to_realq does: torch.cat([addr.real, addr.imag], dim=-1)
    // This means: [re_0, re_1, ..., re_{d-1}, im_0, im_1, ..., im_{d-1}]
    // We have interleaved [re_0, im_0, re_1, im_1, ...] — must de-interleave
    std::vector<float> q_real(d_key);
    for (uint32_t j = 0; j < d_codebook; j++) {
        q_real[j]              = addr[2 * j];     // re
        q_real[d_codebook + j] = addr[2 * j + 1]; // im
    }
    print_vec("q_real (split re||im)", q_real.data(), d_key);
    printf("\n");

    printf("--- Step 6: Retrieve (top-k soft lookup into slot store) ---\n");
    // sims[s] = dot(q_real, slot_keys[s]) for each slot s
    std::vector<float> sims(n_slots);
    for (uint32_t s = 0; s < n_slots; s++) {
        sims[s] = dot(q_real.data(), slot_keys.data() + s * d_key, d_key);
    }

    // Find top_k indices
    std::vector<int> indices(n_slots);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&](int a, int b) { return sims[a] > sims[b]; });

    printf("  top_%u indices: [", top_k);
    for (uint32_t i = 0; i < top_k; i++) {
        if (i) printf(", ");
        printf("%d (sim=%.6f)", indices[i], sims[indices[i]]);
    }
    printf("]\n");

    // Softmax over top_k similarities
    std::vector<float> topk_sims(top_k);
    for (uint32_t i = 0; i < top_k; i++) {
        topk_sims[i] = sims[indices[i]];
    }
    softmax_inplace(topk_sims.data(), top_k);

    printf("  softmax weights: [");
    for (uint32_t i = 0; i < top_k; i++) {
        if (i) printf(", ");
        printf("%.6f", topk_sims[i]);
    }
    printf("]\n");

    // Weighted sum of slot values
    std::vector<float> gathered(d_value, 0.0f);
    for (uint32_t i = 0; i < top_k; i++) {
        int idx = indices[i];
        for (uint32_t v = 0; v < d_value; v++) {
            gathered[v] += topk_sims[i] * slot_vals[idx * d_value + v];
        }
    }
    print_vec("gathered (weighted slot values)", gathered.data(), d_value);
    printf("\n");

    printf("--- Step 7: Output projection (out_proj @ gathered) ---\n");
    // out_proj is (d_in, d_value) in row-major
    std::vector<float> output(d_in);
    matvec(out_proj.data(), gathered.data(), output.data(), d_in, d_value);
    print_vec("output (delta)", output.data(), d_in);
    printf("\n");

    // ── 3. Compare against gold ─────────────────────────────────

    printf("=== Comparison ===\n");
    print_vec("gold output", gold.data(), d_in);
    print_vec("computed output", output.data(), d_in);

    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int   max_abs_idx = 0;
    int   max_rel_idx = 0;
    double sum_sq_err = 0.0;
    double sum_sq_gold = 0.0;

    for (uint32_t i = 0; i < d_in; i++) {
        float err = fabsf(output[i] - gold[i]);
        float rel = (fabsf(gold[i]) > 1e-8f) ? err / fabsf(gold[i]) : 0.0f;
        sum_sq_err  += (double)err * err;
        sum_sq_gold += (double)gold[i] * gold[i];
        if (err > max_abs_err) { max_abs_err = err; max_abs_idx = i; }
        if (rel > max_rel_err) { max_rel_err = rel; max_rel_idx = i; }
    }

    float rmse = (float)sqrt(sum_sq_err / d_in);
    float gold_norm = (float)sqrt(sum_sq_gold);

    printf("\n");
    printf("  max absolute error: %.10e  (index %d)\n", max_abs_err, max_abs_idx);
    printf("  max relative error: %.10e  (index %d)\n", max_rel_err, max_rel_idx);
    printf("  RMSE:               %.10e\n", rmse);
    printf("  gold L2 norm:       %.10e\n", gold_norm);
    printf("\n");

    // Tolerance: fp32 accumulation across small dims should be very tight.
    // We allow 1e-5 absolute error (conservatively).
    const float tol = 1e-5f;
    bool pass = (max_abs_err < tol);

    if (pass) {
        printf("PASS  (max_abs_err=%.2e < tol=%.2e)\n", max_abs_err, tol);
    } else {
        printf("FAIL  (max_abs_err=%.2e >= tol=%.2e)\n", max_abs_err, tol);
        printf("\n  Element-wise comparison (first 8):\n");
        for (uint32_t i = 0; i < std::min(d_in, 8u); i++) {
            printf("    [%2u] gold=%12.8f  got=%12.8f  err=%12.8e\n",
                   i, gold[i], output[i], (double)(output[i] - gold[i]));
        }
    }

    return pass ? 0 : 1;
}
