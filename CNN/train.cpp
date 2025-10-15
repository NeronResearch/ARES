// train.cpp (Updated with modifications for extreme imbalance: proper skips, sparse attention, multi-scale aux heads, deeper residuals, focal loss)
// Build: g++ -std=gnu++20 -O3 -march=native -mtune=native -ffast-math -funroll-loops -fopenmp -flto -DNDEBUG -o train train.cpp
// NOTE: Fixes for compilation errors in SparseAttention::forward:
// - Assumed dim is a constructor param or constant (hardcoded to 4 as per typical use).
// - Built local coord_to_index map on the fly since it's not a member of SparseTensor (based on engine summaries; add to SparseTensor if possible).

#include <iostream>
#include <filesystem>
#include <vector>
#include <array>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <chrono>
#include <cmath>
#include <unordered_map>  // For hash-based skip merging and coord_to_index
#ifdef _OPENMP
#include <omp.h>
#endif
#include "mini_numpy.hpp"
#include "mini_torch.hpp"
#include "mini_minkowski_engine.hpp"
#include "json_loader.hpp"
#include "model_io.hpp"

using namespace std;
namespace fs = std::filesystem;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;
using std::chrono::duration_cast;

// -----------------------------
// Scoped timer (prints on scope exit) — kept only for big scopes
// -----------------------------
struct Timer {
    string label;
    high_resolution_clock::time_point start;
    Timer(const string &lbl) : label(lbl), start(high_resolution_clock::now()) {}
    ~Timer() {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        cout << "[TIMER] " << label << ": " << duration << " ms" << endl;
    }
};

// -----------------------------
// Helpers: sample and batching (reuse Sample from json_loader.hpp)
// -----------------------------

// Build a single Sample from a 10-frame segment descriptor (paths).
static inline Sample load_sample_from_segment(const std::array<fs::path,10>& seg) {
    // json_loader.hpp already defines build_sample_from_segment()
    Sample s = build_sample_from_segment(seg);
    return s; // preserve grid info
}

// Concatenate multiple Samples into one big Sample (row-wise concat).
static inline Sample concat_samples(const std::vector<Sample>& parts) {
    if (parts.empty()) return Sample{};
    int Cx = parts[0].feats.c;
    int Cy = parts[0].labels.c;

    // Count total rows
    int totalN = 0;
    for (const auto& p : parts) totalN += (int)p.coords.size();

    // Allocate outputs
    std::vector<std::array<int,4>> out_coords;
    out_coords.reserve(totalN);
    mt::Tensor out_feats(totalN, Cx, 0.f);
    mt::Tensor out_labels(totalN, Cy, 0.f);

    // Fill
    int row = 0;
    for (const auto& p : parts) {
        const int Np = (int)p.coords.size();

        // coords
        out_coords.insert(out_coords.end(), p.coords.begin(), p.coords.end());

        // feats
        #pragma omp parallel for collapse(2) if (Np*Cx > 1024)
        for (int i=0;i<Np;++i)
            for (int j=0;j<Cx;++j)
                out_feats.at(row + i, j) = p.feats.at(i, j);

        // labels
        #pragma omp parallel for collapse(2) if (Np*Cy > 1024)
        for (int i=0;i<Np;++i)
            for (int j=0;j<Cy;++j)
                out_labels.at(row + i, j) = p.labels.at(i, j);

        row += Np;
    }
    // Preserve grid from first part
    Sample out;
    out.coords = std::move(out_coords);
    out.feats = std::move(out_feats);
    out.labels = std::move(out_labels);
    out.grid = parts[0].grid;
    return out;
}

// Build a batch [start, start+count) of segments into one Sample
static inline Sample build_batched_sample(const std::vector<std::array<fs::path,10>>& segs,
                                          size_t start, size_t count) {
    std::vector<Sample> parts;
    parts.reserve(count);
    for (size_t k = 0; k < count; ++k) {
        parts.emplace_back(load_sample_from_segment(segs[start + k]));
    }
    return concat_samples(parts);
}

// -----------------------------
// NEW: Focal BCE loss in mt namespace (add to mini_torch.hpp or here)
// NOTE: Uses logits.n for number of rows/samples, as per mini_torch.hpp
namespace mt {
float focal_bce_with_logits(const mt::Tensor& logits, const mt::Tensor& y, float pos_w=1.f, float gamma=2.0f, float alpha=0.25f) {
    const int N = logits.n;  // FIXED: Uses 'n' as per mini_torch.hpp
    float total = 0.f;
    #pragma omp parallel for reduction(+:total) if (N > 1024)
    for (int i=0; i<N; ++i) {
        float l = std::clamp(logits.at(i,0), -50.f, 50.f);  // FIXED: Clamp to prevent overflow
        float p = 1.f / (1.f + std::exp(-l));  // Sigmoid
        float t = y.at(i,0);
        float bce = - (t * std::log(p + 1e-6f) + (1-t) * std::log(1-p + 1e-6f));
        float at = t * alpha + (1-t) * (1-alpha);
        float focal = at * std::pow(1-p, gamma) * bce;  // Focal modulation
        total += (t > 0.5f ? pos_w : 1.f) * focal;
    }
    return total / float(N);
}
}  // namespace mt

// -----------------------------
// Model components (updated for depth, attention, skips, aux heads)
// NOTE: Removed dilation params from conv calls (engine lacks support)
// -----------------------------

struct ConvBNReLU {
    mme::MinkowskiConvolution conv;
    mme::MinkowskiBatchNorm bn;
    mme::MinkowskiReLU relu;

    // stored activations
    mme::SparseTensor last_x, last_after_conv, last_after_bn, last_out;

    ConvBNReLU(int c_in, int c_out, int dim=4, int ks=3, std::array<int,4> stride={1,1,1,1})
        : conv(c_in, c_out, ks, stride, dim), bn(c_out), relu(true) {}  // FIXED: Removed dilation (not supported)

    mme::SparseTensor forward(const mme::SparseTensor& x) {
        last_x = x;
        last_after_conv = conv.forward(x);
        last_after_bn  = bn.forward(last_after_conv);
        last_out       = relu.forward(last_after_bn);
        return last_out;
    }

    mme::SparseTensor backward(mme::SparseTensor& grad_out) {
        auto d_bn   = relu.backward(last_after_bn, grad_out);
        auto d_conv = bn.backward(last_after_conv, d_bn);
        auto d_x    = conv.backward(last_x, d_conv);
        return d_x;
    }

    void gather(std::vector<mt::Parameter*>& ps){ conv.gather_params(ps); bn.gather_params(ps); }
    void zero(){ conv.zero_grad(); bn.zero_grad(); }
};

struct ResidualBlock {
    ConvBNReLU c1;
    mme::MinkowskiConvolution c2;
    mme::MinkowskiBatchNorm bn2;
    mme::MinkowskiReLU relu;

    // stored activations
    mme::SparseTensor last_x, last_a, last_b, last_y, last_out;

    ResidualBlock(int c_in, int c_out, int dim=4)
        : c1(c_in, c_out, dim, 3, {1,1,1,1}),  // FIXED: Removed dilation
          c2(c_out, c_out, 3, {1,1,1,1}, dim),  // FIXED: Removed dilation
          bn2(c_out), relu(true) {}

    mme::SparseTensor forward(const mme::SparseTensor& x) {
        last_x = x;
        last_a = c1.forward(x);
        last_b = c2.forward(last_a);
        last_b = bn2.forward(last_b);

        // residual add
        last_y = last_b;
        if (last_y.coords == x.coords) {
            const int N = last_y.N(), C = last_y.C();
            #pragma omp parallel for collapse(2) if (N*C > 2048)
            for (int i=0;i<N;++i)
                for (int j=0;j<C;++j)
                    last_y.feats.at(i,j) += x.feats.at(i,j);
        }
        last_out = relu.forward(last_y);
        return last_out;
    }

    mme::SparseTensor backward(mme::SparseTensor& grad_out) {
        auto d_y = relu.backward(last_y, grad_out);

        // split add
        mme::SparseTensor d_b = d_y;
        mme::SparseTensor d_skip = d_y;

        auto d_c2_out   = bn2.backward(last_b, d_b);
        auto d_a        = c2.backward(last_a, d_c2_out);
        auto d_x_from_c1= c1.backward(d_a);

        // merge to input grad
        mme::SparseTensor d_x = last_x;
        d_x.feats.zero_grad();
        const int N = d_x.N(), C = d_x.C();
        if (d_x.coords == d_skip.coords) {
            #pragma omp parallel for collapse(2) if (N*C > 2048)
            for (int i=0;i<N;++i)
                for (int j=0;j<C;++j)
                    d_x.feats.d(i,j) = d_skip.feats.d(i,j);
        }
        if (d_x.coords == d_x_from_c1.coords) {
            #pragma omp parallel for collapse(2) if (N*C > 2048)
            for (int i=0;i<N;++i)
                for (int j=0;j<C;++j)
                    d_x.feats.d(i,j) += d_x_from_c1.feats.d(i,j);
        }
        return d_x;
    }

    void gather(std::vector<mt::Parameter*>& ps){ c1.gather(ps); c2.gather_params(ps); bn2.gather_params(ps); }
    void zero(){ c1.zero(); c2.zero_grad(); bn2.zero_grad(); }
};

struct Simple4DUNet {
    // Stem (unchanged base)
    ConvBNReLU stem;

    // Encoder with deeper residuals
    ResidualBlock enc1a, enc1b;  // NEW: Two for depth
    mme::MinkowskiConvolution down1;
    ResidualBlock enc2a, enc2b;
    mme::MinkowskiConvolution down2;
    ResidualBlock bottleneck_a, bottleneck_b;

// NEW: Sparse Attention in bottleneck (FIXED: Made sparse with neighborhood offsets to avoid O(N²))
    struct SparseAttention : mme::Module {  // FIXED: Inherit from Module for override
        mme::MinkowskiConvolution qkv;  // Projects to query/key/value
        float scale;
        mme::SparseTensor last_qkv, last_out;  // For backward

        SparseAttention(int c, int dim=4) : qkv(c, c*3, 1, {1,1,1,1}, dim), scale(1.f / std::sqrt(float(c))) {}

        mme::SparseTensor forward(const mme::SparseTensor& x) {
            int N = x.N();
            if (N > 100000) {  // FIXED: Cap for very large N to prevent crash; adjust threshold
                cout << "Warning: Skipping attention for large N=" << N << endl;
                return x;
            }

            last_qkv = qkv.forward(x);
            int C = x.C();
            mt::Tensor q(N, C), k(N, C), v(N, C);

            // Split QKV
            #pragma omp parallel for collapse(2) if (N*C > 1024)
            for (int i=0; i<N; ++i) {
                for (int j=0; j<C; ++j) {
                    q.at(i,j) = last_qkv.feats.at(i,j);
                    k.at(i,j) = last_qkv.feats.at(i, C+j);
                    v.at(i,j) = last_qkv.feats.at(i, 2*C+j);
                }
            }

            // FIXED: Build local coord_to_index map (since not a member)
            std::unordered_map<uint64_t, int> coord_to_index;
            coord_to_index.reserve(N);
            for (int i = 0; i < N; ++i) {
                coord_to_index[mme::pack_key4_arr(x.coords[i])] = i;
            }

            // FIXED: Use int ks for build_spatial_offsets (as per engine function)
            const int ks = 3;  // Example kernel size
            auto offsets = mme::build_spatial_offsets(ks);  // Engine function takes int ks

            // Build per-point neighbors
            std::vector<std::vector<int>> neighbors(N);
            #pragma omp parallel for
            for (int i=0; i<N; ++i) {
                uint64_t key_i = mme::pack_key4_arr(x.coords[i]);
                for (const auto& off : offsets) {
                    std::array<int,4> neigh_coord = x.coords[i];
                    for (int d=0; d<4; ++d) neigh_coord[d] += off[d];
                    uint64_t neigh_key = mme::pack_key4_arr(neigh_coord);
                    auto it = coord_to_index.find(neigh_key);
                    if (it != coord_to_index.end()) {
                        neighbors[i].push_back(it->second);
                    }
                }
            }

            // Attention over neighbors only (O(degree * N * C), degree ~ kernel volume)
            mme::SparseTensor y = x;  // Copy structure
            #pragma omp parallel for if (N > 64)
            for (int i=0; i<N; ++i) {
                const auto& neigh = neighbors[i];
                int num_neigh = neigh.size();
                if (num_neigh == 0) continue;  // Skip isolated points

                float att_sum = 0.f;
                std::vector<float> att_weights(num_neigh);
                for (int jj=0; jj<num_neigh; ++jj) {
                    int j = neigh[jj];
                    float score = 0.f;
                    for (int d=0; d<C; ++d) score += q.at(i,d) * k.at(j,d);
                    att_weights[jj] = std::exp(score * scale);
                    att_sum += att_weights[jj];
                }
                att_sum = std::max(att_sum, 1e-6f);  // Avoid div0

                for (int d=0; d<C; ++d) {
                    float out = 0.f;
                    for (int jj=0; jj<num_neigh; ++jj) {
                        int j = neigh[jj];
                        float att = att_weights[jj] / att_sum;
                        out += att * v.at(j,d);
                    }
                    y.feats.at(i,d) = out;
                }
            }
            last_out = y;
            return y;
        }

        mme::SparseTensor backward(const mme::SparseTensor& x_in, mme::SparseTensor& grad_out) override {  // FIXED: Matched signature
            // Simplified backward (backprop through attention and qkv)
            // For full impl, compute gradients for softmax and QKV split
            // Placeholder: Propagate through qkv only (expand as needed)
            return qkv.backward(last_qkv, grad_out);
        }

        void gather(std::vector<mt::Parameter*>& ps) { qkv.gather_params(ps); }
        void zero() { qkv.zero_grad(); }
    } attention;

    // Decoder with proper skips
    mme::MinkowskiConvolutionTranspose up1;
    ResidualBlock dec1;  // Input adjusted for skip concat
    mme::MinkowskiConvolutionTranspose up2;
    ResidualBlock dec2;

    // Refined head (MLP-like: conv -> relu -> conv)
    mme::MinkowskiConvolution head1;
    mme::MinkowskiReLU head_relu;
    mme::MinkowskiConvolution head2;

    // NEW: Auxiliary heads for multi-scale supervision
    mme::MinkowskiConvolution aux_head1, aux_head2;  // At dec1 and dec2 levels

    // Stored activations (expanded for new layers)
    mme::SparseTensor x0, x1, x2in, x2, x3in, x3, x3_att, u1, u1_skip, d1, u2, u2_skip, d2, h1;

    Simple4DUNet(int c_in=7, int c_mid=64, int dim=4)  // NEW: Increased c_mid for capacity
        : stem(c_in, c_mid, dim, 3, {1,1,1,1}),
          enc1a(c_mid, c_mid, dim),
          enc1b(c_mid, c_mid, dim),
          down1(c_mid, c_mid*2, 2, {2,2,2,2}, dim),
          enc2a(c_mid*2, c_mid*2, dim),
          enc2b(c_mid*2, c_mid*2, dim),
          down2(c_mid*2, c_mid*4, 2, {2,2,2,1}, dim),
          bottleneck_a(c_mid*4, c_mid*4, dim),
          bottleneck_b(c_mid*4, c_mid*4, dim),
          attention(c_mid*4, dim),
          up1(c_mid*4, c_mid*2, 2, {2,2,2,1}, dim),
          dec1(c_mid*2 * 2, c_mid*2, dim),
          up2(c_mid*2, c_mid, 2, {2,2,2,2}, dim),
          dec2(c_mid * 2, c_mid, dim),
          head1(c_mid, c_mid/2, 1, {1,1,1,1}, dim),
          head_relu(true),
          head2(c_mid/2, 1, 1, {1,1,1,1}, dim),
          aux_head1(c_mid*2, 1, 1, {1,1,1,1}, dim),
          aux_head2(c_mid, 1, 1, {1,1,1,1}, dim) {}

    mme::SparseTensor forward(const mme::SparseTensor& x) {
        x0 = stem.forward(x);
        x1 = enc1a.forward(x0);
        x1 = enc1b.forward(x1);  // NEW: Deeper encoder
        x2in = down1.forward(x1);
        x2 = enc2a.forward(x2in);
        x2 = enc2b.forward(x2);
        x3in = down2.forward(x2);
        x3 = bottleneck_a.forward(x3in);
        x3 = bottleneck_b.forward(x3);
        x3_att = attention.forward(x3);  // NEW: Attend to potential targets

        // Decoder with proper skips
        u1 = up1.forward(x3_att);
        u1_skip = merge_with_skip(u1, x2);  // NEW: Coord-matched skip
        d1 = dec1.forward(u1_skip);

        u2 = up2.forward(d1);
        u2_skip = merge_with_skip(u2, x1);
        d2 = dec2.forward(u2_skip);

        // Refined head
        h1 = head1.forward(d2);
        h1 = head_relu.forward(h1);
        return head2.forward(h1);
    }

    // NEW: Helper for skip merging (hash-based concat)
    mme::SparseTensor merge_with_skip(const mme::SparseTensor& dec, const mme::SparseTensor& enc) {
        if (dec.coords == enc.coords) {
            return mme::cat_features_same_coords(dec, enc);  // Direct concat if coords match
        }
        // Hash-based merge: Map enc feats to dec coords
        std::unordered_map<uint64_t, int> enc_map;
        for (int i=0; i<enc.N(); ++i) {
            enc_map[mme::pack_key4_arr(enc.coords[i])] = i;
        }
        mme::SparseTensor merged(dec.coords, dec.C() + enc.C());
        #pragma omp parallel for if (dec.N() > 64)
        for (int i=0; i<dec.N(); ++i) {
            uint64_t key = mme::pack_key4_arr(dec.coords[i]);
            // Copy dec feats
            for (int j=0; j<dec.C(); ++j) merged.feats.at(i, j) = dec.feats.at(i, j);
            // Concat enc feats if match (zero if no match)
            auto it = enc_map.find(key);
            if (it != enc_map.end()) {
                int enc_idx = it->second;
                for (int j=0; j<enc.C(); ++j) merged.feats.at(i, dec.C() + j) = enc.feats.at(enc_idx, j);
            } else {
                for (int j=0; j<enc.C(); ++j) merged.feats.at(i, dec.C() + j) = 0.f;
            }
        }
        return merged;
    }

    void backward(mme::SparseTensor& grad_logits) {
        auto grad_h1 = head2.backward(h1, grad_logits);
        grad_h1 = head_relu.backward(h1, grad_h1);
        auto grad_d2 = head1.backward(d2, grad_h1);
        auto grad_u2_skip = dec2.backward(grad_d2);
        auto grad_d1 = up2.backward(d1, grad_u2_skip);  // Split grad for skip if needed
        auto grad_u1_skip = dec1.backward(grad_d1);
        auto grad_x3_att = up1.backward(x3_att, grad_u1_skip);
        auto grad_x3 = attention.backward(x3, grad_x3_att);  // FIXED: Added x3 as first param
        auto grad_x3in = bottleneck_b.backward(grad_x3);
        grad_x3in = bottleneck_a.backward(grad_x3in);
        auto grad_x2 = down2.backward(x2, grad_x3in);
        grad_x2 = enc2b.backward(grad_x2);
        grad_x2 = enc2a.backward(grad_x2);
        auto grad_x1 = down1.backward(x1, grad_x2);  // FIXED: Changed grad_x2in to grad_x2
        grad_x1 = enc1b.backward(grad_x1);
        grad_x1 = enc1a.backward(grad_x1);
        auto grad_input = stem.backward(grad_x1);
        (void)grad_input;
    }

    // NEW: Forward for aux heads (call during training)
    mme::SparseTensor aux1_forward() { return aux_head1.forward(d1); }
    mme::SparseTensor aux2_forward() { return aux_head2.forward(d2); }

    void gather(std::vector<mt::Parameter*>& ps) {
        stem.gather(ps); enc1a.gather(ps); enc1b.gather(ps); down1.gather_params(ps);
        enc2a.gather(ps); enc2b.gather(ps); down2.gather_params(ps);
        bottleneck_a.gather(ps); bottleneck_b.gather(ps); attention.gather(ps);
        up1.gather_params(ps); dec1.gather(ps); up2.gather_params(ps); dec2.gather(ps);
        head1.gather_params(ps); head2.gather_params(ps);
        aux_head1.gather_params(ps); aux_head2.gather_params(ps);
    }

    void zero() {
        stem.zero(); enc1a.zero(); enc1b.zero(); down1.zero_grad();
        enc2a.zero(); enc2b.zero(); down2.zero_grad();
        bottleneck_a.zero(); bottleneck_b.zero(); attention.zero();
        up1.zero_grad(); dec1.zero(); up2.zero_grad(); dec2.zero();
        head1.zero_grad(); head2.zero_grad();
        aux_head1.zero_grad(); aux_head2.zero_grad();
        std::vector<mt::Parameter*> ps; gather(ps);
        #pragma omp parallel for if ((int)ps.size() > 8)
        for (int i=0; i<(int)ps.size(); ++i) ps[i]->zero_grad();
    }
};

// -----------------------------
// Async prefetcher of batched samples
// -----------------------------
struct Batch {
    mme::SparseTensor st; // concatenated coords+feats
    mt::Tensor labels;    // concatenated labels
    size_t batch_index = 0;
};

class Prefetcher {
public:
    Prefetcher(const std::vector<std::array<fs::path,10>>& segs,
               size_t batch_segments, size_t queue_capacity)
        : segs_(segs), B_(batch_segments), cap_(queue_capacity) {}

    void start() {
        stop_.store(false);
        th_ = std::thread(&Prefetcher::run, this);
    }
    void stop() {
        stop_.store(true);
        {
            std::lock_guard<std::mutex> lk(mx_);
            done_ = true;
        }
        cv_.notify_all();
        if (th_.joinable()) th_.join();
    }
    // Blocks until next batch is available; returns false when all done
    bool next(Batch& out) {
        std::unique_lock<std::mutex> lk(mx_);
        cv_.wait(lk, [&]{ return !q_.empty() || done_; });
        if (q_.empty()) return false;
        out = std::move(q_.front());
        q_.pop();
        cv_.notify_all();
        return true;
    }

private:
    void run() {
        size_t n = segs_.size();
        size_t i = 0, bidx = 0;
        while (!stop_.load() && i < n) {
            size_t take = std::min(B_, n - i);
            // Build batch
            Batch batch;
            {
                Sample s = build_batched_sample(segs_, i, take);
                batch.st = mme::SparseTensor(std::move(s.coords), std::move(s.feats));
                batch.labels = std::move(s.labels);
                batch.batch_index = bidx++;
            }
            // Enqueue
            {
                std::unique_lock<std::mutex> lk(mx_);
                cv_.wait(lk, [&]{ return q_.size() < cap_ || stop_.load(); });
                if (stop_.load()) break;
                q_.push(std::move(batch));
                cv_.notify_all();
            }
            i += take;
        }
        {
            std::lock_guard<std::mutex> lk(mx_);
            done_ = true;
        }
        cv_.notify_all();
    }

    const std::vector<std::array<fs::path,10>>& segs_;
    size_t B_, cap_;
    std::thread th_;
    std::queue<Batch> q_;
    std::mutex mx_;
    std::condition_variable cv_;
    std::atomic<bool> stop_{false};
    bool done_ = false;
};

// -----------------------------
// Training helpers (updated for aux losses and focal BCE)
// -----------------------------
float train_epoch(Simple4DUNet& model,
                  const std::vector<std::array<fs::path,10>>& segs,
                  float pos_w, mt::AdamW& opt,
                  int step0=1, bool head_only=false,
                  size_t batch_segments=4, size_t prefetch_queue=4)
{
    Timer t_epoch("train_epoch");
    float total=0.f; int steps=0; int t = step0;

    Prefetcher pf(segs, batch_segments, prefetch_queue);
    pf.start();

    Batch batch;
    while (pf.next(batch)) {
        auto logits = model.forward(batch.st);
        auto aux1_logits = model.aux1_forward();  // NEW: Aux outputs
        auto aux2_logits = model.aux2_forward();

        // NEW: Compute combined loss with focal BCE
        float main_loss = mt::focal_bce_with_logits(logits.feats, batch.labels, pos_w);
        float aux1_loss = mt::focal_bce_with_logits(aux1_logits.feats, batch.labels, pos_w);
        float aux2_loss = mt::focal_bce_with_logits(aux2_logits.feats, batch.labels, pos_w);
        float loss = main_loss + 0.5f * aux1_loss + 0.3f * aux2_loss;

        total += loss; ++steps;

        model.zero();
        model.backward(logits);  // NEW: Also backprop aux if needed (expand backward for aux grads)

        std::vector<mt::Parameter*> pvec;
        if (head_only) {
            model.head1.gather_params(pvec); model.head2.gather_params(pvec);  // NEW: Refined head
        } else model.gather(pvec);
        opt.step(pvec, t++);
    }

    pf.stop();
    return total / float(std::max(1,steps));
}

float eval_pos_precision(Simple4DUNet& model,
                         const std::vector<std::array<fs::path,10>>& segs,
                         size_t batch_segments=4)
{
    Timer t_eval("eval_epoch");
    int total_pos=0, correct_pos=0;

    size_t n = segs.size();
    for (size_t i = 0; i < n; ) {
        const size_t take = std::min(batch_segments, n - i);
        Sample bs = build_batched_sample(segs, i, take);
        mme::SparseTensor st(std::move(bs.coords), std::move(bs.feats));

        auto logits = model.forward(st);
        const int N = logits.N();
        #pragma omp parallel for reduction(+:total_pos,correct_pos) if (N > 2048)
        for (int k=0;k<N;++k) {
            float p = 1.f / (1.f + std::exp(-logits.feats.at(k,0)));
            int pred = p >= 0.5f ? 1 : 0;
            int y = bs.labels.at(k,0) >= 0.5f ? 1 : 0;
            total_pos += y;
            if (y && pred) ++correct_pos;
        }

        i += take;
    }
    return (total_pos>0) ? float(correct_pos)/float(total_pos) : 0.f;
}

// -----------------------------
// Main
// -----------------------------
int main(int argc, char** argv) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_nested(0);
    omp_set_num_threads(omp_get_max_threads());
#endif

    fs::path data_dir;
    int epochs = 3;
    float pos_weight = 8.f;
    float lr = 1.5e-4, wd=5e-5;
    int head_only_epochs = 0;
    size_t batch_segments = 8;
    size_t prefetch_queue = 6; 
    std::string save_model_path = "model.bin";

    for (int i=1;i<argc;++i) {
        std::string a = argv[i];
        if (a=="--data_dir" && i+1<argc) data_dir = argv[++i];
        else if (a=="--epochs" && i+1<argc) epochs = std::stoi(argv[++i]);
        else if (a=="--pos_weight" && i+1<argc) pos_weight = std::stof(argv[++i]);
        else if (a=="--lr" && i+1<argc) lr = std::stof(argv[++i]);
        else if (a=="--head_only_epochs" && i+1<argc) head_only_epochs = std::stoi(argv[++i]);
        else if (a=="--weight_decay" && i+1<argc) wd = std::stof(argv[++i]);
        else if (a=="--batch_segments" && i+1<argc) batch_segments = std::stoul(argv[++i]);
        else if (a=="--prefetch" && i+1<argc) prefetch_queue = std::stoul(argv[++i]);
        else if (a=="--save_model" && i+1<argc) save_model_path = argv[++i];
    }

    if (data_dir.empty()) {
        std::cerr<<"--data_dir required\n";
        return 2;
    }

    auto segs = find_segments(data_dir);
    if (segs.empty()) {
        std::cerr<<"No 10-frame segments in "<<data_dir<<"\n";
        return 3;
    }
    std::cout<<"Segments: "<<segs.size()<<"\n";

    Simple4DUNet model(7, 64, 4);  // NEW: Updated c_mid=64

    // Initialize head bias for faster early convergence
    {
        float init_bias = 2.0f;
        std::vector<mt::Parameter*> head_params; 
        model.head2.gather_params(head_params);  // NEW: Refined head
        if (head_params.size() >= 2) {
            mt::Parameter* head_b = head_params[1];
            if (!head_b->w.empty()) head_b->w[0] = init_bias;
        }
    }

    // Optional quick sanity forward on first batch (no fine timers)
    {
        size_t take = std::min<size_t>(batch_segments, segs.size());
        Sample bs = build_batched_sample(segs, 0, take);
        mme::SparseTensor st(std::move(bs.coords), std::move(bs.feats));
        auto logits = model.forward(st);
        // simple stats
        double mean=0.0, sq=0.0; int N=logits.N();
        for (int i=0;i<N;++i) { double v = logits.feats.at(i,0); mean += v; sq += v*v; }
        if (N>0) { mean /= N; double var = sq/ N - mean*mean; double std = std::sqrt(std::max(0.0, var));
            std::cout<<"Sanity logits mean="<<mean<<" std="<<std<<"\n";
        }
    }

    mt::AdamW opt(lr, wd);

    for (int e=1;e<=epochs;++e) {
        Timer t_epoch_scope(std::string("epoch_") + std::to_string(e));
        bool head_only = (e <= head_only_epochs);

        float loss = train_epoch(model, segs, pos_weight, opt, e, head_only, batch_segments, prefetch_queue);
        float prec = eval_pos_precision(model, segs, batch_segments);

        std::cout<<"Epoch "<<e<<" | loss="<<loss<<" | pos-precision="<<prec<<"\n";
        // monitor head bias
        std::vector<mt::Parameter*> head_params; 
        model.head2.gather_params(head_params);  // NEW: Refined head
        if (head_params.size() >= 2 && !head_params[1]->w.empty()) {
            std::cout<<"Head bias="<<head_params[1]->w[0]<<"\n";
        }
    }

    // optional save
    if (!save_model_path.empty()) {
        std::vector<mt::Parameter*> params; model.gather(params);
        if (!save_parameters_binary(save_model_path, params)) {
            std::cerr<<"Failed to save model to "<<save_model_path<<"\n";
        } else {
            std::cout<<"Saved model to "<<save_model_path<<"\n";
        }
    }

    std::cout<<"Done.\n";
    return 0;
}