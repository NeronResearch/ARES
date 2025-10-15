// infer.cpp
// Inference for Simple4DUNet (identical to trainer's architecture and param order).
// Produces per-voxel probability heatmaps for each 10-frame segment.
// Usage: infer <model.bin> <frames_dir> <out_dir>

#include <iostream>
#include <filesystem>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "mini_numpy.hpp"
#include "mini_torch.hpp"
#include "mini_minkowski_engine.hpp"
#include "json_loader.hpp"
#include "model_io.hpp"

namespace fs = std::filesystem;

// -----------------------------
// Model blocks (duplicated from trainer)
// -----------------------------
struct ConvBNReLU {
    mme::MinkowskiConvolution conv;
    mme::MinkowskiBatchNorm bn;
    mme::MinkowskiReLU relu;

    // cached for backward in trainer, not needed here but harmless
    mme::SparseTensor last_x, last_after_conv, last_after_bn, last_out;

    ConvBNReLU(int c_in, int c_out, int dim=4, int ks=3, std::array<int,4> stride={1,1,1,1})
        : conv(c_in, c_out, ks, stride, dim), bn(c_out), relu(true) {}

    mme::SparseTensor forward(const mme::SparseTensor& x) {
        last_x = x;
        last_after_conv = conv.forward(x);
        last_after_bn  = bn.forward(last_after_conv);
        last_out       = relu.forward(last_after_bn);
        return last_out;
    }

    void gather(std::vector<mt::Parameter*>& ps){ conv.gather_params(ps); bn.gather_params(ps); }
    void zero(){ conv.zero_grad(); bn.zero_grad(); }
};

struct ResidualBlock {
    ConvBNReLU c1;
    mme::MinkowskiConvolution c2;
    mme::MinkowskiBatchNorm bn2;
    mme::MinkowskiReLU relu;

    // cached tensors (unused in inference)
    mme::SparseTensor last_x, last_a, last_b, last_y, last_out;

    ResidualBlock(int c, int dim=4)
        : c1(c,c,dim,3), c2(c,c,3,{1,1,1,1},dim), bn2(c), relu(true) {}

    mme::SparseTensor forward(const mme::SparseTensor& x) {
        last_x = x;
        last_a = c1.forward(x);
        last_b = c2.forward(last_a);
        last_b = bn2.forward(last_b);

        // residual add (coords expected to match in this toy engine)
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

    void gather(std::vector<mt::Parameter*>& ps){ c1.gather(ps); c2.gather_params(ps); bn2.gather_params(ps); }
    void zero(){ c1.zero(); c2.zero_grad(); bn2.zero_grad(); }
};

struct Simple4DUNet {
    // stem
    ConvBNReLU stem;
    ResidualBlock enc1;
    mme::MinkowskiConvolution down1;
    ResidualBlock enc2;
    mme::MinkowskiConvolution down2;
    ResidualBlock bottleneck;
    // decoder
    mme::MinkowskiConvolutionTranspose up1;
    ResidualBlock dec1;
    mme::MinkowskiConvolutionTranspose up2;
    ResidualBlock dec2;
    // head
    mme::MinkowskiConvolution head;

    // cache (unused in inference)
    mme::SparseTensor x0, x1, x2in, x2, x3in, x3, u1, d1, u2, d2;

    Simple4DUNet(int c_in=7, int c_mid=32, int dim=4)
        : stem(c_in, c_mid, dim, 3),
          enc1(c_mid, dim),
          down1(c_mid, c_mid*2, 2, {2,2,2,2}, dim),
          enc2(c_mid*2, dim),
          down2(c_mid*2, c_mid*4, 2, {2,2,2,1}, dim),
          bottleneck(c_mid*4, dim),
          up1(c_mid*4, c_mid*2, 2, {2,2,2,1}, dim),
          dec1(c_mid*2, dim),
          up2(c_mid*2, c_mid, 2, {2,2,2,2}, dim),
          dec2(c_mid, dim),
          head(c_mid, 1, 1, {1,1,1,1}, dim) {}

    mme::SparseTensor forward(const mme::SparseTensor& x) {
        x0 = stem.forward(x);
        x1 = enc1.forward(x0);
        x2in = down1.forward(x1);
        x2 = enc2.forward(x2in);
        x3in = down2.forward(x2);
        x3 = bottleneck.forward(x3in);
        u1 = up1.forward(x3);
        d1 = dec1.forward(u1);
        u2 = up2.forward(d1);
        d2 = dec2.forward(u2);
        return head.forward(d2);
    }

    void gather(std::vector<mt::Parameter*>& ps) {
        stem.gather(ps); enc1.gather(ps); down1.gather_params(ps);
        enc2.gather(ps); down2.gather_params(ps); bottleneck.gather(ps);
        up1.gather_params(ps); dec1.gather(ps); up2.gather_params(ps); dec2.gather(ps); head.gather_params(ps);
    }
};

// -----------------------------
// IO helpers
// -----------------------------
static inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

static bool ensure_dir(const fs::path& p) {
    std::error_code ec;
    if (fs::exists(p, ec)) return fs::is_directory(p, ec);
    return fs::create_directories(p, ec);
}

static void write_segment_csv(const fs::path& out_csv,
                              const mme::SparseTensor& logits_st)
{
    std::ofstream f(out_csv);
    if (!f) {
        std::cerr << "Failed to open " << out_csv << " for write\n";
        return;
    }
    f << "x,y,z,t,prob\n";
    const int N = logits_st.N();
    for (int i=0;i<N;++i) {
        const auto& c = logits_st.coords[i];
        const float logit = logits_st.feats.at(i,0);
        const float p = sigmoid(logit);
        f << c[0] << ',' << c[1] << ',' << c[2] << ',' << c[3] << ',' << p << '\n';
    }
}

// -----------------------------
// Inference over directory of 10-frame segments
// -----------------------------
int main(int argc, char** argv) {
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_nested(0);
    omp_set_num_threads(omp_get_max_threads());
#endif

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <model.bin> <frames_dir> <out_dir>\n";
        return 1;
    }
    const fs::path model_path = argv[1];
    const fs::path frames_dir = argv[2];
    const fs::path out_dir    = argv[3];

    if (!fs::exists(model_path)) {
        std::cerr << "Model file not found: " << model_path << "\n";
        return 1;
    }
    if (!fs::exists(frames_dir) || !fs::is_directory(frames_dir)) {
        std::cerr << "Frames dir not found: " << frames_dir << "\n";
        return 1;
    }
    if (!ensure_dir(out_dir)) {
        std::cerr << "Failed to create output dir: " << out_dir << "\n";
        return 1;
    }

    // Build identical network and load weights
    Simple4DUNet model(7, 32, 4);
    std::vector<mt::Parameter*> params;
    model.gather(params);

    // Optional: report expected param count to help debug mismatches
    std::cout << "Expected parameter tensors: " << params.size() << "\n";

    if (!load_parameters_binary(model_path.string(), params)) {
        std::cerr << "Failed to load model weights from " << model_path << "\n";
        return 2;
    }
    // Quick checksum-like print
    double wsum = 0.0;
    size_t wcnt = 0;
    for (auto* p : params) {
        for (float w : p->w) { wsum += w; ++wcnt; }
    }
    std::cout << "Loaded weights: tensors=" << params.size()
              << " scalars=" << wcnt
              << " sum=" << wsum << "\n";

    // Find 10-frame segments
    auto segs = find_segments(frames_dir);
    if (segs.empty()) {
        std::cerr << "No 10-frame segments in " << frames_dir << "\n";
        return 3;
    }
    std::cout << "Segments: " << segs.size() << "\n";

    // Process each segment: build Sample -> SparseTensor -> forward -> CSV
    for (size_t si=0; si<segs.size(); ++si) {
        // Build inputs exactly like training
        Sample s = build_sample_from_segment(segs[si]);
        mme::SparseTensor st(std::move(s.coords), std::move(s.feats));

        // Forward
        mme::SparseTensor logits = model.forward(st);

        // Stats
        const int N = logits.N();
        double mean=0.0, sq=0.0;
        for (int i=0;i<N;++i) { double v = logits.feats.at(i,0); mean += v; sq += v*v; }
        if (N>0) {
            mean /= N;
            double var = sq / N - mean*mean;
            double sd  = std::sqrt(std::max(0.0, var));
            std::cout << "Segment " << si << ": logits mean=" << mean << " std=" << sd << "\n";
        } else {
            std::cout << "Segment " << si << ": empty\n";
        }

        // Derive a filename from first and last frame ID in the segment
        int first_idx = std::stoi(segs[si][0].stem().string());
        int last_idx  = std::stoi(segs[si][9].stem().string());
        fs::path csv_path = out_dir / (std::to_string(first_idx) + "-" + std::to_string(last_idx) + "_heatmap.csv");

        // Write CSV: x,y,z,t,prob
        write_segment_csv(csv_path, logits);
    }

    std::cout << "Done.\n";
    return 0;
}
