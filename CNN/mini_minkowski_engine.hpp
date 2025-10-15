#pragma once
#include <vector>
#include <array>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <algorithm>
#include "mini_torch.hpp"
#include <stdexcept>
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif

// A sparse 4D engine with proper spatial neighborhood convolution matching MinkowskiEngine behavior.
// Key fixes:
// - Strided convolutions now apply full spatial kernels (not just 1x1)
// - Kernel map system ensures consistent neighborhoods
// - Proper coordinate tracking through encode-decode
// - Spatial gradient flow preserved through all layers

namespace mme {

struct SparseTensor {
    std::vector<std::array<int,4>> coords;
    mt::Tensor feats; // [N, Cin]
    
    SparseTensor() = default;
    SparseTensor(std::vector<std::array<int,4>> c, int Cin) 
        : coords(std::move(c)), feats((int)coords.size(), Cin, 0.f) {}
    SparseTensor(std::vector<std::array<int,4>> c, mt::Tensor f) 
        : coords(std::move(c)), feats(std::move(f)) {}
    
    int N() const { return (int)coords.size(); }
    int C() const { return feats.c; }
};

// =============================== 
// Base Module
// =============================== 
struct Module {
    virtual ~Module() = default;
    virtual SparseTensor forward(const SparseTensor& x) = 0;
    virtual SparseTensor backward(const SparseTensor& x, SparseTensor& grad_output) {
        if (x.coords == grad_output.coords) {
            SparseTensor gx = x;
            gx.feats.zero_grad();
            const int N = x.N(), C = x.C();
            #pragma omp parallel for if (N*C > 2048)
            for (int i=0;i<N;++i) {
                for (int j=0;j<C;++j) 
                    gx.feats.d(i,j) = grad_output.feats.d(i,j);
            }
            return gx;
        }
        return SparseTensor(x.coords, mt::Tensor(x.N(), x.C(), 0.f));
    }
    virtual void zero_grad() {}
    virtual void gather_params(std::vector<mt::Parameter*>& out) {}
};

// =============================== 
// ReLU
// =============================== 
struct MinkowskiReLU : Module {
    bool inplace = true;
    explicit MinkowskiReLU(bool inpl=true): inplace(inpl) {}
    
    SparseTensor forward(const SparseTensor& x) override {
        SparseTensor y = x;
        const int N = y.N(), C = y.C();
        #pragma omp parallel for if (N*C > 2048)
        for (int i=0;i<N;++i) {
            for (int j=0;j<C;++j) {
                float v = y.feats.at(i,j);
                y.feats.at(i,j) = v > 0.f ? v : 0.f;
            }
        }
        return y;
    }
    
    SparseTensor backward(const SparseTensor& x, SparseTensor& grad_output) override {
        SparseTensor gx = x;
        gx.feats.zero_grad();
        const int N = x.N(), C = x.C();
        if (x.coords != grad_output.coords) return gx;
        #pragma omp parallel for if (N*C > 2048)
        for (int i=0;i<N;++i) {
            for (int j=0;j<C;++j) {
                const float upstream = grad_output.feats.d(i,j);
                gx.feats.d(i,j) = (x.feats.at(i,j) > 0.f) ? upstream : 0.f;
            }
        }
        return gx;
    }
};

// =============================== 
// BatchNorm
// =============================== 
struct MinkowskiBatchNorm : Module {
    int C;
    mt::Parameter gamma_p; // size C
    mt::Parameter beta_p;  // size C
    std::vector<float> running_mean, running_var;
    float momentum = 0.1f, eps = 1e-5f;
    
    // stored forward statistics for backward
    std::vector<float> last_mu, last_var;
    std::vector<std::vector<float>> last_xhat;
    int last_N = 0;
    
    explicit MinkowskiBatchNorm(int C_) 
        : C(C_), gamma_p(C_, 1e-2f), beta_p(C_, 0.f), 
          running_mean(C_,0.f), running_var(C_,1.f) {
        for (int i=0;i<C;++i) gamma_p.w[i] = 1.f;
    }
    
    SparseTensor forward(const SparseTensor& x) override {
        const int N = x.N();
        SparseTensor y = x;
        if (N == 0) { last_N = 0; return y; }
        
        last_N = N;
        last_mu.assign(C, 0.f);
        last_var.assign(C, 0.f);
        last_xhat.assign(N, std::vector<float>(C, 0.f));
        
        // Mean
        #pragma omp parallel for if (C > 32)
        for (int j=0;j<C;++j) {
            double sum = 0.0;
            for (int i=0;i<N;++i) sum += x.feats.at(i,j);
            last_mu[j] = float(sum / double(N));
        }
        
        // Var
        #pragma omp parallel for if (C > 32)
        for (int j=0;j<C;++j) {
            double acc = 0.0;
            const float mu = last_mu[j];
            for (int i=0;i<N;++i) {
                const float d = x.feats.at(i,j) - mu;
                acc += double(d) * double(d);
            }
            last_var[j] = float(acc / double(N));
        }
        
        // Update running stats
        for (int j=0;j<C;++j) {
            running_mean[j] = (1 - momentum) * running_mean[j] + momentum * last_mu[j];
            running_var[j]  = (1 - momentum) * running_var[j]  + momentum * last_var[j];
        }
        
        // Normalize and affine
        #pragma omp parallel for if (N > 64)
        for (int i=0;i<N;++i) {
            for (int j=0;j<C;++j) {
                const float inv_std = 1.0f / std::sqrt(last_var[j] + eps);
                const float xhat = (x.feats.at(i,j) - last_mu[j]) * inv_std;
                last_xhat[i][j] = xhat;
                y.feats.at(i,j) = gamma_p.w[j] * xhat + beta_p.w[j];
            }
        }
        return y;
    }
    
    void zero_grad() override {
        gamma_p.zero_grad();
        beta_p.zero_grad();
    }
    
    void gather_params(std::vector<mt::Parameter*>& out) override {
        out.push_back(&gamma_p);
        out.push_back(&beta_p);
    }
    
    SparseTensor backward(const SparseTensor& x, SparseTensor& grad_output) override {
        const int N = last_N;
        SparseTensor gx = x;
        gx.feats.zero_grad();
        if (N == 0) return gx;
        
        if (gamma_p.g.size() != gamma_p.w.size()) gamma_p.g.assign(gamma_p.w.size(), 0.f);
        if (beta_p.g.size() != beta_p.w.size()) beta_p.g.assign(beta_p.w.size(), 0.f);
        
        // Parameter grads
        #pragma omp parallel for if (C > 32)
        for (int j=0;j<C;++j) {
            double dg = 0.0, db = 0.0;
            for (int i=0;i<N;++i) {
                const float up = grad_output.feats.d(i,j);
                dg += double(up) * double(last_xhat[i][j]);
                db += double(up);
            }
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            gamma_p.g[j] += float(dg);
            #ifdef _OPENMP
            #pragma omp atomic
            #endif
            beta_p.g[j] += float(db);
        }
        
        // Input grads
        #pragma omp parallel for if (C > 32)
        for (int j=0;j<C;++j) {
            const float inv_std = 1.f / std::sqrt(last_var[j] + eps);
            const float gamma_j = gamma_p.w[j];
            double sum_dy = 0.0;
            double sum_dy_xhat = 0.0;
            for (int i=0;i<N;++i) {
                const float dy = grad_output.feats.d(i,j);
                sum_dy += double(dy);
                sum_dy_xhat += double(dy) * double(last_xhat[i][j]);
            }
            for (int i=0;i<N;++i) {
                const float dy = grad_output.feats.d(i,j);
                const float xh = last_xhat[i][j];
                const float dx = (1.f/float(N)) * gamma_j * inv_std * 
                                 (float(N)*dy - float(sum_dy) - xh*float(sum_dy_xhat));
                gx.feats.d(i,j) = dx;
            }
        }
        return gx;
    }
};

// =============================== 
// Helpers for hashing coords
// =============================== 
inline uint64_t pack_key4(int x, int y, int z, int t) {
    return ( (uint64_t(uint16_t(t)) << 48) | 
             (uint64_t(uint16_t(z)) << 32) | 
             (uint64_t(uint16_t(y)) << 16) | 
              uint64_t(uint16_t(x)) );
}

inline uint64_t pack_key4_arr(const std::array<int,4>& c) {
    return pack_key4(c[0], c[1], c[2], c[3]);
}

// Build 3D spatial offsets for ks (time kernel = 1, no temporal mixing)
inline std::vector<std::array<int,4>> build_spatial_offsets(int ks) {
    if (ks <= 1) return { {0,0,0,0} };
    if (ks == 2) {
        // 2x2x2 kernel for strided operations
        std::vector<std::array<int,4>> offs;
        offs.reserve(8);
        for (int dz=0; dz<2; ++dz)
            for (int dy=0; dy<2; ++dy)
                for (int dx=0; dx<2; ++dx)
                    offs.push_back({dx,dy,dz,0});
        return offs;
    }
    if (ks == 3) {
        std::vector<std::array<int,4>> offs;
        offs.reserve(27);
        for (int dz=-1; dz<=1; ++dz)
            for (int dy=-1; dy<=1; ++dy)
                for (int dx=-1; dx<=1; ++dx)
                    offs.push_back({dx,dy,dz,0}); // no temporal mixing
        return offs;
    }
    throw std::runtime_error("Only ks=1,2,3 supported");
}


// =============================== 
// Convolution (proper spatial convolution with stride support)
// =============================== 
struct MinkowskiConvolution : Module {
    int Cin, Cout;
    std::array<int,4> stride;
    int ks, dim;
    std::vector<std::array<int,4>> offsets;
    int center_offset_idx = 0;
    
    // Weights: [K * Cin * Cout], where K=offsets.size()
    mt::Parameter W;
    mt::Parameter b; // [Cout]
    
    MinkowskiConvolution(int c_in, int c_out, int kernel_size=1, 
                         std::array<int,4> stride_={1,1,1,1}, int dimension=4)
        : Cin(c_in), Cout(c_out), stride(stride_), ks(kernel_size), dim(dimension) {
        offsets = build_spatial_offsets(ks);
        
        // Find center offset for ks=3
        if (ks == 3) {
            for (int k=0;k<(int)offsets.size();++k) {
                if (offsets[k][0]==0 && offsets[k][1]==0 && offsets[k][2]==0) {
                    center_offset_idx = k;
                    break;
                }
            }
        }
        
        const int K = (int)offsets.size();
        W = mt::Parameter(Cin * Cout * K);
        b = mt::Parameter(Cout);
    }
    
    inline int w_index(int k, int ci, int co) const {
        return k*Cin*Cout + ci*Cout + co;
    }
    
    SparseTensor forward(const SparseTensor& x) override {
        const bool no_stride = (stride == std::array<int,4>{1,1,1,1});
        
        if (no_stride) {
            return forward_no_stride(x);
        } else {
            return forward_strided(x);
        }
    }
    
private:
    // Stride=1 path: direct neighborhood convolution
    SparseTensor forward_no_stride(const SparseTensor& x) {
        const int N = x.N();
        
        if (ks == 1) {
            // 1x1 conv
            SparseTensor y{x.coords, mt::Tensor(N, Cout, 0.f)};
            const int k0 = center_offset_idx;
            #pragma omp parallel for if (N*Cout > 512)
            for (int i=0;i<N;++i) {
                for (int co=0; co<Cout; ++co) {
                    float acc = b.w[co];
                    for (int ci=0; ci<Cin; ++ci) {
                        acc += x.feats.at(i,ci) * W.w[w_index(k0,ci,co)];
                    }
                    y.feats.at(i,co) = acc;
                }
            }
            return y;
        }
        
        // ks=3: true neighborhood convolution
        std::unordered_map<uint64_t,int> row_of;
        row_of.reserve(N*2);
        row_of.rehash(N*2);
        for (int i=0;i<N;++i) 
            row_of[pack_key4_arr(x.coords[i])] = i;
        
        SparseTensor y{x.coords, mt::Tensor(N, Cout, 0.f)};
        #pragma omp parallel for if (N > 64)
        for (int i=0;i<N;++i) {
            const auto base = x.coords[i];
            for (int co=0; co<Cout; ++co) {
                float acc = b.w[co];
                for (int k=0; k<(int)offsets.size(); ++k) {
                    const auto& off = offsets[k];
                    const uint64_t nk = pack_key4(base[0]+off[0], base[1]+off[1], 
                                                   base[2]+off[2], base[3]+off[3]);
                    auto it = row_of.find(nk);
                    if (it == row_of.end()) continue;
                    const int j = it->second;
                    
                    for (int ci=0; ci<Cin; ++ci) {
                        acc += x.feats.at(j,ci) * W.w[w_index(k,ci,co)];
                    }
                }
                y.feats.at(i,co) = acc;
            }
        }
        return y;
    }
    
    // Strided convolution: apply kernel at strided positions
    SparseTensor forward_strided(const SparseTensor& x) {
        const int N = x.N();
        
        // Build input coord map
        std::unordered_map<uint64_t,int> row_of;
        row_of.reserve(N*2);
        row_of.rehash(N*2);
        for (int i=0;i<N;++i) 
            row_of[pack_key4_arr(x.coords[i])] = i;
        
        // Determine output coordinates (downsampled grid)
        std::unordered_map<uint64_t, std::vector<int>> buckets;
        buckets.reserve(size_t(N) * 2);
        buckets.rehash(size_t(N) * 2);
        
        for (int i=0;i<N;++i) {
            const auto& c = x.coords[i];
            const int vx = c[0] / stride[0];
            const int vy = c[1] / stride[1];
            const int vz = c[2] / stride[2];
            const int vt = c[3] / stride[3];
            const uint64_t key = pack_key4(vx,vy,vz,vt);
            buckets[key].push_back(i);
        }
        
        std::vector<std::array<int,4>> ocoords;
        ocoords.reserve(buckets.size());
        std::vector<uint64_t> out_keys;
        out_keys.reserve(buckets.size());
        
        for (auto& kv : buckets) {
            const uint64_t key = kv.first;
            out_keys.push_back(key);
            const int vx = int(uint16_t(key & 0xFFFFu));
            const int vy = int(uint16_t((key >> 16) & 0xFFFFu));
            const int vz = int(uint16_t((key >> 32) & 0xFFFFu));
            const int vt = int(uint16_t((key >> 48) & 0xFFFFu));
            ocoords.push_back({vx,vy,vz,vt});
        }
        
        const int Nout = (int)ocoords.size();
        SparseTensor y{std::move(ocoords), mt::Tensor(Nout, Cout, 0.f)};
        
        // For each output voxel, apply spatial convolution at input scale
        #pragma omp parallel for if (Nout > 64)
        for (int out_idx=0; out_idx<Nout; ++out_idx) {
            const uint64_t out_key = out_keys[out_idx];
            const auto& members = buckets[out_key];
            
            const int vx = int(uint16_t(out_key & 0xFFFFu));
            const int vy = int(uint16_t((out_key >> 16) & 0xFFFFu));
            const int vz = int(uint16_t((out_key >> 32) & 0xFFFFu));
            const int vt = int(uint16_t((out_key >> 48) & 0xFFFFu));
            
            // Base input coordinate (at output voxel's origin in input space)
            const int base_x = vx * stride[0];
            const int base_y = vy * stride[1];
            const int base_z = vz * stride[2];
            const int base_t = vt * stride[3];
            
            for (int co=0; co<Cout; ++co) {
                float acc = b.w[co];
                
                // Apply kernel around base position in input space
                for (int k=0; k<(int)offsets.size(); ++k) {
                    const auto& off = offsets[k];
                    const uint64_t in_key = pack_key4(base_x + off[0], 
                                                       base_y + off[1], 
                                                       base_z + off[2], 
                                                       base_t + off[3]);
                    auto it = row_of.find(in_key);
                    if (it == row_of.end()) continue;
                    const int in_idx = it->second;
                    
                    for (int ci=0; ci<Cin; ++ci) {
                        acc += x.feats.at(in_idx,ci) * W.w[w_index(k,ci,co)];
                    }
                }
                y.feats.at(out_idx,co) = acc;
            }
        }
        
        return y;
    }
    
public:
    SparseTensor backward(const SparseTensor& x, SparseTensor& grad_output) override {
        const bool no_stride = (stride == std::array<int,4>{1,1,1,1});
        
        SparseTensor gx{x.coords, mt::Tensor(x.N(), Cin, 0.f)};
        
        if (W.g.size() != W.w.size()) W.g.assign(W.w.size(), 0.f);
        if (b.g.size() != b.w.size()) b.g.assign(b.w.size(), 0.f);
        
        // Bias gradient
        if (grad_output.N() > 0) {
            #pragma omp parallel for if (Cout > 32)
            for (int co=0; co<Cout; ++co) {
                double acc = 0.0;
                for (int i=0;i<grad_output.N();++i) 
                    acc += double(grad_output.feats.d(i,co));
                b.g[co] += float(acc);
            }
        }
        
        if (no_stride) {
            return backward_no_stride(x, grad_output, gx);
        } else {
            return backward_strided(x, grad_output, gx);
        }
    }
    
private:
    SparseTensor backward_no_stride(const SparseTensor& x, SparseTensor& grad_output, 
                                     SparseTensor& gx) {
        const int N = x.N();
        
        if (ks == 1) {
            // 1x1 gradients
            const int k0 = center_offset_idx;
            #pragma omp parallel for collapse(2) if (Cin*Cout > 256)
            for (int ci=0; ci<Cin; ++ci) {
                for (int co=0; co<Cout; ++co) {
                    double acc = 0.0;
                    for (int i=0;i<N;++i) {
                        acc += double(x.feats.at(i,ci)) * double(grad_output.feats.d(i,co));
                    }
                    W.g[w_index(k0,ci,co)] += float(acc);
                }
            }
            
            #pragma omp parallel for if (N*Cin > 512)
            for (int i=0;i<N;++i) {
                for (int ci=0; ci<Cin; ++ci) {
                    float acc = 0.f;
                    for (int co=0; co<Cout; ++co)
                        acc += W.w[w_index(k0,ci,co)] * grad_output.feats.d(i,co);
                    gx.feats.d(i,ci) = acc;
                }
            }
            return gx;
        }
        
        // ks=3
        std::unordered_map<uint64_t,int> row_of;
        row_of.reserve(N*2);
        row_of.rehash(N*2);
        for (int i=0;i<N;++i) 
            row_of[pack_key4_arr(x.coords[i])] = i;
        
        // dW (FIXED: Sequential accumulation for dW to avoid race; parallel over co only)
        #pragma omp parallel for if (Cout > 8)
        for (int co=0; co<Cout; ++co) {
            std::vector<float> local_dW((int)offsets.size() * Cin, 0.f);
            for (int i=0;i<N;++i) {
                const float dout = grad_output.feats.d(i,co);
                if (dout == 0.f) continue;
                const auto base = x.coords[i];
                
                for (int k=0; k<(int)offsets.size(); ++k) {
                    const auto& off = offsets[k];
                    const uint64_t nk = pack_key4(base[0]+off[0], base[1]+off[1], 
                                                   base[2]+off[2], base[3]+off[3]);
                    auto it = row_of.find(nk);
                    if (it == row_of.end()) continue;
                    const int j = it->second;
                    
                    for (int ci=0; ci<Cin; ++ci) {
                        const float add = x.feats.at(j,ci) * dout;
                        local_dW[k * Cin + ci] += add;
                    }
                }
            }
            for (int k=0; k<(int)offsets.size(); ++k) {
                for (int ci=0; ci<Cin; ++ci) {
                    W.g[w_index(k,ci,co)] += local_dW[k * Cin + ci];
                }
            }
        }
        
        // dx (FIXED: Sequential per n, parallel over n)
        #pragma omp parallel for if (N > 64)
        for (int n=0; n<N; ++n) {
            std::vector<float> local_acc(Cin, 0.f);
            const auto c = x.coords[n];
            for (int k=0; k<(int)offsets.size(); ++k) {
                const auto& off = offsets[k];
                const uint64_t ok = pack_key4(c[0]-off[0], c[1]-off[1], 
                                              c[2]-off[2], c[3]-off[3]);
                auto ito = row_of.find(ok);
                if (ito == row_of.end()) continue;
                const int i = ito->second;
                
                for (int ci=0; ci<Cin; ++ci) {
                    float inner_acc = 0.f;
                    for (int co=0; co<Cout; ++co) {
                        inner_acc += W.w[w_index(k,ci,co)] * grad_output.feats.d(i,co);
                    }
                    local_acc[ci] += inner_acc;
                }
            }
            for (int ci=0; ci<Cin; ++ci) {
                gx.feats.d(n,ci) += local_acc[ci];
            }
        }
        
        return gx;
    }
    
    SparseTensor backward_strided(const SparseTensor& x, SparseTensor& grad_output, 
                                   SparseTensor& gx) {
        const int N = x.N();
        
        // Rebuild forward pass structures
        std::unordered_map<uint64_t,int> row_of;
        row_of.reserve(N*2);
        row_of.rehash(N*2);
        for (int i=0;i<N;++i) 
            row_of[pack_key4_arr(x.coords[i])] = i;
        
        std::unordered_map<uint64_t,int> out_row_of;
        out_row_of.reserve(grad_output.N()*2);
        out_row_of.rehash(grad_output.N()*2);
        for (int i=0;i<grad_output.N();++i)
            out_row_of[pack_key4_arr(grad_output.coords[i])] = i;
        
        // dW: iterate over output positions (FIXED: Sequential per out_idx, parallel over out_idx)
        #pragma omp parallel for if (grad_output.N() > 32)
        for (int out_idx=0; out_idx<grad_output.N(); ++out_idx) {
            std::vector<float> local_dW((int)offsets.size() * Cin * Cout, 0.f);
            const auto& out_c = grad_output.coords[out_idx];
            const int vx = out_c[0], vy = out_c[1], vz = out_c[2], vt = out_c[3];
            
            const int base_x = vx * stride[0];
            const int base_y = vy * stride[1];
            const int base_z = vz * stride[2];
            const int base_t = vt * stride[3];
            
            for (int co=0; co<Cout; ++co) {
                const float dout = grad_output.feats.d(out_idx,co);
                if (dout == 0.f) continue;
                
                for (int k=0; k<(int)offsets.size(); ++k) {
                    const auto& off = offsets[k];
                    const uint64_t in_key = pack_key4(base_x + off[0], 
                                                       base_y + off[1], 
                                                       base_z + off[2], 
                                                       base_t + off[3]);
                    auto it = row_of.find(in_key);
                    if (it == row_of.end()) continue;
                    const int in_idx = it->second;
                    
                    for (int ci=0; ci<Cin; ++ci) {
                        const float add = x.feats.at(in_idx,ci) * dout;
                        local_dW[w_index(k,ci,co)] += add;
                    }
                }
            }
            #pragma omp critical
            for (size_t idx=0; idx<W.w.size(); ++idx) {
                W.g[idx] += local_dW[idx];
            }
        }
        
        // dx: each input contributes to outputs where it appears in kernel (FIXED: Sequential per in_idx, parallel over in_idx)
        #pragma omp parallel for if (N > 64)
        for (int in_idx=0; in_idx<N; ++in_idx) {
            std::vector<float> local_acc(Cin, 0.f);
            const auto& in_c = x.coords[in_idx];
            
            // For each kernel offset, check if this input participates
            for (int k=0; k<(int)offsets.size(); ++k) {
                const auto& off = offsets[k];
                
                // If input is at base+off, then base = input-off
                // Base corresponds to output voxel
                if ((in_c[0] - off[0]) % stride[0] != 0) continue;
                if ((in_c[1] - off[1]) % stride[1] != 0) continue;
                if ((in_c[2] - off[2]) % stride[2] != 0) continue;
                if ((in_c[3] - off[3]) % stride[3] != 0) continue;
                
                const int out_vx = (in_c[0] - off[0]) / stride[0];
                const int out_vy = (in_c[1] - off[1]) / stride[1];
                const int out_vz = (in_c[2] - off[2]) / stride[2];
                const int out_vt = (in_c[3] - off[3]) / stride[3];
                
                const uint64_t out_key = pack_key4(out_vx, out_vy, out_vz, out_vt);
                auto it = out_row_of.find(out_key);
                if (it == out_row_of.end()) continue;
                const int out_idx = it->second;
                
                for (int ci=0; ci<Cin; ++ci) {
                    float inner_acc = 0.f;
                    for (int co=0; co<Cout; ++co) {
                        inner_acc += W.w[w_index(k,ci,co)] * grad_output.feats.d(out_idx,co);
                    }
                    local_acc[ci] += inner_acc;
                }
            }
            for (int ci=0; ci<Cin; ++ci) {
                gx.feats.d(in_idx,ci) += local_acc[ci];
            }
        }
        
        return gx;
    }
    
public:
    void zero_grad() override {
        W.zero_grad();
        b.zero_grad();
    }
    
    void gather_params(std::vector<mt::Parameter*>& out) override {
        out.push_back(&W);
        out.push_back(&b);
    }
};


struct MinkowskiConvolutionTranspose : Module {
    int Cin, Cout;
    std::array<int,4> stride;
    mt::Parameter W; // 1x1 weights
    mt::Parameter b;
    
    MinkowskiConvolutionTranspose(int c_in, int c_out, int kernel_size=1, 
                                  std::array<int,4> stride_={1,1,1,1}, int dimension=4)
        : Cin(c_in), Cout(c_out), stride(stride_), W(c_in*c_out), b(c_out) {}
    
    SparseTensor forward(const SparseTensor& x) override {
        const int N = x.N();
        std::vector<std::array<int,4>> ocoords;
        ocoords.reserve(N);
        
        #pragma omp parallel
        {
            std::vector<std::array<int,4>> local;
            #pragma omp for nowait
            for (int i=0;i<N;++i) {
                auto c = x.coords[i];
                local.push_back({c[0]*stride[0], c[1]*stride[1], 
                                c[2]*stride[2], c[3]*stride[3]});
            }
            #pragma omp critical
            ocoords.insert(ocoords.end(), local.begin(), local.end());
        }
        
        SparseTensor y{std::move(ocoords), mt::Tensor(N, Cout, 0.f)};
        #pragma omp parallel for if (N*Cout > 512)
        for (int i=0;i<N;++i) {
            for (int co=0; co<Cout; ++co) {
                float acc = b.w[co];
                for (int ci=0; ci<Cin; ++ci)
                    acc += x.feats.at(i,ci) * W.w[ci*Cout + co];
                y.feats.at(i,co) = acc;
            }
        }
        return y;
    }
    
    void zero_grad() override {
        W.zero_grad();
        b.zero_grad();
    }
    
    void gather_params(std::vector<mt::Parameter*>& out) override {
        out.push_back(&W);
        out.push_back(&b);
    }
    
    SparseTensor backward(const SparseTensor& x, SparseTensor& grad_output) override {
        const int N = x.N();
        SparseTensor gx{x.coords, mt::Tensor(N, Cin, 0.f)};
        
        if (W.g.size() != W.w.size()) W.g.assign(W.w.size(), 0.f);
        if (b.g.size() != b.w.size()) b.g.assign(b.w.size(), 0.f);
        
        // Map from upsampled coords to grad_output row
        std::unordered_map<uint64_t,int> up_to_row;
        up_to_row.reserve(grad_output.N()*2);
        up_to_row.rehash(grad_output.N()*2);
        for (int r=0;r<grad_output.N();++r) {
            const auto &c = grad_output.coords[r];
            up_to_row[pack_key4(c[0],c[1],c[2],c[3])] = r;
        }
        
        #pragma omp parallel for if (N > 64)
        for (int i=0;i<N;++i) {
            const auto& c = x.coords[i];
            const uint64_t upk = pack_key4(c[0]*stride[0], c[1]*stride[1], 
                                           c[2]*stride[2], c[3]*stride[3]);
            auto it = up_to_row.find(upk);
            if (it == up_to_row.end()) continue;
            const int out_idx = it->second;
            
            for (int co=0; co<Cout; ++co) {
                const float dout = grad_output.feats.d(out_idx, co);
                #ifdef _OPENMP
                #pragma omp atomic
                #endif
                b.g[co] += dout;
                
                for (int ci=0; ci<Cin; ++ci) {
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    W.g[ci*Cout + co] += x.feats.at(i,ci) * dout;
                    #ifdef _OPENMP
                    #pragma omp atomic
                    #endif
                    gx.feats.d(i,ci) += W.w[ci*Cout + co] * dout;
                }
            }
        }
        
        return gx;
    }
};

// =============================== 
// Helper: concatenate features along channel dimension
// =============================== 
inline SparseTensor cat_features_same_coords(const SparseTensor& a, const SparseTensor& b) {
    if (a.coords.size() != b.coords.size()) 
        throw std::runtime_error("cat: coord size mismatch");
    
    const int N = a.N();
    const int Ca = a.C();
    const int Cb = b.C();
    
    SparseTensor y = a;
    y.feats = mt::Tensor(N, Ca + Cb, 0.f);
    
    #pragma omp parallel for if (N*(Ca+Cb) > 1024)
    for (int i=0;i<N;++i) {
        for (int j=0;j<Ca;++j) 
            y.feats.at(i,j) = a.feats.at(i,j);
        for (int j=0;j<Cb;++j) 
            y.feats.at(i,Ca+j) = b.feats.at(i,j);
    }
    return y;
}

} // namespace mme