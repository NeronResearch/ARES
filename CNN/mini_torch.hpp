#pragma once
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <unordered_map>

namespace mt {

struct Tensor {
    int n = 0, c = 0;
    std::vector<float> data;
    std::vector<float> grad;
    
    Tensor() = default;
    Tensor(int n_, int c_, float init=0.f): n(n_), c(c_), data(n_*c_, init), grad(n_*c_, 0.f) {}
    
    inline float& at(int i, int j) { return data[i*c + j]; }
    inline const float& at(int i, int j) const { return data[i*c + j]; }
    inline float& d(int i, int j) { return grad[i*c + j]; }
    
    void zero_grad() { std::fill(grad.begin(), grad.end(), 0.f); }
};

struct Parameter {
    std::vector<float> w;
    std::vector<float> g;
    
    Parameter() = default;
    
    // Kaiming uniform initialization (matches PyTorch default for Conv layers)
    Parameter(int sz, float scale=1.0f) : w(sz), g(sz, 0.f) {
        if (sz == 0) return;
        
        // Use random device for non-deterministic seed (critical for training!)
        std::random_device rd;
        std::mt19937 rng(rd());
        
        // Kaiming uniform: U(-bound, bound) where bound = sqrt(3 / fan_in) * scale
        // For conv weights [K*Cin*Cout], fan_in ≈ K*Cin
        // For bias or 1D params, use simple uniform
        float bound = std::sqrt(3.0f / std::max(1.0f, float(sz))) * scale;
        std::uniform_real_distribution<float> dist(-bound, bound);
        
        for (auto& x : w) x = dist(rng);
    }
    
    void zero_grad() { std::fill(g.begin(), g.end(), 0.f); }
};

// BCE with logits, per-point pos_weight on positives
inline float bce_with_logits(Tensor& logits, const Tensor& labels, float pos_weight) {
    float loss = 0.f;
    const float eps = 1e-7f; // Numerical stability
    
    for (int i=0;i<logits.n;++i) {
        float z = logits.at(i,0);
        float y = labels.at(i,0);
        
        // Numerically stable BCE computation
        // loss = -[w*y*log(sigmoid(z)) + (1-y)*log(1-sigmoid(z))]
        // Using log-sum-exp trick for stability
        float max_val = std::max(0.0f, -z);
        float log_sigmoid = -(max_val + std::log(std::exp(-max_val) + std::exp(-z - max_val)));
        float log_1minus_sigmoid = -(max_val + std::log(std::exp(-max_val) + std::exp(z - max_val)));
        
        if (y >= 0.5f) {
            loss += -pos_weight * log_sigmoid;
        } else {
            loss += -log_1minus_sigmoid;
        }
        
        // Gradient: dL/dz = w*(sigmoid(z) - 1) for y=1, sigmoid(z) for y=0
        float p = 1.0f / (1.0f + std::exp(-z));
        float dz = (y >= 0.5f) ? pos_weight * (p - 1.0f) : p;
        logits.d(i,0) += dz;
    }
    
    return loss / float(std::max(1, logits.n));
}

// AdamW optimizer
struct AdamW {
    float lr, wd, b1, b2, eps;
    std::unordered_map<void*, std::vector<float>> m, v;
    
    AdamW(float lr_=1e-3f, float wd_=1e-4f, float b1_=0.9f, float b2_=0.999f, float eps_=1e-8f)
        : lr(lr_), wd(wd_), b1(b1_), b2(b2_), eps(eps_) {}
    
    void step(std::vector<Parameter*>& params, int t) {
        for (auto* p : params) {
            if (p->w.empty()) continue;
            if (p->g.empty()) p->g.resize(p->w.size(), 0.f);
            
            // AdamW: apply weight decay directly to weights (not to gradient)
            // This is the key difference from Adam+L2
            for (size_t i=0;i<p->w.size();++i) {
                p->w[i] *= (1.0f - lr * wd); // Weight decay
            }
            
            auto& mm = m[p];
            auto& vv = v[p];
            if (mm.size() != p->w.size()) mm.assign(p->w.size(), 0.f);
            if (vv.size() != p->w.size()) vv.assign(p->w.size(), 0.f);
            
            float bias_correction1 = 1.0f - std::pow(b1, float(t));
            float bias_correction2 = 1.0f - std::pow(b2, float(t));
            
            for (size_t i=0;i<p->w.size();++i) {
                // Update biased first moment estimate
                mm[i] = b1 * mm[i] + (1.0f - b1) * p->g[i];
                // Update biased second raw moment estimate
                vv[i] = b2 * vv[i] + (1.0f - b2) * p->g[i] * p->g[i];
                
                // Compute bias-corrected estimates
                float mhat = mm[i] / bias_correction1;
                float vhat = vv[i] / bias_correction2;
                
                // Update parameters
                p->w[i] -= lr * mhat / (std::sqrt(vhat) + eps);
            }
        }
    }
};

// Simple SGD optimizer (in-place): params' g must be populated prior to call
inline void sgd_step(std::vector<Parameter*>& params, float lr) {
    for (auto* p : params) {
        if (p->g.empty()) continue;
        for (size_t i=0;i<p->w.size();++i) p->w[i] -= lr * p->g[i];
    }
}

} // namespace mt