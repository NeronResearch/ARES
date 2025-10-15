#pragma once
#include <vector>
#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace mini_np {

inline float mean(const std::vector<float>& v) {
    if (v.empty()) return 0.f;
    float s = std::accumulate(v.begin(), v.end(), 0.f);
    return s / float(v.size());
}

inline float stddev(const std::vector<float>& v, float mu) {
    if (v.size() <= 1) return 0.f;
    double acc = 0.0;
    for (float x : v) { double d = double(x) - double(mu); acc += d*d; }
    return float(std::sqrt(acc / double(v.size())));
}

inline int floor_div(int a, int b) {
    int q = a / b;
    int r = a % b;
    if ((r != 0) && ((r > 0) != (b > 0))) --q;
    return q;
}

inline std::array<float,5> one_hot_motion(int mt) {
    int idx = 0;
    switch (mt) { case -1: idx=0; break; case 0: idx=1; break; case 1: idx=2; break; case 2: idx=3; break; case 3: idx=4; break; default: idx=0; }
    std::array<float,5> v{0,0,0,0,0}; v[idx] = 1.f; return v;
}

inline std::array<int,3> meters_to_grid(const std::array<double,3>& pos_m,
                                        const std::array<double,3>& origin_m,
                                        double voxel) {
    std::array<int,3> out{};
    for (int i=0;i<3;++i) {
        double g = std::floor((pos_m[i] - origin_m[i]) / voxel);
        out[i] = int(g);
    }
    return out;
}

}
