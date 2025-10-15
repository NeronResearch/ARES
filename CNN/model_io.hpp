#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include "mini_torch.hpp"

// Simple binary serializer for mt::Parameter vectors.
// Format:
// 4-byte magic 'MIO1' (0x4D494F31), 4-byte uint32_t: param_count
// For each parameter:
//   uint64_t: size (number of floats)
//   raw float[size] little-endian

inline bool save_parameters_binary(const std::string& path, const std::vector<mt::Parameter*>& params) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    // magic
    uint32_t magic = 0x4D494F31u;
    f.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    uint32_t cnt = (uint32_t)params.size();
    f.write(reinterpret_cast<const char*>(&cnt), sizeof(cnt));
    for (auto* p : params) {
        uint64_t sz = p ? (uint64_t)p->w.size() : 0ull;
        f.write(reinterpret_cast<const char*>(&sz), sizeof(sz));
        if (sz) f.write(reinterpret_cast<const char*>(p->w.data()), sizeof(float) * (size_t)sz);
    }
    return f.good();
}

inline bool load_parameters_binary(const std::string& path, const std::vector<mt::Parameter*>& params) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open file\n"; return false; }

    uint32_t magic = 0;
    f.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x4D494F31u) { std::cerr << "Bad magic: " << std::hex << magic << std::dec << "\n"; return false; }

    uint32_t cnt = 0;
    f.read(reinterpret_cast<char*>(&cnt), sizeof(cnt));
    if (cnt != params.size()) {
        std::cerr << "Param count mismatch. File=" << cnt << " Expected=" << params.size() << "\n";
        return false;
    }

    for (uint32_t i=0;i<cnt;++i) {
        uint64_t sz = 0;
        f.read(reinterpret_cast<char*>(&sz), sizeof(sz));
        if (!f) { std::cerr << "Read error at param " << i << "\n"; return false; }
        if (sz != params[i]->w.size()) {
            std::cerr << "Size mismatch at param " << i 
                      << " File=" << sz << " Expected=" << params[i]->w.size() << "\n";
            return false;
        }
        if (sz) f.read(reinterpret_cast<char*>(params[i]->w.data()), sizeof(float) * (size_t)sz);
    }
    return f.good();
}

