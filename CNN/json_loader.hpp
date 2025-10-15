// json_loader.hpp
#pragma once
#include <filesystem>
#include <fstream>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <iostream>
#include "third_party/json.hpp"
#include "mini_numpy.hpp"
#include "mini_torch.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---- Frame schema ----
struct GridInfo {
    double voxel_size_m = 1.0;
    std::array<double,3> origin_m{0.0,0.0,0.0};
    std::array<int,3> dimensions{0,0,0};
};
struct VoxelRec { std::array<int,3> coord{0,0,0}; float intensity=0.f; int motion_type=0; };
struct TargetRec { int frame=0; std::array<double,3> position_m{0,0,0}; };
struct Frame { GridInfo grid; std::vector<VoxelRec> voxels; std::vector<TargetRec> targets; };

inline json read_json(const fs::path& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Failed to open: " + path.string());
    json j; f >> j;
    return j;
}

inline Frame parse_frame_json(const fs::path& path) {
    json j = read_json(path);
    Frame f;

    // grid_info
    if (j.contains("grid_info")) {
        auto& g = j["grid_info"];
        if (g.contains("voxel_size_m")) f.grid.voxel_size_m = g["voxel_size_m"].get<double>();
        if (g.contains("origin_m")) {
            for (int i=0;i<3;++i) f.grid.origin_m[i] = g["origin_m"][i].get<double>();
        }
        if (g.contains("dimensions")) {
            for (int i=0;i<3;++i) f.grid.dimensions[i] = g["dimensions"][i].get<int>();
        }
    }

    // voxels
    if (j.contains("voxels") && j["voxels"].is_array()) {
        for (auto& vx : j["voxels"]) {
            VoxelRec v;
            for (int i=0;i<3;++i) v.coord[i] = vx["coordinates"][i].get<int>();
            v.intensity   = vx["intensity"].get<float>();
            v.motion_type = vx["motion_type"].get<int>();
            f.voxels.push_back(v);
        }
    }

    // targets
    if (j.contains("targets") && j["targets"].is_array()) {
        for (auto& tx : j["targets"]) {
            TargetRec t;
            t.frame = tx["frame"].get<int>();
            for (int i=0;i<3;++i) t.position_m[i] = tx["position_m"][i].get<double>();
            f.targets.push_back(t);
        }
    }

    return f;
}

// segments finder
inline std::vector<std::array<fs::path,10>> find_segments(const fs::path& dir) {
    std::vector<fs::path> files;
    for (auto& p : fs::directory_iterator(dir))
        if (p.is_regular_file() && p.path().extension() == ".json")
            files.push_back(p.path());

    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b){
        return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
    });

    std::vector<std::array<fs::path,10>> segs;
    for (size_t i=0; i+9<files.size(); ++i) {
        int s = std::stoi(files[i].stem().string());
        int e = std::stoi(files[i+9].stem().string());
        if (e - s == 9) {
            std::array<fs::path,10> w{};
            for (int j=0;j<10;++j) w[j]=files[i+j];
            segs.push_back(w);
        }
    }
    return segs;
}

// dataset sample
struct Sample {
    std::vector<std::array<int,4>> coords;
    mt::Tensor feats;
    mt::Tensor labels;
    GridInfo grid;
};

inline Sample build_sample_from_segment(const std::array<fs::path,10>& seg) {
    std::array<Frame,10> frames;
    for (int i=0;i<10;++i) frames[i] = parse_frame_json(seg[i]);
    const auto& g0 = frames[0].grid;

    std::vector<float> intens; intens.reserve(1024);
    size_t N=0;
    for (int t=0;t<10;++t) { N += frames[t].voxels.size(); for (auto& v: frames[t].voxels) intens.push_back(v.intensity); }
    if (N==0) throw std::runtime_error("Empty voxels in segment");

    float mu = mini_np::mean(intens);
    float sd = mini_np::stddev(intens, mu) + 1e-6f;

    std::map<int, std::array<int,3>> target_by_t;
    for (int t=0;t<10;++t) {
        if (frames[t].targets.empty()) continue;
        TargetRec sel = frames[t].targets[0];
        for (auto& tr : frames[t].targets) if (tr.frame==t) { sel = tr; break; }
        target_by_t[t] = mini_np::meters_to_grid(sel.position_m, g0.origin_m, g0.voxel_size_m);
    }

    Sample s;
    s.grid = g0;
    s.coords.reserve(N);
    s.feats = mt::Tensor((int)N, 7, 0.f);
    s.labels = mt::Tensor((int)N, 1, 0.f);

    int row=0;
    for (int t=0;t<10;++t) {
        for (auto& v : frames[t].voxels) {
            s.coords.push_back({v.coord[0], v.coord[1], v.coord[2], t});
            auto oh = mini_np::one_hot_motion(v.motion_type);
            float inorm = (v.intensity - mu) / sd;
            s.feats.at(row,0)=inorm;
            for (int k=0;k<5;++k) s.feats.at(row,1+k)=oh[k];
            s.feats.at(row,6)=float(t)/9.f;
            ++row;
        }
    }

    int radius = 2;
    int r2 = radius*radius;
    for (int i=0;i<(int)s.coords.size(); ++i) {
        int t = s.coords[i][3];
        auto it = target_by_t.find(t);
        if (it==target_by_t.end()) continue;
        auto ctr = it->second;
        int dx = s.coords[i][0]-ctr[0];
        int dy = s.coords[i][1]-ctr[1];
        int dz = s.coords[i][2]-ctr[2];
        int d2 = dx*dx + dy*dy + dz*dz;
        s.labels.at(i,0) = (d2 <= r2) ? 1.f : 0.f;
    }
    return s;
}

