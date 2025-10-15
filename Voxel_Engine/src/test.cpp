#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "InferenceBridge.hpp"

// -----------------------------------------------------------
// SparseVoxelGrid mockup (replace this with your real structure)
// -----------------------------------------------------------
struct SparseVoxelGrid {
    std::vector<int32_t> coords;  // [N, 4] -> x, y, z, t
    std::vector<float> features;  // [N, F] -> feature vector per voxel
    size_t ncoords;
    size_t nfeats;
};

// -----------------------------------------------------------
// Generate a dummy sparse voxel grid for testing
// -----------------------------------------------------------
SparseVoxelGrid generateTestGrid(size_t npoints = 1024, size_t nfeats = 7) {
    SparseVoxelGrid grid;
    grid.ncoords = npoints;
    grid.nfeats = nfeats;
    grid.coords.resize(npoints * 4);
    grid.features.resize(npoints * nfeats);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> coord_dist(0, 31);
    std::uniform_real_distribution<float> feat_dist(0.0f, 1.0f);

    for (size_t i = 0; i < npoints; ++i) {
        grid.coords[i * 4 + 0] = coord_dist(rng);
        grid.coords[i * 4 + 1] = coord_dist(rng);
        grid.coords[i * 4 + 2] = coord_dist(rng);
        grid.coords[i * 4 + 3] = coord_dist(rng); // time/frame index
        for (size_t j = 0; j < nfeats; ++j) {
            grid.features[i * nfeats + j] = feat_dist(rng);
        }
    }
    return grid;
}

// -----------------------------------------------------------
// Main program
// -----------------------------------------------------------
int main() {
    try {
        std::cout << "[C++] Starting inference bridge demo..." << std::endl;

        // Initialize inference bridge
        InferenceBridge bridge("tcp://127.0.0.1:5555");
        bridge.connect();

        // Create a dummy sparse voxel grid
        SparseVoxelGrid svg = generateTestGrid(2048, 7);
        std::cout << "[C++] Generated test grid: " << svg.ncoords << " voxels, "
                  << svg.nfeats << " features each" << std::endl;

        // Perform inference and time it
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> output = bridge.infer(svg.coords, svg.features, svg.ncoords, svg.nfeats);
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();

        // Report
        std::cout << "[C++] Inference complete. Output size: " << output.size()
                  << " values (" << elapsed_ms << " ms)" << std::endl;

        // Print first few values
        for (size_t i = 0; i < std::min<size_t>(10, output.size()); ++i)
            std::cout << output[i] << " ";
        std::cout << std::endl;

        std::cout << "[C++] Done." << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "[C++ ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
