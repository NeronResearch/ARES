#include <iostream>
#include <vector>
#include <fstream>
#include "PixelMotion.hpp"
#include "SparseVoxelEngine.hpp"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"
#include "../third_party/json.hpp"

using json = nlohmann::json;
using namespace std;
using namespace std::chrono;

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

int main() {
    int w1, h1, c1, w2, h2, c2;

    // Load input images
    unsigned char* img1 = stbi_load("data/3_0200.jpg", &w1, &h1, &c1, 0);
    unsigned char* img2 = stbi_load("data/3_0203.jpg", &w2, &h2, &c2, 0);
    if (!img1 || !img2) {
        std::cerr << "Error: could not load one or both images.\n";
        return 1;
    }

    // Convert to byte vectors
    std::vector<uint8_t> v1(img1, img1 + w1 * h1 * c1);
    std::vector<uint8_t> v2(img2, img2 + w2 * h2 * c2);
    stbi_image_free(img1);
    stbi_image_free(img2);

    // Compute motion map
    PixelMotion::Settings motionSettings;
    motionSettings.motionThreshold = 15.0f;

    PixelMotion motion(w1, h1, motionSettings);
    std::vector<uint8_t> mask = {0};
    { Timer timer("Compute motion map");
      mask = motion.computeMotion(v1, v2);
    }
    // Save motion image for visual confirmation
    stbi_write_png("out/motion.png", w1, h1, c1, mask.data(), w1 * c1);
    std::cout << "Wrote out/motion.png\n";

    // Setup SparseVoxelEngine
    SparseVoxelEngine::Settings voxelSettings;
    voxelSettings.cx = w1 / 2.0f;
    voxelSettings.cy = h1 / 2.0f;
    voxelSettings.r_fisheye = w1 * 0.66f;
    voxelSettings.focal_px = voxelSettings.r_fisheye / std::sqrt(2.0f);
    voxelSettings.voxelScale = 10.0f;
    voxelSettings.intensityThreshold = 5.0f;
    voxelSettings.maxDistance = 300.0f;

    SparseVoxelEngine voxelEngine(w1, h1, voxelSettings);

    // Convert motion mask to grayscale if multi-channel
    std::vector<uint8_t> grayMask;
    grayMask.reserve(w1 * h1);
    if (c1 == 1) {
        grayMask = mask;
    } else {
        for (size_t i = 0; i < mask.size(); i += c1) {
            uint8_t r = mask[i + 0];
            uint8_t g = (c1 > 1) ? mask[i + 1] : r;
            uint8_t b = (c1 > 2) ? mask[i + 2] : r;
            uint8_t lum = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
            grayMask.push_back(lum);
        }
    }

    {
        Timer timer("Sparse voxel projection");
        voxelEngine.projectMask(grayMask);
    }

    voxelEngine.debugPrintStats();

    // Compute bounds for debugging
    int minx = 1e9, maxx = -1e9, miny = 1e9, maxy = -1e9, minz = 1e9, maxz = -1e9;
    for (auto& kv : voxelEngine.getGrid()) {
        minx = std::min(minx, kv.first.x);
        maxx = std::max(maxx, kv.first.x);
        miny = std::min(miny, kv.first.y);
        maxy = std::max(maxy, kv.first.y);
        minz = std::min(minz, kv.first.z);
        maxz = std::max(maxz, kv.first.z);
    }
    std::cout << "Voxel bounds: x[" << minx << "," << maxx
              << "] y[" << miny << "," << maxy
              << "] z[" << minz << "," << maxz << "]\n";

    // Serialize voxels to JSON
    json voxelJson = json::array();
    for (const auto& kv : voxelEngine.getGrid()) {
        const auto& k = kv.first;
        uint8_t intensity = kv.second;
        voxelJson.push_back({
            {"x", k.x},
            {"y", k.y},
            {"z", k.z},
            {"intensity", intensity}
        });
    }

    // Save JSON
    std::ofstream outFile("out/voxels.json");
    outFile << voxelJson.dump(2);
    outFile.close();

    std::cout << "Wrote out/voxels.json (" << voxelJson.size() << " voxels)\n";
    std::cout << "Sparse voxel projection complete.\n";
    return 0;
}
