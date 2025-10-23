#pragma once
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <stdexcept>
#include <algorithm>
#include <iostream>

class SparseVoxelEngine {
public:
    struct Settings {
        float cx = 0.0f;                 // image center x
        float cy = 0.0f;                 // image center y
        float xOffsetRatio = 0.02f;      // horizontal offset (fraction of width)
        float r_fisheye = 1.0f;          // radius of fisheye circle (pixels)
        float focal_px = 1.0f;           // focal length in pixels
        float voxelScale = 1.0f;         // world scale per voxel (meters per voxel, etc.)
        float intensityThreshold = 1.0f; // min intensity (0–255) to project
        float maxDistance = 100.0f;       // maximum projection distance in world units
    };

    struct VoxelKey {
        int x, y, z;
        bool operator==(const VoxelKey& o) const noexcept {
            return x == o.x && y == o.y && z == o.z;
        }
    };
    
    struct VoxelHash {
        size_t operator()(const VoxelKey& k) const noexcept {
            return ((k.x * 73856093) ^ (k.y * 19349663) ^ (k.z * 83492791));
        }
    };

    using Grid = std::unordered_map<VoxelKey, uint8_t, VoxelHash>;

    SparseVoxelEngine(int width, int height, const Settings& settings)
        : width(width), height(height), settings(settings)
    {
        if (width <= 0 || height <= 0)
            throw std::invalid_argument("Invalid image dimensions");
        if (settings.r_fisheye <= 0.0f)
            throw std::invalid_argument("r_fisheye must be > 0");
        if (settings.voxelScale <= 0.0f)
            throw std::invalid_argument("voxelScale must be > 0");
        if (settings.maxDistance <= 0.0f)
            throw std::invalid_argument("maxDistance must be > 0");

        // Apply X-axis offset (shift fisheye center right)
        this->cx = settings.cx + width * settings.xOffsetRatio;
        this->cy = settings.cy;
    }

    void projectMask(const std::vector<uint8_t>& mask) {
        if (mask.size() != static_cast<size_t>(width * height))
            throw std::runtime_error("Mask size mismatch");

        const float rf = settings.r_fisheye;
        const float voxelScale = settings.voxelScale;
        const float maxDistance = settings.maxDistance;
        const float intensityThreshold = settings.intensityThreshold;

        // Convert distance limit to voxel steps
        const int maxSteps = std::max(1, static_cast<int>(std::floor(maxDistance / voxelScale)));

        #pragma omp parallel
        {
            std::unordered_map<VoxelKey, uint8_t, VoxelHash> localGrid;

            #pragma omp for schedule(static)
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    uint8_t val = mask[y * width + x];
                    if (val < intensityThreshold)
                        continue;

                    float dx = x - cx;
                    float dy = y - cy;
                    float r = std::sqrt(dx * dx + dy * dy);
                    if (r > rf)
                        continue;  // outside fisheye region

                    // Normalized radius
                    float r_norm = r / rf;
                    float theta = 2.0f * std::asin(r_norm / 2.0f); // equisolid
                    float phi = std::atan2(dy, dx);

                    // Unit direction vector
                    float sx = std::sin(theta) * std::cos(phi);
                    float sy = std::sin(theta) * std::sin(phi);
                    float sz = std::cos(theta);

                    // March ray through voxel space up to maxDistance
                    for (int s = 0; s < maxSteps; ++s) {
                        float t = (s + 1) * voxelScale;
                        if (t > maxDistance)
                            break;

                        VoxelKey vk {
                            static_cast<int>(std::round(sx * t)),
                            static_cast<int>(std::round(sy * t)),
                            static_cast<int>(std::round(sz * t))
                        };
                        localGrid[vk] = std::min<uint8_t>(255, val);
                    }
                }
            }

            #pragma omp critical
            grid.insert(localGrid.begin(), localGrid.end());
        }
    }

    const Grid& getGrid() const noexcept { return grid; }

    void clear() noexcept { grid.clear(); }

    void debugPrintStats() const {
        std::cout << "SparseVoxelEngine: " << grid.size()
                  << " voxels populated.\n";
    }

private:
    int width;
    int height;
    float cx;
    float cy;
    Settings settings;
    Grid grid;
};
