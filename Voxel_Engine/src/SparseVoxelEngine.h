#pragma once

#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include <future>
#include <atomic>
#include <cstdint>
#include <utility>
#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <thread>
#include <limits>

#include "XYZ.h"
#include "Matrix3x3.h"
#include "Voxel.h"
#include "Camera.h"
#include "../third_party/json.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @class SparseVoxelGrid
 * @brief High-performance sparse voxel grid that only allocates memory for occupied voxels
 * 
 * Drop-in replacement for dense VoxelGrid with significant memory savings and performance improvements.
 * Maintains API compatibility while providing sparse tensor export for Minkowski Engine integration.
 */
class SparseVoxelGrid {
public:
    SparseVoxelGrid() : size(0, 0, 0), origin(0, 0, 0), sx(0), sy(0), sz(0), voxelSize(1.0f) {}

    SparseVoxelGrid(const XYZ& size, const XYZ& origin, float voxelSize = 1.0f)
        : origin(origin), voxelSize(voxelSize) {
        sx = static_cast<int>(std::ceil(size.getX() / voxelSize));
        sy = static_cast<int>(std::ceil(size.getY() / voxelSize));
        sz = static_cast<int>(std::ceil(size.getZ() / voxelSize));

        this->size = XYZ(sx * voxelSize, sy * voxelSize, sz * voxelSize);
        
        // PERFORMANCE: More aggressive pre-allocation based on camera count and expected ray density
        size_t totalVoxels = static_cast<size_t>(sx) * sy * sz;
        size_t expectedOccupancy = std::min(totalVoxels / 4, static_cast<size_t>(500000)); // Up to 25% occupancy
        sparseVoxels.reserve(expectedOccupancy);
        
        // Pre-allocate with optimal load factor for performance
        sparseVoxels.max_load_factor(0.7f);
    }

    // API compatibility methods
    inline Voxel& at(const XYZ& worldCoords) {
        auto [xi, yi, zi] = worldToIndices(worldCoords);
        return getOrCreateVoxel(xi, yi, zi);
    }

    inline const Voxel& at(const XYZ& worldCoords) const {
        auto [xi, yi, zi] = worldToIndices(worldCoords);
        return getVoxel(xi, yi, zi);
    }

    // Core sparse voxel access methods - OPTIMIZED for single hash lookup
    inline Voxel& getOrCreateVoxel(int x, int y, int z) {
        // Ignore bottom 2 rows (z < 2)
        if (z < 2) {
            return const_cast<Voxel&>(emptyVoxel);
        }
        
        size_t linearIdx = indexFromIndices(x, y, z);
        
        // PERFORMANCE: Use emplace for single hash lookup instead of find+insert
        auto [it, inserted] = sparseVoxels.emplace(linearIdx, Voxel());
        
        if (inserted) {
            // Only set position for new voxels
            it->second.setPosition(XYZ(
                origin.getX() + x * voxelSize + voxelSize * 0.5f,
                origin.getY() + y * voxelSize + voxelSize * 0.5f,
                origin.getZ() + z * voxelSize + voxelSize * 0.5f
            ));
        }
        
        return it->second;
    }
    
    inline const Voxel& getVoxel(int x, int y, int z) const {
        // Ignore bottom 2 rows (z < 2)
        if (z < 2) {
            return emptyVoxel;
        }
        
        size_t linearIdx = indexFromIndices(x, y, z);
        auto it = sparseVoxels.find(linearIdx);
        if (it != sparseVoxels.end()) {
            return it->second;
        }
        return emptyVoxel;
    }

    // Maintain exact same interface as dense VoxelGrid
    inline size_t indexFromIndices(int xi, int yi, int zi) const {
        return (static_cast<size_t>(zi) * sy + yi) * sx + xi;
    }

    inline std::tuple<int, int, int> worldToIndices(const XYZ& worldCoords) const {
        int xi = static_cast<int>(std::floor((worldCoords.getX() - origin.getX()) / voxelSize));
        int yi = static_cast<int>(std::floor((worldCoords.getY() - origin.getY()) / voxelSize));
        int zi = static_cast<int>(std::floor((worldCoords.getZ() - origin.getZ()) / voxelSize));

        // Ignore bottom 2 rows (z < 2)
        if (xi < 0 || xi >= sx || yi < 0 || yi >= sy || zi < 2 || zi >= sz) {
            throw std::out_of_range("SparseVoxelGrid::worldToIndices() - coordinates out of bounds or in ignored bottom rows");
        }
        return {xi, yi, zi};
    }

    // Getters for compatibility
    void setOrigin(const XYZ& o) { origin = o; }
    XYZ getOrigin() const { return origin; }
    XYZ getSize() const { return size; }
    int getSizeX() const { return sx; }
    int getSizeY() const { return sy; }
    int getSizeZ() const { return sz; }
    float getVoxelSize() const { return voxelSize; }
    
    // Sparse-specific functionality
    size_t getActiveVoxelCount() const { return sparseVoxels.size(); }
    size_t getMaxVoxelCount() const { return static_cast<size_t>(sx) * sy * sz; }
    float getSparsityRatio() const { 
        return 1.0f - (static_cast<float>(sparseVoxels.size()) / getMaxVoxelCount()); 
    }
    
    // Iterator support for processing only active voxels
    auto begin() { return sparseVoxels.begin(); }
    auto end() { return sparseVoxels.end(); }
    auto begin() const { return sparseVoxels.begin(); }
    auto end() const { return sparseVoxels.end(); }
    
    // PERFORMANCE: Optimized finalization for sparse grids with better cache locality
    void finalizeAllIntersections() {
        const size_t numVoxels = sparseVoxels.size();
        if (numVoxels == 0) return;
        
        // PERFORMANCE: Skip iterator vector creation for small datasets
        if (numVoxels < 1000) {
            for (auto& [idx, voxel] : sparseVoxels) {
                voxel.finalizeIntersections();
            }
            return;
        }
        
        // For larger datasets, use parallel processing with better memory access pattern
        std::vector<Voxel*> voxelPtrs;
        voxelPtrs.reserve(numVoxels);
        
        for (auto& [idx, voxel] : sparseVoxels) {
            voxelPtrs.push_back(&voxel);
        }
        
        #pragma omp parallel for schedule(static, 64)
        for (size_t i = 0; i < voxelPtrs.size(); ++i) {
            voxelPtrs[i]->finalizeIntersections();
        }
    }

    // Export functionality for Minkowski Engine
    struct SparseExport {
        std::vector<std::vector<int>> coordinates;
        std::vector<float> features;
        int spatial_dims[3];
        size_t num_points;
    };
    
    SparseExport exportSparseData() const {
        SparseExport export_data;
        export_data.spatial_dims[0] = sx;
        export_data.spatial_dims[1] = sy;
        export_data.spatial_dims[2] = sz;
        
        export_data.coordinates.reserve(sparseVoxels.size());
        export_data.features.reserve(sparseVoxels.size());
        
        for (const auto& [linearIdx, voxel] : sparseVoxels) {
            if (voxel.getIntersectionCount() > 0) {
                // Convert linear index back to 3D coordinates
                int z = static_cast<int>(linearIdx / (sx * sy));
                int y = static_cast<int>((linearIdx % (sx * sy)) / sx);
                int x = static_cast<int>(linearIdx % sx);
                
                export_data.coordinates.push_back({x, y, z});
                export_data.features.push_back(voxel.getIntersectionCount());
            }
        }
        
        export_data.num_points = export_data.coordinates.size();
        return export_data;
    }
    
    void exportToFiles(const std::string& base_filename) const {
        auto export_data = exportSparseData();
        
        // Export coordinates (x, y, z per line)
        std::ofstream coords_file(base_filename + "_coords.txt");
        for (const auto& coord : export_data.coordinates) {
            coords_file << coord[0] << " " << coord[1] << " " << coord[2] << "\n";
        }
        coords_file.close();
        
        // Export features (one per line)
        std::ofstream features_file(base_filename + "_features.txt");
        for (float feature : export_data.features) {
            features_file << feature << "\n";
        }
        features_file.close();
        
        // Export metadata
        std::ofstream meta_file(base_filename + "_meta.txt");
        meta_file << "spatial_size " << export_data.spatial_dims[0] << " " 
                  << export_data.spatial_dims[1] << " " << export_data.spatial_dims[2] << "\n";
        meta_file << "num_points " << export_data.num_points << "\n";
        meta_file << "voxel_size " << voxelSize << "\n";
        meta_file << "origin " << origin.getX() << " " << origin.getY() << " " << origin.getZ() << "\n";
        meta_file.close();
        
        std::cout << "Exported " << export_data.num_points << " sparse points to " 
                  << base_filename << "_*.txt files\n";
    }

private:
    XYZ size, origin;
    int sx, sy, sz;
    float voxelSize;
    
    // Sparse storage: only contains non-empty voxels
    std::unordered_map<size_t, Voxel> sparseVoxels;
    
    // Static empty voxel for const access to unoccupied space
    static const Voxel emptyVoxel;
};

// Static member definition
const Voxel SparseVoxelGrid::emptyVoxel = Voxel();

/**
 * @class SparseVoxelEngine
 * @brief Enhanced VoxelEngine using sparse voxel grids for improved performance and memory efficiency
 */
class SparseVoxelEngine {
public:
    // Nested structures (maintaining compatibility)
    struct LLA {
        double lat;
        double lon;
        double alt;
    };

    class Raycaster {
    public:
        Raycaster(const Camera& cam, const SparseVoxelGrid& grid)
            : camera(cam), voxelGrid(grid) {}

        // Optimized ray intersection for sparse grids
        static void calculateRayIntersectionsUltraFast(SparseVoxelGrid& voxelGrid, 
                                                      const std::vector<Camera>& cameras, 
                                                      float maxDistance = 1000.0f, 
                                                      float topPercentage = 5.0f) {
            
            std::vector<std::atomic<int>> brightnessHistogram(64);
            std::atomic<int> totalPixels(0);
            
            for (auto& bin : brightnessHistogram) {
                bin.store(0);
            }
            
            const int megaSample = 32;
            #pragma omp parallel for
            for (int camIdx = 0; camIdx < static_cast<int>(cameras.size()); ++camIdx) {
                const auto& cam = cameras[camIdx];
                int localPixels = 0;
                
                for (int y = 0; y < cam.getImageHeight(); y += megaSample) {
                    for (int x = 0; x < cam.getImageWidth(); x += megaSample) {
                        float brightness = cam.getPixelBrightness(x, y);
                        int binIndex = static_cast<int>(std::min(63.0f, brightness * 63.0f));
                        brightnessHistogram[binIndex].fetch_add(megaSample * megaSample);
                        localPixels += megaSample * megaSample;
                    }
                }
                totalPixels.fetch_add(localPixels);
            }
            
            int totalPixelCount = totalPixels.load();
            int targetPixelCount = static_cast<int>(totalPixelCount * topPercentage / 100.0f);
            float brightnessThreshold = .5f;
            int cumulativeCount = 0;
            
            for (int i = 63; i >= 0; --i) {
                cumulativeCount += brightnessHistogram[i].load();
                if (cumulativeCount >= targetPixelCount) {
                    brightnessThreshold = i / 63.0f;
                    break;
                }
            }
            
            std::cout << "Brightness threshold=" << brightnessThreshold << "\n";
            
            std::atomic<int> totalRaysProcessed(0);
            
            const int rayStep = 4;
            
            // PERFORMANCE: Use dynamic scheduling for better load balancing across cameras
            #pragma omp parallel for schedule(dynamic, 1) 
            for (int camIdx = 0; camIdx < static_cast<int>(cameras.size()); ++camIdx) {
                const auto& cam = cameras[camIdx];
                int raysThisCamera = 0;
                
                // PERFORMANCE: Larger blocks for better cache performance, adaptive block size
                const int blockSize = std::min(128, std::max(32, cam.getImageWidth() / 16));
                
                for (int by = 0; by < cam.getImageHeight(); by += blockSize) {
                    for (int bx = 0; bx < cam.getImageWidth(); bx += blockSize) {
                        int maxY = std::min(by + blockSize, cam.getImageHeight());
                        int maxX = std::min(bx + blockSize, cam.getImageWidth());
                        
                        // PERFORMANCE: Process pixels in cache-friendly order
                        for (int y = by; y < maxY; y += rayStep) {
                            for (int x = bx; x < maxX; x += rayStep) {
                                float brightness = cam.getPixelBrightness(x, y);
                                
                                if (brightness < brightnessThreshold) continue;
                                
                                Ray ray = cam.generateRay(x, y);
                                bool found = CastRayAndAccumulate(voxelGrid, ray.origin, ray.direction, maxDistance, brightness, camIdx);
                                raysThisCamera += found ? 1 : 0;
                            }
                        }
                    }
                }
                
                // PERFORMANCE: Reduce atomic contention by batching updates
                totalRaysProcessed.fetch_add(raysThisCamera, std::memory_order_relaxed);
            }
            
            std::cout << "Processed " << totalRaysProcessed.load() << " rays\n";
            
            // MAJOR PERFORMANCE IMPROVEMENT: Only process active voxels
            voxelGrid.finalizeAllIntersections();
        }

    private:
        static bool CastRayAndAccumulate(SparseVoxelGrid& grid, const XYZ& origin, const XYZ& dir, 
                                       float maxDistance, float intensity, int cameraId) {
            const float voxelSize = grid.getVoxelSize();
            const float invVoxelSize = 1.0f / voxelSize;
            bool foundIntersection = false;

            // PERFORMANCE: Cache grid properties to avoid repeated method calls
            const XYZ& gridOrigin = grid.getOrigin();
            const int sx = grid.getSizeX();
            const int sy = grid.getSizeY();
            const int sz = grid.getSizeZ();
            
            int x = static_cast<int>(std::floor((origin.getX() - gridOrigin.getX()) * invVoxelSize));
            int y = static_cast<int>(std::floor((origin.getY() - gridOrigin.getY()) * invVoxelSize));
            int z = static_cast<int>(std::floor((origin.getZ() - gridOrigin.getZ()) * invVoxelSize));

            // PERFORMANCE: Branchless step calculation
            const int stepX = (dir.getX() > 0) ? 1 : -1;
            const int stepY = (dir.getY() > 0) ? 1 : -1;
            const int stepZ = (dir.getZ() > 0) ? 1 : -1;

            const float tDeltaX = (dir.getX() != 0) ? std::abs(voxelSize / dir.getX()) : std::numeric_limits<float>::infinity();
            const float tDeltaY = (dir.getY() != 0) ? std::abs(voxelSize / dir.getY()) : std::numeric_limits<float>::infinity();
            const float tDeltaZ = (dir.getZ() != 0) ? std::abs(voxelSize / dir.getZ()) : std::numeric_limits<float>::infinity();

            float tMaxX = (dir.getX() != 0) ? ((gridOrigin.getX() + (x + (stepX > 0 ? 1 : 0)) * voxelSize) - origin.getX()) / dir.getX() : std::numeric_limits<float>::infinity();
            float tMaxY = (dir.getY() != 0) ? ((gridOrigin.getY() + (y + (stepY > 0 ? 1 : 0)) * voxelSize) - origin.getY()) / dir.getY() : std::numeric_limits<float>::infinity();
            float tMaxZ = (dir.getZ() != 0) ? ((gridOrigin.getZ() + (z + (stepZ > 0 ? 1 : 0)) * voxelSize) - origin.getZ()) / dir.getZ() : std::numeric_limits<float>::infinity();

            float traveled = 0.0f;
            
            // PERFORMANCE: Reduce bounds checking frequency
            constexpr int BOUNDS_CHECK_FREQ = 8; // Check bounds every N iterations
            int boundsCounter = 0;

            while (traveled <= maxDistance) {
                // PERFORMANCE: Check bounds less frequently for rays mostly inside grid
                if (++boundsCounter >= BOUNDS_CHECK_FREQ) {
                    // Ignore bottom 2 rows (z < 2)
                    if (x < 0 || x >= sx || y < 0 || y >= sy || z < 2 || z >= sz) break;
                    boundsCounter = 0;
                }

                // PERFORMANCE: Optimized voxel access and intersection tracking
                auto& voxel = grid.getOrCreateVoxel(x, y, z);
                
                // PERFORMANCE: Cache intersection check result to avoid repeated map lookup
                const auto& intersections = voxel.getCameraIntersections();
                const bool hasIntersections = !intersections.empty();
                const bool newCamera = hasIntersections && (intersections.find(cameraId) == intersections.end());
                
                if (newCamera) {
                    foundIntersection = true;
                }
                voxel.addCameraIntersection(cameraId, intensity);

                // PERFORMANCE: Branchless stepping with fewer comparisons
                if (tMaxX <= tMaxY && tMaxX <= tMaxZ) {
                    x += stepX;
                    traveled = tMaxX;
                    tMaxX += tDeltaX;
                } else if (tMaxY <= tMaxZ) {
                    y += stepY;
                    traveled = tMaxY;
                    tMaxY += tDeltaY;
                } else {
                    z += stepZ;
                    traveled = tMaxZ;
                    tMaxZ += tDeltaZ;
                }
            }
            return foundIntersection;
        }

        const Camera& camera;
        const SparseVoxelGrid& voxelGrid;
    };

    class Scene {
    public:
        Scene(std::vector<Camera> cams) : cameras(std::move(cams)) {}
        
        void addCameras(const std::vector<Camera>& cams) {
            cameras.insert(cameras.end(), cams.begin(), cams.end());
        }

        void calculateRayIntersections(float topPercentage = 5.0f, float maxDistance = 300.0f) {
            std::cout << "Calculating dynamic scene bounds based on cameras...\n";
            
            auto [minCorner, maxCorner] = calculateSceneBounds();
            
            XYZ size(
                std::ceil(maxCorner.getX() - minCorner.getX()),
                std::ceil(maxCorner.getY() - minCorner.getY()),
                std::ceil(maxCorner.getZ() - minCorner.getZ())
            );
            
            // Create sparse voxel grid
            sparseVoxelGrid.emplace(size, minCorner, 5.0f); // VOXEL SIZE SETTING !!!!
            
            std::cout << "Sparse voxel grid created: " << size.getX() << "x" << size.getY() << "x" << size.getZ() 
                      << " (origin: " << minCorner.getX() << "," << minCorner.getY() << "," << minCorner.getZ() << ")\n";
            
            std::cout << "Ultra-fast sparse ray intersection...\n";
            auto start = std::chrono::high_resolution_clock::now();
            
            Raycaster::calculateRayIntersectionsUltraFast(*sparseVoxelGrid, cameras, maxDistance, topPercentage);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Sparse ray intersection: " << duration.count() << "ms\n";
            
            // Report performance statistics
            float sparsity = sparseVoxelGrid->getSparsityRatio();
            size_t active = sparseVoxelGrid->getActiveVoxelCount();
            size_t total = sparseVoxelGrid->getMaxVoxelCount();
            
            std::cout << "Sparsity: " << (sparsity * 100) << "% empty (" 
                      << active << "/" << total << " voxels active)\n";
            
            // Estimate memory savings
            size_t dense_bytes = total * sizeof(Voxel);
            size_t sparse_bytes = active * (sizeof(Voxel) + sizeof(size_t));
            float savings = 1.0f - (static_cast<float>(sparse_bytes) / dense_bytes);
            std::cout << "Memory savings: " << (savings * 100) << "% ("
                      << (sparse_bytes / 1024 / 1024) << "MB vs " 
                      << (dense_bytes / 1024 / 1024) << "MB dense)\n";
        }

        void printSceneInfo() const {
            if (!sparseVoxelGrid) {
                std::cout << "No sparse voxel grid available\n";
                return;
            }
            
            std::cout << "Sparse Voxel Grid: " << sparseVoxelGrid->getSize().getX() << " x " 
                      << sparseVoxelGrid->getSize().getY() << " x " << sparseVoxelGrid->getSize().getZ() << "\n";
            
            size_t multiCameraVoxels = 0;
            for (const auto& [idx, voxel] : *sparseVoxelGrid) {
                if (voxel.getNumCamerasIntersecting() > 1) {
                    multiCameraVoxels++;
                }
            }
            
            std::cout << "Multi-camera intersections: " << multiCameraVoxels << "/" 
                      << sparseVoxelGrid->getActiveVoxelCount() << " active voxels\n";
        }

        void exportToMinkowski(const std::string& base_filename) const {
            if (!sparseVoxelGrid) {
                throw std::runtime_error("No sparse voxel grid available for export");
            }
            
            sparseVoxelGrid->exportToFiles(base_filename);
        }
        
        SparseVoxelGrid& getVoxelGrid() { 
            if (!sparseVoxelGrid) {
                throw std::runtime_error("Sparse voxel grid not initialized");
            }
            return *sparseVoxelGrid; 
        }

    private:
        std::pair<XYZ, XYZ> calculateSceneBounds() {
            if (cameras.empty()) {
                return { XYZ(-500.0f, -500.0f, 0.0f), XYZ(500.0f, 500.0f, 300.0f) };
            }
            
            float minX = std::numeric_limits<float>::max();
            float maxX = std::numeric_limits<float>::lowest();
            float minY = std::numeric_limits<float>::max();
            float maxY = std::numeric_limits<float>::lowest();
            float minZ = std::numeric_limits<float>::max();
            float maxZ = std::numeric_limits<float>::lowest();
            
            for (const auto& cam : cameras) {
                const XYZ& pos = cam.getPosition();
                
                minX = std::min(minX, pos.getX());
                maxX = std::max(maxX, pos.getX());
                minY = std::min(minY, pos.getY());
                maxY = std::max(maxY, pos.getY());
                minZ = std::min(minZ, pos.getZ());
                maxZ = std::max(maxZ, pos.getZ());
                
                // Calculate frustum corners for bounds expansion
                float fovRad = cam.getFOV() * 3.14159265359f / 180.0f;
                float viewDistance = 100.0f;
                
                std::vector<std::pair<int, int>> corners = {
                    {0, 0}, {cam.getImageWidth()-1, 0}, 
                    {0, cam.getImageHeight()-1}, {cam.getImageWidth()-1, cam.getImageHeight()-1}
                };
                
                for (const auto& [px, py] : corners) {
                    Ray ray = cam.generateRay(px, py);
                    XYZ endPoint = XYZ(
                        ray.origin.getX() + ray.direction.getX() * viewDistance,
                        ray.origin.getY() + ray.direction.getY() * viewDistance,
                        ray.origin.getZ() + ray.direction.getZ() * viewDistance
                    );
                    
                    minX = std::min(minX, endPoint.getX());
                    maxX = std::max(maxX, endPoint.getX());
                    minY = std::min(minY, endPoint.getY());
                    maxY = std::max(maxY, endPoint.getY());
                    minZ = std::min(minZ, endPoint.getZ());
                    maxZ = std::max(maxZ, endPoint.getZ());
                }
            }
            
            float padding = 50.0f;
            minX -= padding; maxX += padding;
            minY -= padding; maxY += padding;
            minZ = std::max(0.0f, minZ - padding);
            maxZ += padding;
            
            return { XYZ(minX, minY, minZ), XYZ(maxX, maxY, maxZ) };
        }

        std::vector<Camera> cameras;
        std::optional<SparseVoxelGrid> sparseVoxelGrid;
    };

public:
    SparseVoxelEngine() = default;
};