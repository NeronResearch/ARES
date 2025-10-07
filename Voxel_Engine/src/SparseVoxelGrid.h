#pragma once
#include "XYZ.h"
#include "Voxel.h"
#include <unordered_map>
#include <vector>
#include <cmath>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @class SparseVoxelGrid
 * @brief Drop-in replacement for dense VoxelGrid using sparse tensor representation
 * 
 * Maintains same API as VoxelGrid but stores only non-empty voxels internally.
 * Compatible with existing ray casting and processing algorithms.
 */
class SparseVoxelGrid {
public:
    // Maintain same constructor signature for compatibility
    SparseVoxelGrid(const XYZ& size, const XYZ& origin, float voxelSize = 1.0f)
        : origin(origin), voxelSize(voxelSize) {
        sx = static_cast<int>(std::ceil(size.getX() / voxelSize));
        sy = static_cast<int>(std::ceil(size.getY() / voxelSize));
        sz = static_cast<int>(std::ceil(size.getZ() / voxelSize));
        
        this->size = XYZ(sx * voxelSize, sy * voxelSize, sz * voxelSize);
        
        // Pre-allocate hash map with expected occupancy (much smaller than dense)
        sparseVoxels.reserve(std::min(100000, sx * sy * sz / 10));
    }
    
    // Sparse tensor coordinate structure
    struct SparseCoord {
        int x, y, z;
        size_t linearIndex;
        
        SparseCoord(int x_, int y_, int z_, int sx_, int sy_) 
            : x(x_), y(y_), z(z_), linearIndex((static_cast<size_t>(z_) * sy_ + y_) * sx_ + x_) {}
    };
    
    // API compatibility with dense VoxelGrid
    inline Voxel& at(const XYZ& worldCoords) {
        auto [xi, yi, zi] = worldToIndices(worldCoords);
        return getOrCreateVoxel(xi, yi, zi);
    }
    
    inline const Voxel& at(const XYZ& worldCoords) const {
        auto [xi, yi, zi] = worldToIndices(worldCoords);
        return getVoxel(xi, yi, zi);
    }
    
    // Direct access by indices (maintains existing ray casting compatibility)
    inline Voxel& getOrCreateVoxel(int x, int y, int z) {
        size_t linearIdx = indexFromIndices(x, y, z);
        
        auto it = sparseVoxels.find(linearIdx);
        if (it != sparseVoxels.end()) {
            return it->second;
        }
        
        // Create new voxel on-demand
        Voxel& voxel = sparseVoxels[linearIdx];
        voxel.setPosition(XYZ(
            origin.getX() + x * voxelSize + voxelSize / 2.0f,
            origin.getY() + y * voxelSize + voxelSize / 2.0f,
            origin.getZ() + z * voxelSize + voxelSize / 2.0f
        ));
        return voxel;
    }
    
    inline const Voxel& getVoxel(int x, int y, int z) const {
        size_t linearIdx = indexFromIndices(x, y, z);
        auto it = sparseVoxels.find(linearIdx);
        if (it != sparseVoxels.end()) {
            return it->second;
        }
        return emptyVoxel; // Return empty voxel for unoccupied space
    }
    
    // Maintains exact same interface as dense version
    inline size_t indexFromIndices(int xi, int yi, int zi) const {
        return (static_cast<size_t>(zi) * sy + yi) * sx + xi;
    }
    
    inline std::tuple<int, int, int> worldToIndices(const XYZ& worldCoords) const {
        int xi = static_cast<int>(std::floor((worldCoords.getX() - origin.getX()) / voxelSize));
        int yi = static_cast<int>(std::floor((worldCoords.getY() - origin.getY()) / voxelSize));
        int zi = static_cast<int>(std::floor((worldCoords.getZ() - origin.getZ()) / voxelSize));
        
        if (xi < 0 || xi >= sx || yi < 0 || yi >= sy || zi < 0 || zi >= sz) {
            throw std::out_of_range("SparseVoxelGrid::worldToIndices() - coordinates out of bounds");
        }
        return {xi, yi, zi};
    }
    
    // Getters maintain compatibility
    XYZ getOrigin() const { return origin; }
    XYZ getSize() const { return size; }
    int getSizeX() const { return sx; }
    int getSizeY() const { return sy; }
    int getSizeZ() const { return sz; }
    float getVoxelSize() const { return voxelSize; }
    
    // New sparse-specific methods
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
    
    // Batch processing for finalization (replaces dense array iteration)
    void finalizeAllIntersections() {
        #pragma omp parallel for if(sparseVoxels.size() > 1000)
        for (auto it = sparseVoxels.begin(); it != sparseVoxels.end(); ++it) {
            it->second.finalizeIntersections();
        }
    }
    
    // Export sparse coordinates for Minkowski Engine integration
    std::vector<std::vector<int>> getSparseCoordinates() const {
        std::vector<std::vector<int>> coords;
        coords.reserve(sparseVoxels.size());
        
        for (const auto& [linearIdx, voxel] : sparseVoxels) {
            if (voxel.getIntersectionCount() > 0) {
                int z = static_cast<int>(linearIdx / (sx * sy));
                int y = static_cast<int>((linearIdx % (sx * sy)) / sx);
                int x = static_cast<int>(linearIdx % sx);
                coords.push_back({x, y, z});
            }
        }
        return coords;
    }
    
    // Export features for neural networks
    std::vector<float> getSparseFeatures() const {
        std::vector<float> features;
        features.reserve(sparseVoxels.size());
        
        for (const auto& [linearIdx, voxel] : sparseVoxels) {
            if (voxel.getIntersectionCount() > 0) {
                features.push_back(voxel.getIntersectionCount());
            }
        }
        return features;
    }

private:
    XYZ size, origin;
    int sx, sy, sz;
    float voxelSize;
    
    // Sparse storage: only stores non-empty voxels
    std::unordered_map<size_t, Voxel> sparseVoxels;
    
    // Static empty voxel for const access to unoccupied space
    static const Voxel emptyVoxel;
};

// Definition of static empty voxel
const Voxel SparseVoxelGrid::emptyVoxel = Voxel();