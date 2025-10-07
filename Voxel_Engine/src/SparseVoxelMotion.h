#pragma once

#include "VoxelMotion.h"
#include "SparseVoxelEngine.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#else
// Fallback for non-OpenMP builds
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif

// Forward declaration
using VoxelData = VoxelMotionEngine::VoxelData;
using ChangeVoxel = VoxelMotionEngine::ChangeVoxel;

// Sparse extensions for VoxelMotionEngine
namespace SparseVoxelMotionExt {

/**
 * @class SparseVoxelGrid
 * @brief Adapter to make VoxelMotion work with sparse voxel grids
 * 
 * This adapter allows existing VoxelMotion algorithms to work seamlessly
 * with sparse voxel data while maintaining performance optimizations.
 */
class SparseVoxelGridAdapter {
public:
    /**
     * @brief Constructor from SparseVoxelEngine grid
     * @param engineGrid The sparse voxel grid from SparseVoxelEngine
     */
    SparseVoxelGridAdapter(const ::SparseVoxelGrid& engineGrid) {
        // PERFORMANCE: More efficient conversion with pre-calculated values
        const size_t activeCount = engineGrid.getActiveVoxelCount();
        voxels.reserve(activeCount);
        
        // Cache grid dimensions for faster coordinate conversion
        const int sizeX = engineGrid.getSizeX();
        const int sizeY = engineGrid.getSizeY();
        const int sizeXY = sizeX * sizeY;
        
        for (const auto& [linearIdx, voxel] : engineGrid) {
            if (voxel.getIntersectionCount() > 0) {
                // PERFORMANCE: Optimized coordinate conversion with fewer divisions
                const int z = static_cast<int>(linearIdx / sizeXY);
                const int remainder = linearIdx % sizeXY;
                const int y = static_cast<int>(remainder / sizeX);
                const int x = static_cast<int>(remainder % sizeX);
                
                voxels.emplace_back(
                    voxel.getPosition(),
                    voxel.getIntersectionCount(),
                    voxel.getNumCamerasIntersecting(),
                    x, y, z
                );
            }
        }
        
        // PERFORMANCE: Build spatial index with optimal capacity
        buildSpatialIndex();
        
        // Store grid metadata
        gridDimensions = {sizeX, sizeY, engineGrid.getSizeZ()};
        gridOrigin = engineGrid.getOrigin();
        voxelSize = engineGrid.getVoxelSize();
    }

    /**
     * @brief Get voxels vector for compatibility with existing VoxelMotion code
     */
    const std::vector<VoxelMotionEngine::VoxelData>& getVoxels() const { return voxels; }
    
    /**
     * @brief Find voxel by grid coordinates (sparse-optimized)
     */
    const VoxelMotionEngine::VoxelData* findVoxelByIndices(int x, int y, int z) const {
        uint64_t hash = spatialHash(x, y, z);
        auto it = spatialIndex.find(hash);
        return (it != spatialIndex.end()) ? &voxels[it->second] : nullptr;
    }
    
    /**
     * @brief Check if coordinates are within grid bounds
     */
    bool isInBounds(int x, int y, int z) const {
        return x >= 0 && x < gridDimensions[0] &&
               y >= 0 && y < gridDimensions[1] &&
               z >= 0 && z < gridDimensions[2];
    }
    
    /**
     * @brief Get grid metadata
     */
    std::array<int, 3> getDimensions() const { return gridDimensions; }
    XYZ getOrigin() const { return gridOrigin; }
    float getVoxelSize() const { return voxelSize; }
    
    /**
     * @brief Get performance statistics
     */
    size_t getActiveVoxelCount() const { return voxels.size(); }
    size_t getTotalVoxelCount() const { 
        return static_cast<size_t>(gridDimensions[0]) * gridDimensions[1] * gridDimensions[2]; 
    }
    float getSparsityRatio() const {
        return 1.0f - (static_cast<float>(voxels.size()) / getTotalVoxelCount());
    }

private:
    std::vector<VoxelMotionEngine::VoxelData> voxels;
    std::unordered_map<uint64_t, size_t> spatialIndex;
    std::array<int, 3> gridDimensions;
    XYZ gridOrigin;
    float voxelSize;
    
    /**
     * @brief Build spatial index for fast lookups - PERFORMANCE OPTIMIZED
     */
    void buildSpatialIndex() {
        spatialIndex.clear();
        
        // PERFORMANCE: Optimal hash map sizing and load factor
        spatialIndex.reserve(voxels.size());
        spatialIndex.max_load_factor(0.7f);
        
        for (size_t i = 0; i < voxels.size(); ++i) {
            const auto& voxel = voxels[i];
            uint64_t hash = spatialHash(voxel.gridX, voxel.gridY, voxel.gridZ);
            spatialIndex.emplace(hash, i);
        }
    }
    
    /**
     * @brief 3D spatial hash function
     */
    inline uint64_t spatialHash(int x, int y, int z) const {
        return (static_cast<uint64_t>(x + 32768) << 32) | 
               (static_cast<uint64_t>(y + 32768) << 16) | 
               (static_cast<uint64_t>(z + 32768));
    }
};

/**
 * @brief Enhanced VoxelMotionEngine with sparse grid support
 */
class SparseVoxelMotionEngine {
public:
    /**
     * @brief Compute voxel changes between two sparse grids
     * @param sparseGrid1 First sparse voxel grid
     * @param sparseGrid2 Second sparse voxel grid
     * @param minChangeThreshold Minimum change threshold
     * @param brightnessPercentile Brightness percentile filter
     * @return Vector of detected changes
     */
    std::vector<VoxelMotionEngine::ChangeVoxel> computeSparseVoxelChanges(const ::SparseVoxelGrid& sparseGrid1,
                                                      const ::SparseVoxelGrid& sparseGrid2,
                                                      float minChangeThreshold = 0.05f,
                                                      float brightnessPercentile = 99.0f) {
        
        // Create adapters for sparse grids
        SparseVoxelGridAdapter adapter1(sparseGrid1);
        SparseVoxelGridAdapter adapter2(sparseGrid2);
        
        std::cout << "Sparse Motion Analysis:\n";
        std::cout << "Grid 1: " << adapter1.getActiveVoxelCount() << "/" << adapter1.getTotalVoxelCount() 
                  << " active (" << (adapter1.getSparsityRatio() * 100) << "% sparse)\n";
        std::cout << "Grid 2: " << adapter2.getActiveVoxelCount() << "/" << adapter2.getTotalVoxelCount() 
                  << " active (" << (adapter2.getSparsityRatio() * 100) << "% sparse)\n";
        
        // Calculate brightness threshold from both sparse grids
        float brightnessThreshold = calculateSparseBrightnessThreshold(adapter1, adapter2, brightnessPercentile);
        
        std::cout << "Processing sparse voxel changes with brightness threshold: " << brightnessThreshold << "\n";
        
        // PERFORMANCE: More efficient coordinate collection with pre-allocated containers
        std::unordered_set<uint64_t> allCoordinates;
        const size_t estimatedCoords = adapter1.getActiveVoxelCount() + adapter2.getActiveVoxelCount();
        allCoordinates.reserve(estimatedCoords);
        
        // PERFORMANCE: Batch coordinate collection to reduce hash computations
        const auto& voxels1 = adapter1.getVoxels();
        const auto& voxels2 = adapter2.getVoxels();
        
        for (const auto& voxel : voxels1) {
            if (voxel.intersectionCount >= brightnessThreshold) {
                uint64_t hash = spatialHash(voxel.gridX, voxel.gridY, voxel.gridZ);
                allCoordinates.insert(hash);
            }
        }
        
        for (const auto& voxel : voxels2) {
            if (voxel.intersectionCount >= brightnessThreshold) {
                uint64_t hash = spatialHash(voxel.gridX, voxel.gridY, voxel.gridZ);
                allCoordinates.insert(hash);
            }
        }
        
        std::cout << "Analyzing " << allCoordinates.size() << " unique voxel locations...\n";
        
        // PERFORMANCE: Use thread-local storage to reduce mutex contention
        std::vector<uint64_t> coordVector(allCoordinates.begin(), allCoordinates.end());
        
        // Pre-allocate thread-local change vectors
        const int numThreads = std::max(1, omp_get_max_threads());
        std::vector<std::vector<ChangeVoxel>> threadLocalChanges(numThreads);
        
        #pragma omp parallel
        {
            const int threadId = omp_get_thread_num();
            auto& localChanges = threadLocalChanges[threadId];
            localChanges.reserve(coordVector.size() / numThreads + 100);
            
            #pragma omp for schedule(static)
            for (size_t i = 0; i < coordVector.size(); ++i) {
                uint64_t hash = coordVector[i];
                
                // PERFORMANCE: Faster coordinate decoding
                int x = static_cast<int>((hash >> 32) - 32768);
                int y = static_cast<int>(((hash >> 16) & 0xFFFF) - 32768);
                int z = static_cast<int>((hash & 0xFFFF) - 32768);
                
                const VoxelData* voxel1 = adapter1.findVoxelByIndices(x, y, z);
                const VoxelData* voxel2 = adapter2.findVoxelByIndices(x, y, z);
                
                ChangeVoxel change = calculateSparseVoxelChange(voxel1, voxel2, adapter1, adapter2);
                
                if (std::abs(change.changeIntensity) >= minChangeThreshold) {
                    localChanges.push_back(change);
                }
            }
        }
        
        // PERFORMANCE: Merge thread-local results efficiently
        std::vector<ChangeVoxel> changes;
        size_t totalChanges = 0;
        for (const auto& localChanges : threadLocalChanges) {
            totalChanges += localChanges.size();
        }
        changes.reserve(totalChanges);
        
        for (const auto& localChanges : threadLocalChanges) {
            changes.insert(changes.end(), localChanges.begin(), localChanges.end());
        }
        
        // Sort by change intensity (highest first)
        std::sort(changes.begin(), changes.end(), 
                  [](const ChangeVoxel& a, const ChangeVoxel& b) {
                      return std::abs(a.changeIntensity) > std::abs(b.changeIntensity);
                  });
        
        std::cout << "Detected " << changes.size() << " significant changes\n";
        
        return changes;
    }
    
    /**
     * @brief Save sparse change results with enhanced metadata
     */
    void saveSparseChangeGrid(const std::vector<VoxelMotionEngine::ChangeVoxel>& changeVoxels,
                             const ::SparseVoxelGrid& sourceGrid,
                             const std::string& filename) {
        
        json outputData;
        
        // Enhanced metadata for sparse grids
        outputData["metadata"] = {
            {"format_version", "2.0_sparse"},
            {"generation_time", std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()},
            {"total_changes", changeVoxels.size()},
            {"grid_dimensions", {sourceGrid.getSizeX(), sourceGrid.getSizeY(), sourceGrid.getSizeZ()}},
            {"voxel_size_m", sourceGrid.getVoxelSize()},
            {"grid_origin_m", {sourceGrid.getOrigin().getX(), sourceGrid.getOrigin().getY(), sourceGrid.getOrigin().getZ()}},
            {"sparse_statistics", {
                {"active_voxels", sourceGrid.getActiveVoxelCount()},
                {"total_voxels", sourceGrid.getMaxVoxelCount()},
                {"sparsity_ratio", sourceGrid.getSparsityRatio()},
                {"memory_efficiency", (1.0f - static_cast<float>(sourceGrid.getActiveVoxelCount()) / sourceGrid.getMaxVoxelCount()) * 100}
            }}
        };
        
        // Change statistics
        if (!changeVoxels.empty()) {
            float maxChange = 0, avgChange = 0, minChange = 0;
            for (const auto& change : changeVoxels) {
                float absChange = std::abs(change.changeIntensity);
                maxChange = std::max(maxChange, absChange);
                minChange = (minChange == 0) ? absChange : std::min(minChange, absChange);
                avgChange += absChange;
            }
            avgChange /= changeVoxels.size();
            
            outputData["change_statistics"] = {
                {"max_change_intensity", maxChange},
                {"min_change_intensity", minChange},
                {"avg_change_intensity", avgChange}
            };
        }
        
        // Export changes
        json changesArray = json::array();
        for (const auto& change : changeVoxels) {
            changesArray.push_back({
                {"position_m", {change.position.getX(), change.position.getY(), change.position.getZ()}},
                {"grid_indices", {change.gridX, change.gridY, change.gridZ}},
                {"change_intensity", change.changeIntensity},
                {"change_type", change.changeType},
                {"absolute_change", change.absoluteChange},
                {"relative_change", change.relativeChange},
                {"before_intersection_count", change.grid1IntersectionCount},
                {"after_intersection_count", change.grid2IntersectionCount},
                {"before_camera_count", change.grid1CameraCount},
                {"after_camera_count", change.grid2CameraCount}
            });
        }
        outputData["changes"] = changesArray;
        
        // Write to file
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            throw std::runtime_error("Could not open sparse change output file: " + filename);
        }
        
        outFile << outputData.dump(2);
        outFile.close();
        
        std::cout << "Sparse change grid saved to " << filename 
                  << " (" << changeVoxels.size() << " changes, "
                  << (sourceGrid.getSparsityRatio() * 100) << "% sparse)\n";
    }

private:
    /**
     * @brief Calculate brightness threshold for sparse grids
     */
    float calculateSparseBrightnessThreshold(const SparseVoxelGridAdapter& adapter1,
                                           const SparseVoxelGridAdapter& adapter2,
                                           float percentile = 99.0f) {
        
        std::vector<float> allBrightness;
        allBrightness.reserve(adapter1.getActiveVoxelCount() + adapter2.getActiveVoxelCount());
        
        // Collect brightness values from both sparse grids
        for (const auto& voxel : adapter1.getVoxels()) {
            allBrightness.push_back(voxel.intersectionCount);
        }
        for (const auto& voxel : adapter2.getVoxels()) {
            allBrightness.push_back(voxel.intersectionCount);
        }
        
        if (allBrightness.empty()) return 0.0f;
        
        std::sort(allBrightness.begin(), allBrightness.end());
        
        size_t thresholdIndex = static_cast<size_t>(allBrightness.size() * percentile / 100.0f);
        thresholdIndex = std::min(thresholdIndex, allBrightness.size() - 1);
        
        return allBrightness[thresholdIndex];
    }
    
    /**
     * @brief Calculate change metrics for sparse voxel pair
     */
    VoxelMotionEngine::ChangeVoxel calculateSparseVoxelChange(const VoxelMotionEngine::VoxelData* voxel1, const VoxelMotionEngine::VoxelData* voxel2,
                                          const SparseVoxelGridAdapter& adapter1,
                                          const SparseVoxelGridAdapter& adapter2) {
        
        VoxelMotionEngine::ChangeVoxel change;
        
        // Determine position and grid coordinates
        if (voxel1) {
            change.position = voxel1->position;
            change.gridX = voxel1->gridX;
            change.gridY = voxel1->gridY;
            change.gridZ = voxel1->gridZ;
            change.grid1IntersectionCount = voxel1->intersectionCount;
            change.grid1CameraCount = voxel1->numCamerasIntersecting;
        } else if (voxel2) {
            change.position = voxel2->position;
            change.gridX = voxel2->gridX;
            change.gridY = voxel2->gridY;
            change.gridZ = voxel2->gridZ;
            change.grid1IntersectionCount = 0.0f;
            change.grid1CameraCount = 0;
        }
        
        if (voxel2) {
            change.grid2IntersectionCount = voxel2->intersectionCount;
            change.grid2CameraCount = voxel2->numCamerasIntersecting;
        } else {
            change.grid2IntersectionCount = 0.0f;
            change.grid2CameraCount = 0;
        }
        
        // Calculate change metrics
        change.absoluteChange = change.grid2IntersectionCount - change.grid1IntersectionCount;
        
        if (change.grid1IntersectionCount > 0) {
            change.relativeChange = (change.absoluteChange / change.grid1IntersectionCount) * 100.0f;
        } else {
            change.relativeChange = (change.grid2IntersectionCount > 0) ? 100.0f : 0.0f;
        }
        
        // Normalize change intensity (0-1 scale)
        float maxIntensity = std::max(change.grid1IntersectionCount, change.grid2IntersectionCount);
        change.changeIntensity = (maxIntensity > 0) ? (std::abs(change.absoluteChange) / maxIntensity) : 0.0f;
        
        // Determine change type
        if (change.absoluteChange > 0.01f) {
            change.changeType = 1.0f; // Increase
        } else if (change.absoluteChange < -0.01f) {
            change.changeType = -1.0f; // Decrease
        } else {
            change.changeType = 0.0f; // No significant change
        }
        
        return change;
    }
    
    /**
     * @brief Spatial hash function for coordinate mapping
     */
    inline uint64_t spatialHash(int x, int y, int z) const {
        return (static_cast<uint64_t>(x + 32768) << 32) | 
               (static_cast<uint64_t>(y + 32768) << 16) | 
               (static_cast<uint64_t>(z + 32768));
    }
};

} // namespace SparseVoxelMotionExt