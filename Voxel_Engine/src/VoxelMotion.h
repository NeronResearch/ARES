#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <future>
#include <atomic>
#include <memory_resource>
#include <execution>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "XYZ.h"
#include "VoxelGrid.h"
#include "../third_party/json.hpp"

using json = nlohmann::json;

/**
 * @class VoxelMotionEngine
 * @brief High-performance voxel change detection engine for 3D spatial analysis
 * 
 * The VoxelMotionEngine processes two voxel grids to detect changes in 3D space,
 * optimized for real-time performance with SIMD operations, parallel processing,
 * and memory-efficient algorithms. Focuses on the brightest voxels for maximum
 * relevance in motion detection applications.
 */
class VoxelMotionEngine {
public:


    /**
     * @struct VoxelData
     * @brief Container for individual voxel information with spatial indexing
     */
    struct VoxelData {
        XYZ position;                    // World position in meters
        float intersectionCount;         // Ray intersection count
        int numCamerasIntersecting;      // Number of cameras that intersect this voxel
        int gridX, gridY, gridZ;        // Grid indices
        
        // Optimization: cached hash for faster spatial lookups
        mutable uint64_t cachedHash = 0;
        
        VoxelData() = default;
        VoxelData(const XYZ& pos, float count, int cameras, int x, int y, int z);
        
        uint64_t getHash() const;
    };

    /**
     * @struct ChangeVoxel
     * @brief Represents detected changes between two voxel states
     */
    struct ChangeVoxel {
        XYZ position;                    // World position in meters
        float changeIntensity;           // 0.0 = no change, 1.0 = maximum change
        float changeType;                // -1.0 = decrease, 0.0 = no change, 1.0 = increase
        float absoluteChange;            // Absolute difference in intersection counts
        float relativeChange;            // Relative change as percentage
        int gridX, gridY, gridZ;        // Grid indices
        float grid1IntersectionCount;    // Original intersection count
        float grid2IntersectionCount;    // New intersection count
        int grid1CameraCount;            // Original camera count
        int grid2CameraCount;            // New camera count
        
        ChangeVoxel();
    };

    /**
     * @struct VoxelGrid
     * @brief Complete voxel grid with spatial indexing and metadata
     */
    struct VoxelGrid {
        json sceneInfo;                                            // Scene metadata
        json cameras;                                              // Camera configurations
        json targets;                                              // Target configurations
        std::vector<VoxelData> voxels;                            // Voxel data array
        std::unordered_map<uint64_t, size_t> spatialIndex;       // Fast spatial lookup
        int minX, minY, minZ, maxX, maxY, maxZ;                  // Bounding box
        
        static constexpr size_t EXPECTED_VOXEL_COUNT = 50000;
        
        /**
         * @brief Constructor from voxel data vectors
         * @param voxelData Vector of voxel data
         * @param sceneMetadata Scene information JSON
         * @param camerasData Cameras configuration JSON
         * @param targetsData Targets configuration JSON
         */
        VoxelGrid(const std::vector<VoxelData>& voxelData, 
                  const json& sceneMetadata, 
                  const json& camerasData, 
                  const json& targetsData);
        
        /**
         * @brief Simplified constructor from voxel data only
         * @param voxelData Vector of voxel data
         */
        VoxelGrid(const std::vector<VoxelData>& voxelData);
        
        /**
         * @brief Constructor from VoxelEngine VoxelGrid
         * @param engineGrid VoxelEngine's VoxelGrid object
         */
        VoxelGrid(const ::VoxelGrid& engineGrid);
        
        /**
         * @brief Default constructor
         */
        VoxelGrid() = default;
        
        /**
         * @brief Builds spatial index for O(1) voxel lookups
         */
        void buildSpatialIndex();
        
        /**
         * @brief Fast 3D spatial hash function
         */
        inline uint64_t spatialHash(int x, int y, int z) const;
        
        /**
         * @brief Find voxel by grid coordinates
         * @param x Grid X coordinate
         * @param y Grid Y coordinate  
         * @param z Grid Z coordinate
         * @return Pointer to voxel data or nullptr if not found
         */
        const VoxelData* findVoxelByIndices(int x, int y, int z) const;
        
        /**
         * @brief Check if coordinates are within grid bounds
         */
        bool isInBounds(int x, int y, int z) const;
    };



    /**
     * @brief Calculate brightness threshold for voxel filtering
     * @param grid1 First voxel grid
     * @param grid2 Second voxel grid
     * @param percentile Percentile threshold (default 99.0% for top 1% brightest)
     * @return Brightness threshold value
     */
    float calculateBrightnessThreshold(const VoxelGrid& grid1, const VoxelGrid& grid2, 
                                     float percentile = 99.0f);

    /**
     * @brief Compute changes between two voxel grids (optimized for brightest voxels)
     * @param grid1 First voxel grid (before state)
     * @param grid2 Second voxel grid (after state)
     * @param minChangeThreshold Minimum change intensity to include (default 0.05)
     * @param brightnessPercentile Only process voxels above this brightness percentile (default 99.0%)
     * @return Vector of detected changes
     */
    std::vector<ChangeVoxel> computeVoxelChanges(const VoxelGrid& grid1, const VoxelGrid& grid2,
                                               float minChangeThreshold = 0.05f, 
                                               float brightnessPercentile = 99.0f);

    /**
     * @brief Calculate change metrics for a single voxel pair
     * @param voxel1 Voxel from first grid (can be nullptr)
     * @param voxel2 Voxel from second grid (can be nullptr)
     * @param grid1 First voxel grid reference
     * @param grid2 Second voxel grid reference
     * @return Calculated change voxel data
     */
    inline ChangeVoxel calculateVoxelChange(const VoxelData* voxel1, const VoxelData* voxel2, 
                                          const VoxelGrid& grid1, const VoxelGrid& grid2);

    /**
     * @brief Save change detection results to JSON file
     * @param changeVoxels Vector of detected changes
     * @param sourceGrid Source grid for metadata
     * @param filename Output filename
     * @throws std::runtime_error if file cannot be created
     */
    void saveChangeGrid(const std::vector<ChangeVoxel>& changeVoxels, 
                       const VoxelGrid& sourceGrid,
                       const std::string& filename);

    /**
     * @brief Process complete voxel change detection pipeline
     * @param grid1 First voxel grid object
     * @param grid2 Second voxel grid object
     * @param outputPath Path for output change grid file
     * @throws std::runtime_error if processing fails
     */
    void processVoxelChanges(const VoxelGrid& grid1, const VoxelGrid& grid2, 
                           const std::string& outputPath);

private:
    // Future: Add private helper methods here if needed
};
