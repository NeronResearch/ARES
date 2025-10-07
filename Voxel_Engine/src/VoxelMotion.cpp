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
#include "VoxelMotion.h"

using json = nlohmann::json;



// VoxelData method implementations
VoxelMotionEngine::VoxelData::VoxelData(const XYZ& pos, float count, int cameras, int x, int y, int z)
    : position(pos), intersectionCount(count), numCamerasIntersecting(cameras),
      gridX(x), gridY(y), gridZ(z) {}

uint64_t VoxelMotionEngine::VoxelData::getHash() const {
    if (cachedHash == 0) {
        cachedHash = (static_cast<uint64_t>(gridX + 32768) << 32) | 
                   (static_cast<uint64_t>(gridY + 32768) << 16) | 
                   (static_cast<uint64_t>(gridZ + 32768));
    }
    return cachedHash;
}

// ChangeVoxel method implementations
VoxelMotionEngine::ChangeVoxel::ChangeVoxel() 
    : position(0,0,0), changeIntensity(0), changeType(0), 
      absoluteChange(0), relativeChange(0),
      gridX(0), gridY(0), gridZ(0),
      grid1IntersectionCount(0), grid2IntersectionCount(0),
      grid1CameraCount(0), grid2CameraCount(0) {}

// VoxelGrid constructor implementation
VoxelMotionEngine::VoxelGrid::VoxelGrid(const std::vector<VoxelData>& voxelData, 
                                        const json& sceneMetadata, 
                                        const json& camerasData, 
                                        const json& targetsData) 
    : sceneInfo(sceneMetadata), cameras(camerasData), targets(targetsData), voxels(voxelData) {
    buildSpatialIndex();
}

// Simplified constructor for VoxelGrid that only takes voxel data
VoxelMotionEngine::VoxelGrid::VoxelGrid(const std::vector<VoxelData>& voxelData) 
    : voxels(voxelData) {
    buildSpatialIndex();
}

// Constructor from VoxelEngine VoxelGrid
VoxelMotionEngine::VoxelGrid::VoxelGrid(const ::VoxelGrid& engineGrid) {
    // Convert from VoxelEngine::VoxelGrid to VoxelMotionEngine::VoxelData
    voxels.reserve(engineGrid.voxels.size());
    
    for (int z = 0; z < engineGrid.getSizeZ(); ++z) {
        for (int y = 0; y < engineGrid.getSizeY(); ++y) {
            for (int x = 0; x < engineGrid.getSizeX(); ++x) {
                const auto& voxel = engineGrid.voxels[engineGrid.indexFromIndices(x, y, z)];
                
                // Only add voxels that have intersection data
                if (voxel.getIntersectionCount() > 0) {
                    VoxelData voxelData(
                        voxel.getPosition(),
                        voxel.getIntersectionCount(),
                        voxel.getNumCamerasIntersecting(),
                        x, y, z
                    );
                    voxels.push_back(voxelData);
                }
            }
        }
    }
    
    buildSpatialIndex();
}

// VoxelGrid method implementations
void VoxelMotionEngine::VoxelGrid::buildSpatialIndex() {
    spatialIndex.clear();
    spatialIndex.reserve(voxels.size() * 2);
    
    if (voxels.empty()) return;
    
    // SIMD-friendly bounds calculation
    minX = maxX = voxels[0].gridX;
    minY = maxY = voxels[0].gridY;
    minZ = maxZ = voxels[0].gridZ;
    
    // Vectorized bounds finding
    const size_t simd_size = 4;
    size_t i = 0;
    for (; i + simd_size <= voxels.size(); i += simd_size) {
        for (size_t j = 0; j < simd_size; ++j) {
            const auto& voxel = voxels[i + j];
            minX = std::min(minX, voxel.gridX);
            maxX = std::max(maxX, voxel.gridX);
            minY = std::min(minY, voxel.gridY);
            maxY = std::max(maxY, voxel.gridY);
            minZ = std::min(minZ, voxel.gridZ);
            maxZ = std::max(maxZ, voxel.gridZ);
        }
    }
    // Handle remaining elements
    for (; i < voxels.size(); ++i) {
        const auto& voxel = voxels[i];
        minX = std::min(minX, voxel.gridX);
        maxX = std::max(maxX, voxel.gridX);
        minY = std::min(minY, voxel.gridY);
        maxY = std::max(maxY, voxel.gridY);
        minZ = std::min(minZ, voxel.gridZ);
        maxZ = std::max(maxZ, voxel.gridZ);
    }
    
    // Build index using cached hashes
    for (size_t i = 0; i < voxels.size(); ++i) {
        spatialIndex[voxels[i].getHash()] = i;
    }
}

uint64_t VoxelMotionEngine::VoxelGrid::spatialHash(int x, int y, int z) const {
    // Fast 3D hash using bit shifting
    return (static_cast<uint64_t>(x + 32768) << 32) | 
           (static_cast<uint64_t>(y + 32768) << 16) | 
           (static_cast<uint64_t>(z + 32768));
}

const VoxelMotionEngine::VoxelData* VoxelMotionEngine::VoxelGrid::findVoxelByIndices(int x, int y, int z) const {
    uint64_t key = spatialHash(x, y, z);
    auto it = spatialIndex.find(key);
    if (it != spatialIndex.end()) {
        return &voxels[it->second];
    }
    return nullptr;
}

bool VoxelMotionEngine::VoxelGrid::isInBounds(int x, int y, int z) const {
    return x >= minX && x <= maxX && y >= minY && y <= maxY && z >= minZ && z <= maxZ;
}

// VoxelMotionEngine method implementations
float VoxelMotionEngine::calculateBrightnessThreshold(const VoxelGrid& grid1, const VoxelGrid& grid2, float percentile) {
    // Pre-allocate with estimated size
    std::vector<float> allIntersectionCounts;
    allIntersectionCounts.reserve(grid1.voxels.size() + grid2.voxels.size());
    
    // Fast collection with single pass and branch prediction optimization
    for (const auto& voxel : grid1.voxels) {
        if (voxel.intersectionCount > 0.0f) {
            allIntersectionCounts.push_back(voxel.intersectionCount);
        }
    }
    
    for (const auto& voxel : grid2.voxels) {
        if (voxel.intersectionCount > 0.0f) {
            allIntersectionCounts.push_back(voxel.intersectionCount);
        }
    }
    
    if (allIntersectionCounts.empty()) {
        std::cout << "No non-zero voxels found for brightness threshold calculation" << std::endl;
        return 0.0f;
    }
    
    // Use nth_element for O(n) partial sort instead of full O(n log n) sort
    size_t thresholdIndex = static_cast<size_t>((percentile / 100.0f) * allIntersectionCounts.size());
    if (thresholdIndex >= allIntersectionCounts.size()) {
        thresholdIndex = allIntersectionCounts.size() - 1;
    }
    
    std::nth_element(allIntersectionCounts.begin(), 
                    allIntersectionCounts.begin() + thresholdIndex, 
                    allIntersectionCounts.end());
    
    float threshold = allIntersectionCounts[thresholdIndex];
    std::cout << "Brightness threshold (" << percentile << "th percentile): " << threshold 
              << " (from " << allIntersectionCounts.size() << " non-zero voxels)" << std::endl;
    
    return threshold;
}

std::vector<VoxelMotionEngine::ChangeVoxel> VoxelMotionEngine::computeVoxelChanges(const VoxelGrid& grid1, const VoxelGrid& grid2,
                                            float minChangeThreshold, float brightnessPercentile) {
    std::cout << "Computing voxel changes between grids (ULTRA-FAST mode, brightest " << (100.0f - brightnessPercentile) << "% only)..." << std::endl;
    
    // Calculate brightness threshold for filtering
    float brightnessThreshold = calculateBrightnessThreshold(grid1, grid2, brightnessPercentile);
    
    // Create a combined set of all voxel positions from both grids
    std::unordered_map<uint64_t, std::pair<const VoxelData*, const VoxelData*>> voxelPairs;
    
    // Track filtering statistics
    size_t grid1NonZeroCount = 0;
    size_t grid2NonZeroCount = 0;
    
    // Add all voxels from grid1 (only bright voxels above threshold)
    for (const auto& voxel : grid1.voxels) {
        if (voxel.intersectionCount >= brightnessThreshold) {
            uint64_t key = grid1.spatialHash(voxel.gridX, voxel.gridY, voxel.gridZ);
            voxelPairs[key].first = &voxel;
            grid1NonZeroCount++;
        }
    }
    
    // Add all voxels from grid2 (only bright voxels above threshold)
    for (const auto& voxel : grid2.voxels) {
        if (voxel.intersectionCount >= brightnessThreshold) {
            uint64_t key = grid2.spatialHash(voxel.gridX, voxel.gridY, voxel.gridZ);
            voxelPairs[key].second = &voxel;
            grid2NonZeroCount++;
        }
    }
    
    std::cout << "Grid1: " << grid1NonZeroCount << "/" << grid1.voxels.size() 
              << " bright voxels (" << (100.0f * grid1NonZeroCount / grid1.voxels.size()) << "%)" << std::endl;
    std::cout << "Grid2: " << grid2NonZeroCount << "/" << grid2.voxels.size() 
              << " bright voxels (" << (100.0f * grid2NonZeroCount / grid2.voxels.size()) << "%)" << std::endl;
    std::cout << "Total unique voxel positions to process: " << voxelPairs.size() << std::endl;
    
    // Pre-allocate result vector
    std::vector<ChangeVoxel> changeVoxels;
    changeVoxels.reserve(voxelPairs.size());
    
    // Lock-free parallel processing with optimized chunk size
    const int numThreads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<std::vector<ChangeVoxel>> threadResults(numThreads);
    
    // Pre-reserve thread result vectors for better memory efficiency
    size_t estimatedResultsPerThread = voxelPairs.size() / numThreads + 100;
    for (auto& threadResult : threadResults) {
        threadResult.reserve(estimatedResultsPerThread);
    }
    
    std::cout << "Using " << numThreads << " threads for parallel processing" << std::endl;
    
    // Convert map to vector for cache-friendly access
    std::vector<std::pair<uint64_t, std::pair<const VoxelData*, const VoxelData*>>> voxelPairVector;
    voxelPairVector.reserve(voxelPairs.size());
    voxelPairVector.assign(voxelPairs.begin(), voxelPairs.end());
    
    // Optimized chunk size for better cache locality
    const int chunkSize = std::max(1, static_cast<int>(voxelPairVector.size() / (numThreads * 8)));
    
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static, chunkSize) num_threads(numThreads)
    #endif
    for (int i = 0; i < static_cast<int>(voxelPairVector.size()); ++i) {
        const auto& [hash, voxelPair] = voxelPairVector[i];
        const VoxelData* voxel1 = voxelPair.first;
        const VoxelData* voxel2 = voxelPair.second;
        
        #ifdef _OPENMP
        int threadId = omp_get_thread_num();
        #else
        int threadId = 0;
        #endif
        
        // Skip if both voxels are null or below brightness threshold
        float intersect1 = voxel1 ? voxel1->intersectionCount : 0.0f;
        float intersect2 = voxel2 ? voxel2->intersectionCount : 0.0f;
        
        if (intersect1 < brightnessThreshold && intersect2 < brightnessThreshold) {
            continue; // Skip voxels that don't meet brightness criteria
        }
        
        ChangeVoxel change = calculateVoxelChange(voxel1, voxel2, grid1, grid2);
        
        // Only include voxels with significant change
        if (change.changeIntensity >= minChangeThreshold) {
            threadResults[threadId].push_back(change);
        }
    }
    
    // Combine results from all threads
    size_t totalSize = 0;
    for (const auto& threadResult : threadResults) {
        totalSize += threadResult.size();
    }
    
    changeVoxels.reserve(totalSize);
    for (const auto& threadResult : threadResults) {
        changeVoxels.insert(changeVoxels.end(), threadResult.begin(), threadResult.end());
    }
    
    std::cout << "Found " << changeVoxels.size() << " voxels with significant changes (>= " 
              << minChangeThreshold << " intensity)" << std::endl;
    
    return changeVoxels;
}

VoxelMotionEngine::ChangeVoxel VoxelMotionEngine::calculateVoxelChange(const VoxelData* voxel1, const VoxelData* voxel2, 
                                      const VoxelGrid& grid1, const VoxelGrid& grid2) {
    ChangeVoxel change;
    
    // Fast branch-free value extraction
    const float intersect1 = voxel1 ? voxel1->intersectionCount : 0.0f;
    const float intersect2 = voxel2 ? voxel2->intersectionCount : 0.0f;
    const int cameras1 = voxel1 ? voxel1->numCamerasIntersecting : 0;
    const int cameras2 = voxel2 ? voxel2->numCamerasIntersecting : 0;
    
    // Early return optimization - branch prediction friendly
    if ((intersect1 <= 0.0f) & (intersect2 <= 0.0f)) {
        return change;
    }
    
    // Branchless position assignment (prefer voxel1)
    const VoxelData* sourceVoxel = voxel1 ? voxel1 : voxel2;
    if (sourceVoxel) {
        change.position = sourceVoxel->position;
        change.gridX = sourceVoxel->gridX;
        change.gridY = sourceVoxel->gridY;
        change.gridZ = sourceVoxel->gridZ;
    }
    
    // Fast math operations with minimal branching
    const float diff = intersect2 - intersect1;
    change.absoluteChange = std::abs(diff);
    
    // Optimized relative change calculation
    const float maxIntersect = std::max(intersect1, intersect2);
    change.relativeChange = (maxIntersect > 0.0f) ? (change.absoluteChange / maxIntersect) : 0.0f;
    
    // Vectorized intensity calculation
    const float normalizedAbsolute = std::min(1.0f, change.absoluteChange * 0.1f); // Pre-computed 1/10
    const float cameraChangeScore = std::abs(cameras2 - cameras1) * 0.1f;
    change.changeIntensity = std::min(1.0f, 
        change.relativeChange * 0.6f + 
        normalizedAbsolute * 0.3f + 
        cameraChangeScore * 0.1f);
    
    // Branchless change type calculation
    change.changeType = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
    
    // Store values directly
    change.grid1IntersectionCount = intersect1;
    change.grid2IntersectionCount = intersect2;
    change.grid1CameraCount = cameras1;
    change.grid2CameraCount = cameras2;
    
    return change;
}

void VoxelMotionEngine::saveChangeGrid(const std::vector<ChangeVoxel>& changeVoxels, 
                   const VoxelGrid& sourceGrid,
                   const std::string& filename) {
    std::cout << "Saving change grid to: " << filename << std::endl;
    
    json sceneData;
    
    // Create scene_info section matching VoxelEngine::printScene format
    if (!sourceGrid.sceneInfo.empty()) {
        sceneData["scene_info"] = sourceGrid.sceneInfo;
        // Add additional change-specific metadata 
        sceneData["scene_info"]["data_type"] = "voxel_change_detection";
        sceneData["scene_info"]["change_voxels"] = changeVoxels.size();
    } else {
        // Create minimal scene info matching the printScene format
        sceneData["scene_info"] = {
            {"num_cameras", 0}, // Will be updated if cameras are available
            {"voxel_size_m", 5.0f}, // Default voxel size
            {"grid_dimensions", {0, 0, 0}}, // Will be calculated from voxels
            {"grid_origin_m", {0.0f, 0.0f, 0.0f}}, // Will be calculated from voxels
            {"grid_size_m", {0.0f, 0.0f, 0.0f}}, // Will be calculated from voxels
            {"data_type", "voxel_change_detection"},
            {"change_voxels", changeVoxels.size()}
        };
    }
    
    // Copy cameras data if available (matching printScene format)
    if (!sourceGrid.cameras.empty()) {
        sceneData["cameras"] = sourceGrid.cameras;
        // Update num_cameras in scene_info
        sceneData["scene_info"]["num_cameras"] = sourceGrid.cameras.size();
    } else {
        // Create empty cameras array to match format
        sceneData["cameras"] = json::array();
    }
    
    // Voxel data section matching VoxelEngine::printScene format
    // Sort change voxels by intensity (descending) and limit to top 10k
    std::vector<ChangeVoxel> sortedChanges = changeVoxels;
    std::sort(sortedChanges.begin(), sortedChanges.end(), 
        [](const ChangeVoxel& a, const ChangeVoxel& b) {
            return a.changeIntensity > b.changeIntensity;
        });
    
    const size_t maxVoxelsToSave = 1000;
    size_t voxelsToProcess = std::min(maxVoxelsToSave, sortedChanges.size());
    
    json voxelArray = json::array();
    int totalVoxels = changeVoxels.size();
    int nonZeroVoxels = 0;
    
    for (size_t i = 0; i < voxelsToProcess; ++i) {
        const auto& change = sortedChanges[i];
        if (change.changeIntensity > 0) {
            nonZeroVoxels++;
            voxelArray.push_back({
                {"indices", {change.gridX, change.gridY, change.gridZ}},
                {"intersection_count", change.changeIntensity}, // Use change intensity as intersection count
                {"num_cameras", static_cast<int>(std::max(change.grid1CameraCount, change.grid2CameraCount))}
            });
        }
    }
    
    std::cout << "Saving top " << voxelsToProcess << " brightest change voxels (out of " << totalVoxels << " total)" << std::endl;
    
    sceneData["voxel_data"] = {
        {"total_voxels", totalVoxels},
        {"non_zero_voxels", nonZeroVoxels},
        {"voxels", voxelArray}
    };
    
    // Write to file
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }
    
    outFile << sceneData.dump(2);
    outFile.close();
    
    std::cout << "Scene data written to " << filename << " (" << nonZeroVoxels << "/" << totalVoxels << " voxels with changes)\n";
}

void VoxelMotionEngine::processVoxelChanges(const VoxelGrid& grid1, const VoxelGrid& grid2, 
                       const std::string& outputPath) {
    std::cout << "=== ARES Voxel Change Detection Engine ===" << std::endl;
    std::cout << "Processing " << grid1.voxels.size() << " and " << grid2.voxels.size() << " voxels" << std::endl;
    std::cout << "Output: " << outputPath << std::endl;
    std::cout << std::endl;
    
    // Validate that grids are compatible (if metadata is available)
    if (!grid1.sceneInfo.empty() && !grid2.sceneInfo.empty() &&
        (grid1.sceneInfo["voxel_size_m"] != grid2.sceneInfo["voxel_size_m"] ||
         grid1.sceneInfo["grid_dimensions"] != grid2.sceneInfo["grid_dimensions"])) {
        std::cerr << "Warning: Voxel grids have different parameters. Change detection may be inaccurate." << std::endl;
    }
    
    // START PROCESSING TIMER (excluding I/O)
    auto processingStartTime = std::chrono::high_resolution_clock::now();
    
    // Compute changes (only brightest 1% of voxels)
    float brightnessPercentile = 99.0f; // Only process top 1% brightest voxels
    std::vector<ChangeVoxel> changeVoxels = computeVoxelChanges(grid1, grid2, 0.05f, brightnessPercentile);
    
    // END PROCESSING TIMER (excluding I/O)
    auto processingEndTime = std::chrono::high_resolution_clock::now();
    auto processingDuration = std::chrono::duration_cast<std::chrono::milliseconds>(processingEndTime - processingStartTime);
    
    // Save results (I/O time not measured)
    saveChangeGrid(changeVoxels, grid1, outputPath);
    
    std::cout << std::endl << "=== COMPLETED ===" << std::endl;
    std::cout << "Total processing time: " << processingDuration.count() << "ms (" 
              << std::fixed << std::setprecision(2) << (processingDuration.count() / 1000.0) << "s)" << std::endl;
    std::cout << "========================" << std::endl;
}