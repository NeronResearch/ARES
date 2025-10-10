#pragma once
#include "SparseVoxelEngine.h"
#include "SparseVoxelMotion.h"
#include "VoxelMotion.h"
#include "Target.h"
#include "../third_party/json.hpp"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <limits>

using json = nlohmann::json;

/**
 * @class UnifiedVoxelExporter
 * @brief Clean, minimal JSON export for ML and Matlab compatibility
 * No visualization bullshit - just pure voxel data
 */
class UnifiedVoxelExporter {
public:
    /**
     * @brief Export clean voxel data in unified format for ML/Matlab
     * @param grid1 First voxel grid (scene 1)
     * @param grid2 Second voxel grid (scene 2)  
     * @param changes Detected voxel changes between scenes
     * @param cameras Camera configuration data
     * @param filename Output filename
     * @param targets Target positions array
     * @param targetNames Target names array
     * @param frame Frame number for targets
     */
    static void exportUnifiedScene(const SparseVoxelGrid& grid1,
                                  const SparseVoxelGrid& grid2,
                                  const std::vector<VoxelMotionEngine::ChangeVoxel>& changes,
                                  const std::vector<Camera>& cameras,
                                  const std::string& filename,
                                  const std::vector<Target>& targets = std::vector<Target>(),
                                  const std::vector<std::string>& targetNames = std::vector<std::string>(),
                                  int frame = 25) {
        
        json data;
        
        data["grid_info"] = {
            {"dimensions", {grid1.getSizeX(), grid1.getSizeY(), grid1.getSizeZ()}},
            {"voxel_size_m", grid1.getVoxelSize()},
            {"origin_m", {grid1.getOrigin().getX(), grid1.getOrigin().getY(), grid1.getOrigin().getZ()}},
            {"total_changes", changes.size()}
        };
        
        data["cameras"] = createCameraPositions(cameras);
        
        // Target positions array
        json targetsArray = json::array();
        for (size_t i = 0; i < targets.size(); ++i) {
            json targetObj;
            targetObj["name"] = (i < targetNames.size()) ? targetNames[i] : ("target_" + std::to_string(i));
            targetObj["frame"] = frame;
            const XYZ& pos = targets[i].getCurrentPosition();
            targetObj["position_m"] = {pos.getX(), pos.getY(), pos.getZ()};
            targetObj["is_visible"] = targets[i].isVisible();
            targetsArray.push_back(targetObj);
        }
        data["targets"] = targetsArray;
        
        // Post-motion analysis: final voxel state after motion detection
        data["voxels"] = createVoxelObjectArray(grid2, changes);  // Use scene2 as final state
        
        // Motion analysis summary
        data["motion_analysis"] = createMotionSummary(changes);
        
        // Write to file
        writeJsonFile(data, filename);
    }
    
    /**
     * Export unified scene data with clean ML/Matlab compatible format (DENSE)
     * @param grid1 First dense voxel grid
     * @param grid2 Second dense voxel grid
     * @param changes Motion changes between grids  
     * @param cameras Camera configurations
     * @param filename Output filename
     * @param targets Target positions array
     * @param targetNames Target names array
     * @param frame Frame number for targets
     */
    static void exportUnifiedScene(const VoxelMotionEngine::VoxelGrid& grid1,
                                  const VoxelMotionEngine::VoxelGrid& grid2,
                                  const std::vector<VoxelMotionEngine::ChangeVoxel>& changes,
                                  const std::vector<Camera>& cameras,
                                  const std::string& filename,
                                  const std::vector<Target>& targets = std::vector<Target>(),
                                  const std::vector<std::string>& targetNames = std::vector<std::string>(),
                                  int frame = 25) {
        // Convert dense to sparse format internally and use same export logic
        auto sparseGrid1 = convertDenseToSparse(grid1);
        auto sparseGrid2 = convertDenseToSparse(grid2);
        
        // Call the sparse version with converted data
        exportUnifiedScene(sparseGrid1, sparseGrid2, changes, cameras, filename, targets, targetNames, frame);
    }

    /**
     * @brief Create minimal camera position data
     */
    static json createCameraPositions(const std::vector<Camera>& cameras) {
        json cam_array = json::array();
        
        for (size_t i = 0; i < cameras.size(); ++i) {
            const auto& cam = cameras[i];
            const XYZ& pos = cam.getPosition();
            const Matrix3x3& rot = cam.getRotation();
            
            json cam_data;
            cam_data["id"] = i;
            cam_data["position_m"] = {pos.getX(), pos.getY(), pos.getZ()};
            cam_data["rotation_matrix"] = {
                {rot.m[0][0], rot.m[0][1], rot.m[0][2]},
                {rot.m[1][0], rot.m[1][1], rot.m[1][2]},
                {rot.m[2][0], rot.m[2][1], rot.m[2][2]}
            };
            cam_data["fov_deg"] = cam.getFOV();
            cam_data["resolution"] = {cam.getImageWidth(), cam.getImageHeight()};
            
            cam_array.push_back(cam_data);
        }
        
        return cam_array;
    }
    
    /**
     * Create array of voxel objects for post-motion analysis
     * Each voxel is an object with coordinates, intensity, and motion status
     * Only exports the top 10,000 most intense voxels
     */
    static json createVoxelObjectArray(const SparseVoxelGrid& grid, const std::vector<VoxelMotionEngine::ChangeVoxel>& changes) {
        json voxel_array = json::array();
        
        // Create a map of changed voxels for quick lookup
        std::unordered_map<uint64_t, int> changeMap;  // coordinates hash -> change type
        for (const auto& change : changes) {
            uint64_t hash = hashCoordinates(change.gridX, change.gridY, change.gridZ);
            int changeType = (change.changeType > 0.5f) ? 1 : ((change.changeType < -0.5f) ? -1 : 0);
            changeMap[hash] = changeType;
        }
        
        // First pass: collect all voxels with their intensity for sorting
        struct VoxelData {
            size_t linearIdx;
            int x, y, z;
            float intensity;
            int numCameras;
            float intersectionScore;
            XYZ position;
            int motionType;
        };
        
        std::vector<VoxelData> voxelCandidates;
        voxelCandidates.reserve(grid.getActiveVoxelCount());
        
        for (const auto& [linearIdx, voxel] : grid) {
            // Collect all active voxels, not just multi-camera intersections
            if (voxel.getNumCamerasIntersecting() > 0) {
                // Convert linear index to 3D coordinates
                int z = static_cast<int>(linearIdx / (grid.getSizeX() * grid.getSizeY()));
                int y = static_cast<int>((linearIdx % (grid.getSizeX() * grid.getSizeY())) / grid.getSizeX());
                int x = static_cast<int>(linearIdx % grid.getSizeX());
                
                // Calculate total brightness from all cameras as intensity
                float totalIntensity = 0.0f;
                for (const auto& [camId, brightness] : voxel.getCameraIntersections()) {
                    totalIntensity += brightness;
                }
                
                // Check motion type
                uint64_t hash = hashCoordinates(x, y, z);
                auto it = changeMap.find(hash);
                int motionType = (it != changeMap.end()) ? it->second : 0;
                
                voxelCandidates.push_back({
                    linearIdx, x, y, z, totalIntensity,
                    voxel.getNumCamerasIntersecting(),
                    voxel.getIntersectionCount(),
                    voxel.getPosition(),
                    motionType
                });
            }
        }
        
        // Sort by intensity (highest first) and take top 10,000
        std::sort(voxelCandidates.begin(), voxelCandidates.end(), 
                  [](const VoxelData& a, const VoxelData& b) {
                      return a.intensity > b.intensity;
                  });
        
        // Export only the top 10,000 most intense voxels
        const size_t maxVoxels = std::min(static_cast<size_t>(10000), voxelCandidates.size());
        
        for (size_t i = 0; i < maxVoxels; ++i) {
            const auto& vdata = voxelCandidates[i];
            
            json voxel_obj;
            voxel_obj["coordinates"] = {vdata.x, vdata.y, vdata.z};
            voxel_obj["intensity"] = vdata.intensity;
            voxel_obj["num_cameras"] = vdata.numCameras;
            voxel_obj["intersection_score"] = vdata.intersectionScore;
            voxel_obj["position_m"] = {vdata.position.getX(), vdata.position.getY(), vdata.position.getZ()};
            voxel_obj["motion_type"] = vdata.motionType;
            
            voxel_array.push_back(voxel_obj);
        }
        
        return voxel_array;
    }
    
    /**
     * Create motion analysis summary
     */
    static json createMotionSummary(const std::vector<VoxelMotionEngine::ChangeVoxel>& changes) {
        json motion_summary;
        
        int added = 0, removed = 0, modified = 0;
        float totalChangeIntensity = 0.0f;
        
        for (const auto& change : changes) {
            if (change.changeType > 0.5f) {
                added++;
            } else if (change.changeType < -0.5f) {
                removed++;
            } else {
                modified++;
            }
            totalChangeIntensity += std::abs(change.changeIntensity);
        }
        
        motion_summary["total_changes"] = changes.size();
        motion_summary["added_voxels"] = added;
        motion_summary["removed_voxels"] = removed;
        motion_summary["modified_voxels"] = modified;
        motion_summary["average_change_intensity"] = changes.empty() ? 0.0f : (totalChangeIntensity / changes.size());
        
        return motion_summary;
    }
    
    /**
     * @brief Write JSON data to file with error checking
     */
    /**
     * Hash coordinates for fast lookup
     */
    static uint64_t hashCoordinates(int x, int y, int z) {
        return ((uint64_t)x << 32) | ((uint64_t)y << 16) | (uint64_t)z;
    }
    
    /**
     * Convert dense VoxelMotionEngine::VoxelGrid to SparseVoxelGrid format
     */
    static SparseVoxelGrid convertDenseToSparse(const VoxelMotionEngine::VoxelGrid& denseGrid) {
        // Extract basic grid info from dense grid's voxels
        if (denseGrid.voxels.empty()) {
            // Return empty sparse grid with reasonable defaults
            return SparseVoxelGrid(XYZ(100, 100, 100), XYZ(0, 0, 0), 1.0f);
        }
        
        // Calculate grid bounds from voxel data
        int minX = denseGrid.minX, maxX = denseGrid.maxX;
        int minY = denseGrid.minY, maxY = denseGrid.maxY; 
        int minZ = denseGrid.minZ, maxZ = denseGrid.maxZ;
        
        // Create sparse grid with calculated dimensions
        int sizeX = maxX - minX + 1;
        int sizeY = maxY - minY + 1;
        int sizeZ = maxZ - minZ + 1;
        
        // Estimate voxel size from first voxel (assuming uniform spacing)
        float voxelSize = 5.0f; // Default 5m voxels
        XYZ origin(minX * voxelSize, minY * voxelSize, minZ * voxelSize);
        
        SparseVoxelGrid sparseGrid(XYZ(sizeX, sizeY, sizeZ), origin, voxelSize);
        
        // Convert each dense voxel to sparse format
        for (const auto& voxel : denseGrid.voxels) {
            // Calculate grid coordinates relative to sparse grid origin
            int x = voxel.gridX - minX;
            int y = voxel.gridY - minY;
            int z = voxel.gridZ - minZ;
            
            if (x >= 0 && x < sizeX && y >= 0 && y < sizeY && z >= 0 && z < sizeZ) {
                auto& sparseVoxel = sparseGrid.getOrCreateVoxel(x, y, z);
                
                // Convert dense voxel data by simulating camera intersections
                float intersectionCount = voxel.intersectionCount;
                int numCameras = voxel.numCamerasIntersecting;
                
                // Simulate camera intersections to recreate the voxel state
                if (numCameras > 0 && intersectionCount > 0) {
                    float avgBrightness = intersectionCount / numCameras;
                    for (int camId = 0; camId < numCameras; ++camId) {
                        sparseVoxel.addCameraIntersection(camId, avgBrightness);
                    }
                    sparseVoxel.finalizeIntersections();
                }
            }
        }
        
        return sparseGrid;
    }
    
    /**
     * Convert dense changes to sparse change format
     */
    static std::vector<VoxelMotionEngine::ChangeVoxel> convertDenseChangesToSparse(
        const std::vector<VoxelMotionEngine::ChangeVoxel>& denseChanges) {
        // Dense and sparse use the same ChangeVoxel type, so just return as-is
        return denseChanges;
    }
    
    static void writeJsonFile(const json& data, const std::string& filename) {
        std::ofstream outFile(filename);
        if (!outFile.is_open()) {
            throw std::runtime_error("Could not create output file: " + filename);
        }
        
        outFile << data.dump(2);
        outFile.close();
        
        size_t fileSize = std::filesystem::file_size(filename) / 1024;
        std::cout << "Clean voxel data exported to: " << filename << std::endl;
        std::cout << "File size: " << fileSize << " KB" << std::endl;
    }
};