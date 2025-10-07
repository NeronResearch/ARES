#include "VoxelEngine.h"
#include "Camera.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <limits>
#include "../third_party/json.hpp"

using json = nlohmann::json;

// VoxelEngine::Raycaster Implementation
VoxelEngine::Raycaster::Raycaster(const Camera& cam, const VoxelGrid& grid)
    : camera(cam), voxelGrid(grid) {}

void VoxelEngine::Raycaster::calculateRayIntersectionsUltraFast(VoxelGrid& voxelGrid, const std::vector<Camera>& cameras, 
                                                              float maxDistance, float topPercentage) {
    std::vector<std::atomic<int>> brightnessHistogram(64);
    std::atomic<int> totalPixels(0);
    
    for (auto& bin : brightnessHistogram) {
        bin.store(0);
    }
    
    const int megaSample = 32; // Restore original sampling for consistent results
    #pragma omp parallel for
    for (int camIdx = 0; camIdx < static_cast<int>(cameras.size()); ++camIdx) {
        const auto& cam = cameras[camIdx];
        int localPixels = 0;
        
        // Process in SIMD-friendly chunks
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
    
    const int rayStep = 2;  // Keep original step to maintain stats
    #pragma omp parallel for schedule(static) 
    for (int camIdx = 0; camIdx < static_cast<int>(cameras.size()); ++camIdx) {
        const auto& cam = cameras[camIdx];
        int raysThisCamera = 0;
        
        // Cache-friendly access pattern with blocked iteration
        const int blockSize = 32;  // Larger blocks for better cache utilization
        for (int by = 0; by < cam.getImageHeight(); by += blockSize) {
            for (int bx = 0; bx < cam.getImageWidth(); bx += blockSize) {
                int maxY = std::min(by + blockSize, cam.getImageHeight());
                int maxX = std::min(bx + blockSize, cam.getImageWidth());
                
                for (int y = by; y < maxY; y += rayStep) {
                    for (int x = bx; x < maxX; x += rayStep) {
                        float brightness = cam.getPixelBrightness(x, y);
                        
                        // Early continue for better branch prediction
                        if (brightness < brightnessThreshold) continue;
                        
                        Ray ray = cam.generateRay(x, y);
                        bool found = CastRayAndAccumulate(voxelGrid, ray.origin, ray.direction, maxDistance, brightness, camIdx);
                        raysThisCamera += found ? 1 : 0;  // Branchless increment
                    }
                }
            }
        }
        
        totalRaysProcessed.fetch_add(raysThisCamera);
    }
    
    std::cout << "Processed " << totalRaysProcessed.load() << " rays\n";
    
    int totalVoxels = voxelGrid.getSizeX() * voxelGrid.getSizeY() * voxelGrid.getSizeZ();
    
    #pragma omp parallel for
    for (int i = 0; i < totalVoxels; ++i) {
        voxelGrid.voxels[i].finalizeIntersections();
    }
}

bool VoxelEngine::Raycaster::CastRayAndAccumulate(VoxelGrid& grid, const XYZ& origin, const XYZ& dir, 
                                                 float maxDistance, float intensity, int cameraId) {
    const float voxelSize = grid.getVoxelSize();
    const float invVoxelSize = 1.0f / voxelSize;
    bool foundIntersection = false;

    // Optimized initial position calculation
    const XYZ& gridOrigin = grid.getOrigin();
    int x = static_cast<int>(std::floor((origin.getX() - gridOrigin.getX()) * invVoxelSize));
    int y = static_cast<int>(std::floor((origin.getY() - gridOrigin.getY()) * invVoxelSize));
    int z = static_cast<int>(std::floor((origin.getZ() - gridOrigin.getZ()) * invVoxelSize));

    // Precompute step directions and deltas
    const int stepX = (dir.getX() > 0) ? 1 : -1;
    const int stepY = (dir.getY() > 0) ? 1 : -1;
    const int stepZ = (dir.getZ() > 0) ? 1 : -1;

    const float tDeltaX = (dir.getX() != 0) ? std::abs(voxelSize / dir.getX()) : std::numeric_limits<float>::infinity();
    const float tDeltaY = (dir.getY() != 0) ? std::abs(voxelSize / dir.getY()) : std::numeric_limits<float>::infinity();
    const float tDeltaZ = (dir.getZ() != 0) ? std::abs(voxelSize / dir.getZ()) : std::numeric_limits<float>::infinity();

    // Initial boundary calculations
    float tMaxX = (dir.getX() != 0) ? ((gridOrigin.getX() + (x + (stepX > 0 ? 1 : 0)) * voxelSize) - origin.getX()) / dir.getX() : std::numeric_limits<float>::infinity();
    float tMaxY = (dir.getY() != 0) ? ((gridOrigin.getY() + (y + (stepY > 0 ? 1 : 0)) * voxelSize) - origin.getY()) / dir.getY() : std::numeric_limits<float>::infinity();
    float tMaxZ = (dir.getZ() != 0) ? ((gridOrigin.getZ() + (z + (stepZ > 0 ? 1 : 0)) * voxelSize) - origin.getZ()) / dir.getZ() : std::numeric_limits<float>::infinity();

    const int sx = grid.getSizeX();
    const int sy = grid.getSizeY();
    const int sz = grid.getSizeZ();

    float traveled = 0.0f;

    while (traveled <= maxDistance) {
        // Early bounds check with branch prediction optimization
        if (x < 0 || x >= sx || y < 0 || y >= sy || z < 0 || z >= sz) break;

        const size_t idx = grid.indexFromIndices(x, y, z);
        
        // Lock-free atomic operations for better performance
        auto& voxel = grid.voxels[idx];
        const auto& intersections = voxel.getCameraIntersections();
        if (intersections.size() > 0 && intersections.find(cameraId) == intersections.end()) {
            foundIntersection = true;
        }
        voxel.addCameraIntersection(cameraId, intensity);

        // Optimized stepping with fewer branches
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

// VoxelEngine::Scene Implementation
VoxelEngine::Scene::Scene(std::vector<Camera> cams)
    : cameras(std::move(cams)) {
    // Voxel grid will be created dynamically in calculateRayIntersections
}

void VoxelEngine::Scene::addCameras(const std::vector<Camera>& cams) {
    cameras.insert(cameras.end(), cams.begin(), cams.end());
}

void VoxelEngine::Scene::calculateRayIntersections(float topPercentage, float maxDistance) {
    std::cout << "Calculating dynamic scene bounds based on cameras...\n";
    
    // Calculate scene bounds based on camera positions, FOVs, and viewing directions
    auto [minCorner, maxCorner] = calculateSceneBounds();
    
    XYZ size(
        std::ceil(maxCorner.getX() - minCorner.getX()),
        std::ceil(maxCorner.getY() - minCorner.getY()),
        std::ceil(maxCorner.getZ() - minCorner.getZ())
    );
    
    // Create voxel grid with calculated bounds
    voxelGrid.emplace(size, minCorner, 2.0f);
    
    std::cout << "Voxel grid created: " << size.getX() << "x" << size.getY() << "x" << size.getZ() 
              << " (origin: " << minCorner.getX() << "," << minCorner.getY() << "," << minCorner.getZ() << ")\n";
    
    std::cout << "Ultra-fast ray intersection...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    Raycaster::calculateRayIntersectionsUltraFast(*voxelGrid, cameras, maxDistance, topPercentage);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Ray intersection: " << duration.count() << "ms\n";
}

void VoxelEngine::Scene::printSceneInfo() const {
    std::cout << "Voxel Grid: " << voxelGrid->getSize().getX() << " x " 
              << voxelGrid->getSize().getY() << " x " << voxelGrid->getSize().getZ() << "\n";
    
    int totalVoxels = voxelGrid->getSizeX() * voxelGrid->getSizeY() * voxelGrid->getSizeZ();
    int multiCameraVoxels = 0;
    
    for (int i = 0; i < totalVoxels; ++i) {
        if (voxelGrid->voxels[i].getNumCamerasIntersecting() > 1) {
            multiCameraVoxels++;
        }
    }
    
    std::cout << "Multi-camera intersections: " << multiCameraVoxels << "/" << totalVoxels << "\n";
}

void VoxelEngine::Scene::printScene(const std::string& filename) const {
    json sceneData;
    
    sceneData["scene_info"] = {
        {"num_cameras", cameras.size()},
        {"voxel_size_m", voxelGrid->getVoxelSize()},
        {"grid_dimensions", {voxelGrid->getSizeX(), voxelGrid->getSizeY(), voxelGrid->getSizeZ()}},
        {"grid_origin_m", {voxelGrid->getOrigin().getX(), voxelGrid->getOrigin().getY(), voxelGrid->getOrigin().getZ()}},
        {"grid_size_m", {voxelGrid->getSize().getX(), voxelGrid->getSize().getY(), voxelGrid->getSize().getZ()}}
    };

    // Camera data
    json cameraArray = json::array();
    for (size_t i = 0; i < cameras.size(); ++i) {
        const auto& cam = cameras[i];
        const XYZ& camPos = cam.getPosition();
        const Matrix3x3& camRot = cam.getRotation();
        
        json raysArray = json::array();
        std::vector<std::pair<int, int>> samplePixels = {
            {0, 0}, {cam.getImageWidth()-1, 0}, {0, cam.getImageHeight()-1}, 
            {cam.getImageWidth()-1, cam.getImageHeight()-1}, {cam.getImageWidth()/2, cam.getImageHeight()/2}
        };
        
        for (const auto& [px, py] : samplePixels) {
            Ray ray = cam.generateRay(px, py);
            raysArray.push_back({
                {"pixel", {px, py}},
                {"origin_m", {ray.origin.getX(), ray.origin.getY(), ray.origin.getZ()}},
                {"direction", {ray.direction.getX(), ray.direction.getY(), ray.direction.getZ()}}
            });
        }
        
        cameraArray.push_back({
            {"camera_id", i},
            {"position_m", {camPos.getX(), camPos.getY(), camPos.getZ()}},
            {"rotation_matrix", {
                {camRot.m[0][0], camRot.m[0][1], camRot.m[0][2]},
                {camRot.m[1][0], camRot.m[1][1], camRot.m[1][2]},
                {camRot.m[2][0], camRot.m[2][1], camRot.m[2][2]}
            }},
            {"fov_degrees", cam.getFOV()},
            {"resolution", {cam.getImageWidth(), cam.getImageHeight()}},
            {"sample_rays", raysArray}
        });
    }
    sceneData["cameras"] = cameraArray;

    // Voxel data - simplified version for brevity
    json voxelArray = json::array();
    int totalVoxels = voxelGrid->getSizeX() * voxelGrid->getSizeY() * voxelGrid->getSizeZ();
    int nonZeroVoxels = 0;
    
    for (int z = 0; z < voxelGrid->getSizeZ(); ++z) {
        for (int y = 0; y < voxelGrid->getSizeY(); ++y) {
            for (int x = 0; x < voxelGrid->getSizeX(); ++x) {
                const auto& voxel = voxelGrid->voxels[voxelGrid->indexFromIndices(x, y, z)];
                if (voxel.getIntersectionCount() > 0) {
                    nonZeroVoxels++;
                    voxelArray.push_back({
                        {"indices", {x, y, z}},
                        {"intersection_count", voxel.getIntersectionCount()},
                        {"num_cameras", voxel.getNumCamerasIntersecting()}
                    });
                }
            }
        }
    }
    
    sceneData["voxel_data"] = {
        {"total_voxels", totalVoxels},
        {"non_zero_voxels", nonZeroVoxels},
        {"voxels", voxelArray}
    };

    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        throw std::runtime_error("Could not open output file: " + filename);
    }
    
    outFile << sceneData.dump(2);
    outFile.close();
    
    std::cout << "Scene data written to " << filename << " (" << nonZeroVoxels << "/" << totalVoxels << " voxels with intersections)\n";
}

std::pair<XYZ, XYZ> VoxelEngine::Scene::calculateSceneBounds() {
    if (cameras.empty()) {
        // Fallback to default bounds if no cameras
        return { XYZ(-500.0f, -500.0f, 0.0f), XYZ(500.0f, 500.0f, 300.0f) };
    }
    
    float minX = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float minY = std::numeric_limits<float>::max();
    float maxY = std::numeric_limits<float>::lowest();
    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();
    
    // Calculate bounds based on camera positions and their viewing frustums
    for (const auto& cam : cameras) {
        const XYZ& pos = cam.getPosition();
        
        // Include camera position in bounds
        minX = std::min(minX, pos.getX());
        maxX = std::max(maxX, pos.getX());
        minY = std::min(minY, pos.getY());
        maxY = std::max(maxY, pos.getY());
        minZ = std::min(minZ, pos.getZ());
        maxZ = std::max(maxZ, pos.getZ());
        
        // Calculate frustum corners at max distance to expand bounds
        float fovRad = cam.getFOV() * 3.14159265359f / 180.0f;
        float halfFov = fovRad * 0.5f;
        float aspectRatio = static_cast<float>(cam.getImageWidth()) / cam.getImageHeight();
        
        // Estimate viewing distance (could be parameterized)
        float viewDistance = 500.0f;
        
        // Calculate frustum dimensions at view distance
        float frustumHeight = 2.0f * viewDistance * std::tan(halfFov);
        float frustumWidth = frustumHeight * aspectRatio;
        
        // Generate rays to frustum corners and extend bounds
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
    
    // Add some padding to ensure we don't clip important areas
    float padding = 50.0f;
    minX -= padding; maxX += padding;
    minY -= padding; maxY += padding;
    minZ = std::max(0.0f, minZ - padding); // Keep Z >= 0
    maxZ += padding;
    
    return { XYZ(minX, minY, minZ), XYZ(maxX, maxY, maxZ) };
}

