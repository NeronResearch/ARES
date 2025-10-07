#pragma once

#include <vector>
#include <string>
#include <optional>
#include <unordered_map>
#include <future>
#include <atomic>
#include <cstdint>
#include <utility>
#include "XYZ.h"
#include "Matrix3x3.h"
#include "Voxel.h"
#include "VoxelGrid.h"
#include "Camera.h"
class VoxelEngine {
public:
    // Nested structures and classes
    struct LLA {
        double lat;
        double lon;
        double alt;
    };

    class Raycaster {
    public:
        Raycaster(const Camera& cam, const VoxelGrid& grid);

        // Static ray intersection calculation
        static void calculateRayIntersectionsUltraFast(VoxelGrid& voxelGrid, 
                                                      const std::vector<Camera>& cameras, 
                                                      float maxDistance = 1000.0f, 
                                                      float topPercentage = 5.0f);

    private:
        static bool CastRayAndAccumulate(VoxelGrid& grid, 
                                        const XYZ& origin, 
                                        const XYZ& dir, 
                                        float maxDistance, 
                                        float intensity, 
                                        int cameraId);

        const Camera& camera;
        const VoxelGrid& voxelGrid;
    };

    class Scene {
    public:
        Scene(std::vector<Camera> cams);
        
        // Method to add cameras after construction
        void addCameras(const std::vector<Camera>& cams);

        void calculateRayIntersections(float topPercentage = 5.0f, float maxDistance = 1000.0f);
        void printSceneInfo() const;
        void printScene(const std::string& filename) const;
        VoxelGrid getVoxelGrid() const { return voxelGrid.value(); }

    private:
        std::pair<XYZ, XYZ> calculateSceneBounds();

        std::vector<Camera> cameras;
        std::optional<VoxelGrid> voxelGrid;
    };

public:
    // Constructor
    VoxelEngine() = default;
};
