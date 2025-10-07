#pragma once
#include "XYZ.h"
#include <unordered_map>

class Voxel {
public:
    XYZ getPosition() const { return position; }
    void setPosition(const XYZ& pos) { position = pos; }
    float getIntersectionCount() const { return intersectionCount; }
    
    void addCameraIntersection(int cameraId, float brightness) {
        auto it = cameraIntersections.find(cameraId);
        if (it != cameraIntersections.end()) {
            it->second += brightness;
        } else {
            cameraIntersections[cameraId] = brightness;
        }
    }
    
    void finalizeIntersections() {
        if (cameraIntersections.size() > 1) {
            intersectionCount = calculateIntersectionScore();
        } else {
            intersectionCount = 0.0f;
        }
    }
    
    int getNumCamerasIntersecting() const { return static_cast<int>(cameraIntersections.size()); }
    const std::unordered_map<int, float>& getCameraIntersections() const { return cameraIntersections; }
    
private:
    XYZ position;
    std::unordered_map<int, float> cameraIntersections;
    float intersectionCount = 0.0f;
    
    float calculateIntersectionScore() const {
        if (cameraIntersections.size() < 2) return 0.0f;
        
        float score = 1.0f;
        for (const auto& [camId, brightness] : cameraIntersections) {
            score *= brightness;
        }
        return score * cameraIntersections.size();
    }
};
