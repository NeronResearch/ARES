#pragma once

#include <vector>
#include <string>
#include <optional>
#include <future>
#include <chrono>
#include <atomic>
#include <mutex>
#include "Camera.h"
#include "Target.h"
#include "../third_party/json.hpp"

using json = nlohmann::json;

class Scenario {
public:
    // Constructor that loads scenario from file for multiple frames
    explicit Scenario(const std::string& scenarioPath, int frame1, int frame2);
    
    // Getters for frame 1
    const std::vector<Camera>& getCameras1() const { return cameras1; }
    const std::vector<Target>& getTargets1() const { return targets1; }
    
    // Getters for frame 2
    const std::vector<Camera>& getCameras2() const { return cameras2; }
    const std::vector<Target>& getTargets2() const { return targets2; }
    
    // Common getters
    const std::vector<std::string>& getTargetNames() const { return targetNames; }
    const std::string& getScenarioName() const { return scenarioName; }
    
    // Get number of loaded items
    size_t getNumCameras() const { return cameras1.size(); }
    size_t getNumTargets() const { return targets1.size(); }

private:
    // Member variables for both frames
    std::vector<Camera> cameras1;
    std::vector<Camera> cameras2;
    std::vector<Target> targets1;
    std::vector<Target> targets2;
    std::vector<std::string> targetNames;
    std::string scenarioName;
    int frame1, frame2;

    // Private helper methods
    json loadScenarioFile(const std::string& scenarioPath);
    std::vector<Target> loadTargets(const json& scenarioData, int frame);
    std::vector<uint8_t> applyMask(const std::vector<uint8_t>& image, const std::vector<uint8_t>& mask, int width, int height);
    std::vector<Camera> loadCameras(const json& scenarioData, int frame);
    std::optional<Camera> loadCamera(const json& cameraInfo, int frame);
};