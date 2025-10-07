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
#include "SkyDetector.h"
#include "../third_party/json.hpp"

using json = nlohmann::json;

class Scenario {
public:
    // Constructor that loads scenario from file
    explicit Scenario(const std::string& scenarioPath, int frame);
    
    // Getters
    const std::vector<Camera>& getCameras() const { return cameras; }
    const std::vector<Target>& getTargets() const { return targets; }
    const std::vector<std::string>& getTargetNames() const { return targetNames; }
    const std::string& getScenarioName() const { return scenarioName; }
    
    // Get number of loaded items
    size_t getNumCameras() const { return cameras.size(); }
    size_t getNumTargets() const { return targets.size(); }

private:
    // Member variables
    std::vector<Camera> cameras;
    std::vector<Target> targets;
    std::vector<std::string> targetNames;
    std::string scenarioName;
    SkyDetector skyDetector;
    int frame;

    // Private helper methods
    json loadScenarioFile(const std::string& scenarioPath);
    std::vector<Target> loadTargets(const json& scenarioData);
    std::vector<Camera> loadCameras(const json& scenarioData);
    std::optional<Camera> loadCamera(const json& cameraInfo, SkyDetector& detector);
};