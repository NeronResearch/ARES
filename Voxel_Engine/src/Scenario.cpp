#include "Scenario.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <future>
#include <algorithm>
#include "../third_party/stb_image.h"
#include "../third_party/stb_image_write.h"

Scenario::Scenario(const std::string& scenarioPath, int frame) : frame(frame) {
    json scenarioData = loadScenarioFile(scenarioPath);
    targets = loadTargets(scenarioData);
    cameras = loadCameras(scenarioData);
}

json Scenario::loadScenarioFile(const std::string& scenarioPath) {
    std::ifstream scenarioFile(scenarioPath);
    if (!scenarioFile.is_open()) {
        throw std::runtime_error("Could not open scenario file: " + scenarioPath);
    }
    
    json scenarioData;
    scenarioFile >> scenarioData;
    scenarioFile.close();
    
    if (scenarioData.contains("scenario_name")) {
        scenarioName = scenarioData["scenario_name"].get<std::string>();
        std::cout << "Loaded scenario: " << scenarioName << "\n";
    } else {
        scenarioName = "Unknown Scenario";
    }
    
    return scenarioData;
}

std::vector<Target> Scenario::loadTargets(const json& scenarioData) {
    std::vector<Target> targets;
    if (scenarioData.contains("targets")) {
        for (const auto& targetInfo : scenarioData["targets"]) {
            std::string targetFile = targetInfo["file"].get<std::string>();
            std::ifstream targetFileStream(targetFile);
            if (!targetFileStream.is_open()) continue;

            json targetData;
            targetFileStream >> targetData;
            targetFileStream.close();

            if (!targetData["frames"].empty()) {
                // Find the correct frame data for this specific frame number
                json targetFrameData;
                bool frameFound = false;
                
                for (const auto& frameData : targetData["frames"]) {
                    if (frameData.contains("frame") && frameData["frame"].get<int>() == frame) {
                        targetFrameData = frameData;
                        frameFound = true;
                        break;
                    }
                }
                
                // If specific frame not found, use the closest available frame or first frame as fallback
                if (!frameFound) {
                    std::cout << "Warning: Frame " << frame << " not found for target, using first available frame\n";
                }
                
                XYZ targetPos(
                    targetFrameData["position_m"][0].get<float>(),
                    targetFrameData["position_m"][1].get<float>(),
                    targetFrameData["position_m"][2].get<float>()
                );
                targets.emplace_back(targetPos);
                std::string targetName = targetInfo["name"].get<std::string>();
                targetNames.emplace_back(targetName);
                std::cout << "Target: " << targetName 
                          << " at (" << targetPos.getX() << ", " << targetPos.getY() << ", " << targetPos.getZ() << ") for frame " << frame << "\n";
            }
        }
    }

    if (targets.empty()) {
        targets.emplace_back(XYZ(0, -150, 100));
        std::cout << "Default target at (0, -150, 100) for frame " << frame << "\n";
    }
    
    return targets;
}

std::vector<Camera> Scenario::loadCameras(const json& scenarioData) {
    if (!scenarioData.contains("cameras")) return {};

    const auto& camerasJson = scenarioData["cameras"];
    std::vector<Camera> cameras;
    cameras.reserve(camerasJson.size());
    
    std::cout << "Loading " << camerasJson.size() << " cameras...\n";
    
    // Timing for overall image processing
    auto totalProcessingStart = std::chrono::high_resolution_clock::now();

    std::vector<std::future<std::optional<Camera>>> futures;
    futures.reserve(camerasJson.size());

    for (const auto& cameraInfo : camerasJson) {
        futures.emplace_back(std::async(std::launch::async, [this](const json& info) -> std::optional<Camera> {
            return loadCamera(info, skyDetector);
        }, cameraInfo));
    }

    for (auto& future : futures) {
        auto cameraOpt = future.get();
        if (cameraOpt.has_value()) {
            cameras.push_back(std::move(cameraOpt.value()));
        }
    }
    
    auto totalProcessingEnd = std::chrono::high_resolution_clock::now();
    auto totalProcessingTime = std::chrono::duration_cast<std::chrono::milliseconds>(totalProcessingEnd - totalProcessingStart);
    
    std::cout << "Loaded " << cameras.size() << " cameras" << std::endl;
    std::cout << "=== IMAGE PROCESSING TIMING ===\n";
    std::cout << "Total image loading & processing: " << totalProcessingTime.count() << "ms\n";
    std::cout << "==============================\n";
    
    return cameras;
}

std::optional<Camera> Scenario::loadCamera(const json& cameraInfo, SkyDetector& detector) {
    static std::atomic<long long> totalImageLoadTime{0};
    static std::atomic<long long> totalGrayscaleTime{0};
    static std::atomic<long long> totalMotionTime{0};
    static std::atomic<long long> totalSkyDetectionTime{0};
    
    std::string framesDir = cameraInfo["frames_dir"].get<std::string>();
    std::string cameraName = cameraInfo["name"].get<std::string>();
    
    // Format frame numbers with leading zeros (4 digits)
    auto formatFrame = [](int frameNum) -> std::string {
        std::string frameStr = std::to_string(frameNum);
        while (frameStr.length() < 4) {
            frameStr = "0" + frameStr;
        }
        return frameStr;
    };
    
    std::string frameBuffer = formatFrame(frame);
    
    // For img2: if frame is 0, use frame 0, otherwise use previous frame
    int prevFrame = (frame == 0) ? 0 : frame - 1;
    std::string prevFrameBuffer = formatFrame(prevFrame);

    std::cout << "frame buff " << frameBuffer << " prev " << prevFrameBuffer << std::endl;
    
    std::string img1 = framesDir + "/" + frameBuffer + ".jpg";
    std::string img2 = framesDir + "/" + prevFrameBuffer + ".jpg";
    std::string jsonPath = framesDir + "/" + frameBuffer + ".json";

    std::cout << "Loading camera: " << cameraName << " with images: " << img1 << " and " << img2 << std::endl;

    // Time image loading
    auto imageLoadStart = std::chrono::high_resolution_clock::now();
    int imgW1, imgH1, imgW2, imgH2, channels1, channels2;
    auto image1 = Camera::loadImage(img1, imgW1, imgH1, channels1);
    auto image2 = Camera::loadImage(img2, imgW2, imgH2, channels2);
    auto imageLoadEnd = std::chrono::high_resolution_clock::now();
    totalImageLoadTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(imageLoadEnd - imageLoadStart).count());

    // Debug: Check image loading results
    if (image1.empty()) {
        std::cout << "ERROR: Failed to load image1: " << img1 << std::endl;
        return std::nullopt;
    }
    if (image2.empty()) {
        std::cout << "ERROR: Failed to load image2: " << img2 << std::endl;
        return std::nullopt;
    }
    if (imgW1 != imgW2 || imgH1 != imgH2) {
        std::cout << "ERROR: Image dimension mismatch - img1: " << imgW1 << "x" << imgH1 << ", img2: " << imgW2 << "x" << imgH2 << std::endl;
        return std::nullopt;
    }
    
    std::cout << "Successfully loaded images: " << imgW1 << "x" << imgH1 << " channels: " << channels1 << "/" << channels2 << std::endl;

    // Time grayscale conversion
    auto grayscaleStart = std::chrono::high_resolution_clock::now();
    auto image1data = Camera::convertToGrayscale(image1, imgW1, imgH1, channels1);
    auto image2data = Camera::convertToGrayscale(image2, imgW2, imgH2, channels2);
    auto grayscaleEnd = std::chrono::high_resolution_clock::now();
    totalGrayscaleTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(grayscaleEnd - grayscaleStart).count());

    // Time motion map computation
    auto motionStart = std::chrono::high_resolution_clock::now();
    auto motionMap = Camera::computeMotionMap(image1data, image2data, imgW1, imgH1);
    auto motionEnd = std::chrono::high_resolution_clock::now();
    totalMotionTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(motionEnd - motionStart).count());

    // Time sky detection
    auto skyStart = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> maskData = detector.detectSkyMask(image1, imgW1, imgH1, channels1);
    auto skyEnd = std::chrono::high_resolution_clock::now();
    totalSkyDetectionTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(skyEnd - skyStart).count());
 
    // Apply mask to motion map
    std::vector<uint8_t> maskedMotionMap(imgW1 * imgH1, 0);
    
    if (maskData.size() == 640 * 480 && (imgW1 != 640 || imgH1 != 480)) {
        float scaleX = 640.0f / imgW1;
        float scaleY = 480.0f / imgH1;
        
        #pragma omp parallel for collapse(2)
        for (int y = 0; y < imgH1; ++y) {
            for (int x = 0; x < imgW1; ++x) {
                float maskX = x * scaleX;
                float maskY = y * scaleY;
                
                int maskXInt = static_cast<int>(maskX + 0.5f);
                int maskYInt = static_cast<int>(maskY + 0.5f);
                
                maskXInt = std::max(0, std::min(639, maskXInt));
                maskYInt = std::max(0, std::min(479, maskYInt));
                
                int motionIdx = y * imgW1 + x;
                int maskIdx = maskYInt * 640 + maskXInt;
                
                if (maskData[maskIdx] == 0) {
                    maskedMotionMap[motionIdx] = 0;
                } else {
                    maskedMotionMap[motionIdx] = motionMap[motionIdx];
                }
            }
        }
    } else if (maskData.size() == imgW1 * imgH1) {
        #pragma omp parallel for
        for (int i = 0; i < imgW1 * imgH1; ++i) {
            if (maskData[i] == 0) {
                maskedMotionMap[i] = 0;
            } else {
                maskedMotionMap[i] = motionMap[i];
            }
        }
    } else {
        maskedMotionMap = motionMap;
    }
    
    motionMap = std::move(maskedMotionMap);

    std::cout << "frame no " << std::to_string(frame) << std::endl;

    // std::string outputPath = "motion_map_masked_" + cameraName + std::to_string(frame) + ".jpg";
    // std::cout << "Saving motion map masked to: " << outputPath << std::endl;
    // int result = stbi_write_jpg(outputPath.c_str(), imgW1, imgH1, 1, motionMap.data(), 90);

    std::ifstream jsonFile(jsonPath);
    if (!jsonFile.is_open()) {
        std::cout << "ERROR: Failed to open metadata file: " << jsonPath << std::endl;
        return std::nullopt;
    }
    
    json meta;
    jsonFile >> meta;
    jsonFile.close();

    int fov = meta.contains("fov_deg") ? static_cast<int>(meta["fov_deg"].get<float>()) : 140;

    XYZ position(
        meta["position_m"][0].get<float>(),
        meta["position_m"][1].get<float>(),
        meta["position_m"][2].get<float>()
    );

    Matrix3x3 rotation;
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            rotation.m[r][c] = meta["rotation_matrix"][r][c].get<float>();
        }
    }

    float sensorSize = 36.0f;

    return Camera(position, rotation, sensorSize, fov, imgW1, imgH1, std::move(motionMap));
}
