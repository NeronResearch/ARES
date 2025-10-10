#include "Scenario.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <future>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/stb_image_write.h"

Scenario::Scenario(const std::string& scenarioPath, int frame1, int frame2) : frame1(frame1), frame2(frame2) {
    json scenarioData = loadScenarioFile(scenarioPath);
    
    // Load targets for both frames
    targets1 = loadTargets(scenarioData, frame1);
    targets2 = loadTargets(scenarioData, frame2);
    
    // Load cameras for both frames (this will process motion between consecutive frames for each)
    cameras1 = loadCameras(scenarioData, frame1);
    cameras2 = loadCameras(scenarioData, frame2);
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

std::vector<Target> Scenario::loadTargets(const json& scenarioData, int frame) {
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

                bool visible = false;
                if (targetFrameData.contains("is_detected")) {
                    visible = targetFrameData["is_detected"].get<bool>();
                }
                targets.emplace_back(targetPos, visible);
                std::string targetName = targetInfo["name"].get<std::string>();
                // Only add target names once (for the first frame)
                if (frame == frame1) {
                    targetNames.emplace_back(targetName);
                }
                std::cout << "Target: " << targetName 
                          << " at (" << targetPos.getX() << ", " << targetPos.getY() << ", " << targetPos.getZ() << ") for frame " << frame << "\n";
            }
        }
    }
    
    return targets;
}

std::vector<Camera> Scenario::loadCameras(const json& scenarioData, int frame) {
    if (!scenarioData.contains("cameras")) return {};

    const auto& camerasJson = scenarioData["cameras"];
    std::vector<Camera> cameras;
    cameras.reserve(camerasJson.size());
    
    std::cout << "Loading " << camerasJson.size() << " cameras for frame " << frame << "...\n";
    
    // Timing for overall image processing
    auto totalProcessingStart = std::chrono::high_resolution_clock::now();

    std::vector<std::future<std::optional<Camera>>> futures;
    futures.reserve(camerasJson.size());

    for (const auto& cameraInfo : camerasJson) {
        futures.emplace_back(std::async(std::launch::async, [this, frame](const json& info) -> std::optional<Camera> {
            return loadCamera(info, frame);
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
    
    std::cout << "Loaded " << cameras.size() << " cameras for frame " << frame << std::endl;
    std::cout << "=== IMAGE PROCESSING TIMING (Frame " << frame << ") ===\n";
    std::cout << "Total image loading & processing: " << totalProcessingTime.count() << "ms\n";
    std::cout << "==============================\n";
    
    return cameras;
}

std::vector<uint8_t> Scenario::applyMask(const std::vector<uint8_t>& image,
                                         const std::vector<uint8_t>& mask,
                                         int width, int height) {
    size_t totalPixels = static_cast<size_t>(width) * height;
    std::vector<uint8_t> maskedImage(totalPixels, 0);

    if (image.size() < totalPixels) {
        std::cerr << "ERROR: Image size (" << image.size()
                  << ") is smaller than expected (" << totalPixels << ")\n";
        return maskedImage;
    }

    if (mask.size() < totalPixels) {
        std::cerr << "WARNING: Mask size (" << mask.size()
                  << ") does not match expected (" << totalPixels
                  << "). Skipping masking and returning original image.\n";
        return image; // fallback: just return the unmasked image
    }

    #pragma omp parallel for simd
    for (int i = 0; i < static_cast<int>(totalPixels); ++i) {
        maskedImage[i] = (mask[i] > 128) ? image[i] : 0;
    }

    return maskedImage;
}


std::optional<Camera> Scenario::loadCamera(const json& cameraInfo, int frame) {
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

    
    std::string img1 = framesDir + "/" + frameBuffer + ".jpg";
    std::string img2 = framesDir + "/" + prevFrameBuffer + ".jpg";
    std::string skyMaskPath = framesDir + "/sky_mask.jpg";
    std::string jsonPath = framesDir + "/" + frameBuffer + ".json";


    // Time image loading
    auto imageLoadStart = std::chrono::high_resolution_clock::now();
    int imgW1, imgH1, imgW2, imgH2, channels1, channels2, channelsSky;
    auto image1 = Camera::loadImage(img1, imgW1, imgH1, channels1);
    auto image2 = Camera::loadImage(img2, imgW2, imgH2, channels2);
    auto skyMask = Camera::loadImage(skyMaskPath, imgW1, imgH1, channelsSky);

    stbi_write_png("skyMask.png", imgW1, imgH1, 1, skyMask.data(), imgW1);
    

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
    
    // Time grayscale conversion
    auto grayscaleStart = std::chrono::high_resolution_clock::now();
    auto image1data = Camera::convertToGrayscale(image1, imgW1, imgH1, channels1);
    auto image2data = Camera::convertToGrayscale(image2, imgW2, imgH2, channels2);

    auto grayscaleEnd = std::chrono::high_resolution_clock::now();
    totalGrayscaleTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(grayscaleEnd - grayscaleStart).count());

    // Time motion map computation
    auto motionStart = std::chrono::high_resolution_clock::now();
    auto motionMap = Camera::computeMotionMap(image1data, image2data, imgW1, imgH1);

    std::string motionMapPath = "motionMap" + cameraName + ".png";
    stbi_write_png(motionMapPath.c_str(), imgW1, imgH1, 1, motionMap.data(), imgW1);
    auto motionEnd = std::chrono::high_resolution_clock::now();
    totalMotionTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(motionEnd - motionStart).count());

    // Time sky detection
    auto skyStart = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> maskData = skyMask;
    auto skyEnd = std::chrono::high_resolution_clock::now();
    totalSkyDetectionTime.fetch_add(std::chrono::duration_cast<std::chrono::microseconds>(skyEnd - skyStart).count());
 
    // Apply mask to motion map
    std::vector<uint8_t> maskedMotionMap = applyMask(motionMap, maskData, imgW1, imgH1);
    
    motionMap = std::move(maskedMotionMap);

    std::string maskedMotionMapPath = "maskedMotionMap" + cameraName + ".png";

    stbi_write_png(maskedMotionMapPath.c_str(), imgW1, imgH1, 1, motionMap.data(), imgW1);

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
