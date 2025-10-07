#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "VoxelEngine.h"
#include "Scenario.h"
#include "VoxelMotion.h"

int main(int argc, char** argv) {
    auto programStart = std::chrono::high_resolution_clock::now();
    std::cout << "=== ARES Voxel Engine ===\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scenario_file>\n";
        return 1;
    }

    std::string scenarioPath = argv[1];
    
    // Time scenario loading (includes image I/O and processing)
    auto scenarioLoadStart = std::chrono::high_resolution_clock::now();
    Scenario scenario1(scenarioPath, 1);
    Scenario scenario2(scenarioPath, 25);
    auto scenarioLoadEnd = std::chrono::high_resolution_clock::now();

    const auto& cameras1 = scenario1.getCameras();
    const auto& cameras2 = scenario2.getCameras();
    
    std::cout << "Loaded " << cameras1.size() << " cameras from scenario: " << scenario1.getScenarioName() << "\n";

    // Create VoxelEngine and scene with cameras
    VoxelEngine voxelEngine;
    VoxelEngine::Scene scene1(cameras1);
    VoxelEngine::Scene scene2(cameras2);

    float topPercentage = 1.0f;  // Process even fewer pixels
    std::cout << "\n=== RAY INTERSECTION PROCESSING ===\n";
    auto rayProcessingStart = std::chrono::high_resolution_clock::now();
    scene1.calculateRayIntersections(topPercentage);
    scene2.calculateRayIntersections(topPercentage);
    auto rayProcessingEnd = std::chrono::high_resolution_clock::now();

    auto voxelGrid1 = scene1.getVoxelGrid();
    auto voxelGrid2 = scene2.getVoxelGrid();

    // Time voxel motion processing (core computation only)
    std::cout << "\n=== VOXEL MOTION PROCESSING ===\n";
    auto voxelMotionStart = std::chrono::high_resolution_clock::now();
    VoxelMotionEngine::VoxelGrid grid1(voxelGrid1);
    VoxelMotionEngine::VoxelGrid grid2(voxelGrid2);
    VoxelMotionEngine motionEngine;
    
    // Process voxel changes but separate I/O timing
    std::string outputPath = "voxel_changes.json";
    auto voxelChanges = motionEngine.computeVoxelChanges(grid1, grid2, 0.05f, 99.0f);
    auto voxelMotionEnd = std::chrono::high_resolution_clock::now();
    
    // Time file I/O separately
    std::cout << "\n=== FILE OUTPUT TIMING ===\n";
    auto fileOutputStart = std::chrono::high_resolution_clock::now();
    motionEngine.saveChangeGrid(voxelChanges, grid1, outputPath);
    auto fileOutputEnd = std::chrono::high_resolution_clock::now();


    // Calculate individual phase timings
    auto scenarioLoadTime = std::chrono::duration_cast<std::chrono::milliseconds>(scenarioLoadEnd - scenarioLoadStart);
    auto rayProcessingTime = std::chrono::duration_cast<std::chrono::milliseconds>(rayProcessingEnd - rayProcessingStart);
    auto voxelMotionTime = std::chrono::duration_cast<std::chrono::milliseconds>(voxelMotionEnd - voxelMotionStart);
    auto fileOutputTime = std::chrono::duration_cast<std::chrono::milliseconds>(fileOutputEnd - fileOutputStart);
    
    // Calculate core processing time (excluding I/O)
    auto coreProcessingTime = rayProcessingTime.count() + voxelMotionTime.count();
    
    auto programEnd = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(programEnd - programStart);
    
    std::cout << "\n=== DETAILED TIMING BREAKDOWN ===\n";
    std::cout << "Scenario loading (images + processing): " << scenarioLoadTime.count() << "ms\n";
    std::cout << "Ray intersection processing: " << rayProcessingTime.count() << "ms\n";
    std::cout << "Voxel motion computation: " << voxelMotionTime.count() << "ms\n";
    std::cout << "File output (voxel_changes.json): " << fileOutputTime.count() << "ms\n";
    std::cout << "\n=== CORE PROCESSING SUMMARY ===\n";
    std::cout << "Pure computation time: " << coreProcessingTime << "ms\n";
    std::cout << "I/O time (load + save): " << (scenarioLoadTime.count() + fileOutputTime.count()) << "ms\n";
    std::cout << "Total runtime: " << totalDuration.count() << "ms (" 
              << std::fixed << std::setprecision(2) << (totalDuration.count() / 1000.0) << "s)\n";
    std::cout << "================================\n";

    return 0;
}