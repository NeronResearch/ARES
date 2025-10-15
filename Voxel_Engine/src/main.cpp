#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include "SparseVoxelEngine.h"
#include "SparseVoxelMotion.h"
#include "UnifiedVoxelExporter.h"
#include "Scenario.h"
#include "InferenceBridge.hpp"


using namespace std;
using namespace std::chrono;

struct Timer {
    string label;
    high_resolution_clock::time_point start;
    Timer(const string &lbl) : label(lbl), start(high_resolution_clock::now()) {}
    ~Timer() {
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        cout << "[TIMER] " << label << ": " << duration << " ms" << endl;
    }
};

int main(int argc, char** argv) {
    Timer programTimer("Total Program Runtime");
    std::cout << "=== ARES Sparse Voxel Engine ===\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scenario_file>\n";
        return 1;
    }


    std::string scenarioPath = argv[1];
    int frame1 = argc > 2 ? std::stoi(argv[2]) : 100;
    int frame2 = argc > 3 ? std::stoi(argv[3]) : 125;
    std::string outputPath = argc > 4 ? argv[4] : "unified_scene_data.json";
    
    Scenario scenario = [&]() {
        Timer scenarioTimer("Scenario Loading");
        return Scenario(scenarioPath, frame1, frame2);
    }();

    const auto& cameras1 = scenario.getCameras1();
    const auto& cameras2 = scenario.getCameras2();
    
    // Extract target information for both frames
    const auto& targets1 = scenario.getTargets1();
    const auto& targets2 = scenario.getTargets2();
    const auto& targetNames2 = scenario.getTargetNames();
    Target target1 = targets1.empty() ? Target(0, -150, 100) : targets1[0];
    Target target2 = targets2.empty() ? Target(0, -150, 100) : targets2[0];
    
    float topPercentage = .01f;
    
    std::cout << "\n=== SPARSE VOXEL ENGINE ===\n";
    
    // Sparse implementation
    SparseVoxelEngine sparseEngine;
    SparseVoxelEngine::Scene sparseScene1(cameras1);
    SparseVoxelEngine::Scene sparseScene2(cameras2);

    {
        Timer rayTimer("Ray Intersection Processing");
        sparseScene1.calculateRayIntersections(topPercentage);
        sparseScene2.calculateRayIntersections(topPercentage);
    }

    auto& sparseVoxelGrid1 = sparseScene1.getVoxelGrid();
    auto& sparseVoxelGrid2 = sparseScene2.getVoxelGrid();

    // Enhanced motion processing with sparse grids
    std::vector<MotionTypes::ChangeVoxel> sparseVoxelChanges;
    {
        Timer motionTimer("Motion Analysis Processing");
        SparseVoxelMotionExt::SparseVoxelMotionEngine sparseMotionEngine;
        sparseVoxelChanges = sparseMotionEngine.computeSparseVoxelChanges(sparseVoxelGrid1, sparseVoxelGrid2, 0.01f, 99.9f);
    }
    
    // Also save legacy format for compatibility
    // sparseMotionEngine.saveSparseChangeGrid(sparseVoxelChanges, sparseVoxelGrid1, "voxel_changes.json");

    std::cout << "Sparse processing complete: " << sparseVoxelChanges.size() << " changes detected\n";

    // Export unified JSON data
    std::cout << "\n=== CLEAN JSON OUTPUT (ML/Matlab Ready) ===\n";
    {
        Timer exportTimer("JSON Export");
        UnifiedVoxelExporter::exportUnifiedScene(
            sparseVoxelGrid1, sparseVoxelGrid2, sparseVoxelChanges, 
            cameras1, outputPath, targets2, targetNames2, frame2
        );
    }

    // Performance analysis
    size_t totalVoxels = sparseVoxelGrid1.getMaxVoxelCount();
    size_t activeVoxels = sparseVoxelGrid1.getActiveVoxelCount();
    size_t memoryUsageMb = (activeVoxels * (sizeof(Voxel) + sizeof(size_t))) / (1024 * 1024);
    float sparsityRatio = sparseVoxelGrid1.getSparsityRatio();

    std::cout << "\n=== PERFORMANCE METRICS ===\n";
    std::cout << "Memory usage: " << memoryUsageMb << "MB\n";
    std::cout << "Sparsity: " << std::fixed << std::setprecision(2) << (sparsityRatio * 100) << "% empty space\n";
    std::cout << "Active voxels: " << activeVoxels << " / " << totalVoxels << "\n";

    // Sparse grid analysis
    std::cout << "\n=== SPARSE GRID ANALYSIS ===\n";
    sparseScene1.printSceneInfo();
    
    return 0;
}