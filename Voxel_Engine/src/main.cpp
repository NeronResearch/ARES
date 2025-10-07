#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <optional>
#include "VoxelEngine.h"
#include "SparseVoxelEngine.h"
#include "SparseVoxelMotion.h"
#include "UnifiedVoxelExporter.h"
#include "Scenario.h"
#include "arm_benchmark.h"

/**
 * @brief Performance comparison between dense and sparse implementations
 */
struct PerformanceComparison {
    struct Metrics {
        std::chrono::milliseconds ray_processing_time;
        std::chrono::milliseconds motion_processing_time;
        size_t memory_usage_mb;
        size_t active_voxels;
        size_t total_voxels;
        float sparsity_ratio;
    };
    
    Metrics dense;
    Metrics sparse;
    
    void printComparison() const {
        std::cout << "\n=== PERFORMANCE COMPARISON ===\n";
        std::cout << std::fixed << std::setprecision(2);
        
        std::cout << "Implementation    | Ray Time | Motion Time | Memory  | Active/Total Voxels | Sparsity\n";
        std::cout << "-----------------|----------|-------------|---------|---------------------|----------\n";
        
        std::cout << "Dense (Original) | " << std::setw(7) << dense.ray_processing_time.count() << "ms | "
                  << std::setw(10) << dense.motion_processing_time.count() << "ms | "
                  << std::setw(6) << dense.memory_usage_mb << "MB | "
                  << std::setw(19) << (std::to_string(dense.active_voxels) + "/" + std::to_string(dense.total_voxels)) << " | "
                  << std::setw(7) << (dense.sparsity_ratio * 100) << "%\n";
        
        std::cout << "Sparse (New)     | " << std::setw(7) << sparse.ray_processing_time.count() << "ms | "
                  << std::setw(10) << sparse.motion_processing_time.count() << "ms | "
                  << std::setw(6) << sparse.memory_usage_mb << "MB | "
                  << std::setw(19) << (std::to_string(sparse.active_voxels) + "/" + std::to_string(sparse.total_voxels)) << " | "
                  << std::setw(7) << (sparse.sparsity_ratio * 100) << "%\n";
        
        std::cout << "\nImprovements:\n";
        
        float ray_speedup = static_cast<float>(dense.ray_processing_time.count()) / sparse.ray_processing_time.count();
        float motion_speedup = static_cast<float>(dense.motion_processing_time.count()) / sparse.motion_processing_time.count();
        float memory_savings = 1.0f - (static_cast<float>(sparse.memory_usage_mb) / dense.memory_usage_mb);
        
        std::cout << "- Ray Processing: " << ray_speedup << "x speedup\n";
        std::cout << "- Motion Analysis: " << motion_speedup << "x speedup\n";
        std::cout << "- Memory Usage: " << (memory_savings * 100) << "% savings\n";
        std::cout << "- Sparsity Gain: " << ((sparse.sparsity_ratio - dense.sparsity_ratio) * 100) << "% more efficient\n";
    }
};

int main(int argc, char** argv) {
    auto programStart = std::chrono::high_resolution_clock::now();
    std::cout << "=== ARES Sparse Voxel Engine Demonstration ===\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scenario_file> [--sparse-only]\n";
        std::cerr << "  --sparse-only: Run only sparse implementation (faster for testing)\n";
        return 1;
    }

    std::string scenarioPath = argv[1];
    bool denseOnly = (argc > 2 && std::string(argv[2]) == "--dense");
    // Default to sparse processing unless --dense is specified
    
    // Time scenario loading
    auto scenarioLoadStart = std::chrono::high_resolution_clock::now();
    Scenario scenario1(scenarioPath, 1);
    Scenario scenario2(scenarioPath, 25);
    auto scenarioLoadEnd = std::chrono::high_resolution_clock::now();

    const auto& cameras1 = scenario1.getCameras();
    const auto& cameras2 = scenario2.getCameras();
    
    // Extract target information for both frames
    const auto& targets1 = scenario1.getTargets();
    const auto& targets2 = scenario2.getTargets();
    const auto& targetNames2 = scenario2.getTargetNames();
    Target target1 = targets1.empty() ? Target(0, -150, 100) : targets1[0];
    Target target2 = targets2.empty() ? Target(0, -150, 100) : targets2[0];
    
    std::cout << "Loaded " << cameras1.size() << " cameras from scenario: " << scenario1.getScenarioName() << "\n";
    std::cout << "Frame 1 target: (" << target1.getCurrentPosition().getX() << ", " << target1.getCurrentPosition().getY() << ", " << target1.getCurrentPosition().getZ() << ")\n";
    std::cout << "Frame 25 target: (" << target2.getCurrentPosition().getX() << ", " << target2.getCurrentPosition().getY() << ", " << target2.getCurrentPosition().getZ() << ")\n";

    PerformanceComparison comparison;
    float topPercentage = 1.0f;
    float maxDistance = 100.0f;
    
    // Dense processing variables (used for export)
    std::optional<VoxelMotionEngine::VoxelGrid> denseGrid1Opt, denseGrid2Opt;
    std::vector<VoxelMotionEngine::ChangeVoxel> denseVoxelChanges;

    // HARDWARE-ADAPTIVE PERFORMANCE OPTIMIZATION
    auto systemInfo = ARMBenchmark::SystemInfo::detect();
    systemInfo.print();
    
    // Run memory bandwidth test for cache optimization guidance
    ARMBenchmark::MemoryBandwidthTest::runTest();
    
    // Get performance recommendations based on detected hardware
    auto perfRec = ARMBenchmark::PerformanceRecommendations::generate(systemInfo);
    perfRec.print();
    
    // Apply hardware-specific optimizations
    bool preferDense = denseOnly;
    
    // Override parameters based on hardware capabilities
    if (systemInfo.is_low_power) {
        topPercentage = perfRec.recommended_top_percentage;
        maxDistance = perfRec.recommended_max_distance;
        
        std::cout << "\nApplied low-power device optimizations:\n";
        std::cout << "- Ray percentage: " << topPercentage << "%\n";
        std::cout << "- Max distance: " << maxDistance << "m\n";
    }
    
    std::cout << "\n=== ALGORITHM SELECTION ===\n";
    if (denseOnly) {
        std::cout << "Force dense mode (--dense flag)\n";
        std::cout << "Selected algorithm: DENSE (user specified)\n";
    } else {
        std::cout << "Hardware-adaptive sparse mode\n";
        std::cout << "Selected algorithm: SPARSE (";
        if (systemInfo.is_arm) std::cout << "ARM-optimized";
        if (systemInfo.has_neon) std::cout << " + NEON SIMD";
        if (systemInfo.is_low_power) std::cout << " + Low-power tweaks";
        std::cout << ")\n";
    }

    if (preferDense) {
        std::cout << "\n=== DENSE VOXEL ENGINE (ORIGINAL) ===\n";
        
        // Original dense implementation
        VoxelEngine voxelEngine;
        VoxelEngine::Scene denseScene1(cameras1);
        VoxelEngine::Scene denseScene2(cameras2);

        auto denseRayStart = std::chrono::high_resolution_clock::now();
        denseScene1.calculateRayIntersections(topPercentage, maxDistance);
        denseScene2.calculateRayIntersections(topPercentage, maxDistance);
        auto denseRayEnd = std::chrono::high_resolution_clock::now();

        // denseScene1.printScene("dense_scene_1.json");
        // denseScene2.printScene("dense_scene_2.json");

        auto denseVoxelGrid1 = denseScene1.getVoxelGrid();
        auto denseVoxelGrid2 = denseScene2.getVoxelGrid();

        auto denseMotionStart = std::chrono::high_resolution_clock::now();
        denseGrid1Opt.emplace(denseVoxelGrid1);
        denseGrid2Opt.emplace(denseVoxelGrid2);
        VoxelMotionEngine denseMotionEngine;
        
        denseVoxelChanges = denseMotionEngine.computeVoxelChanges(*denseGrid1Opt, *denseGrid2Opt, 0.01f, 99.0f);
        auto denseMotionEnd = std::chrono::high_resolution_clock::now();

        // denseMotionEngine.saveChangeGrid(denseVoxelChanges, *denseGrid1Opt, "dense_voxel_changes.json");

        // Record dense metrics
        comparison.dense.ray_processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(denseRayEnd - denseRayStart);
        comparison.dense.motion_processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(denseMotionEnd - denseMotionStart);
        comparison.dense.total_voxels = denseVoxelGrid1.getSizeX() * denseVoxelGrid1.getSizeY() * denseVoxelGrid1.getSizeZ();
        comparison.dense.active_voxels = denseGrid1Opt->voxels.size();
        comparison.dense.memory_usage_mb = (comparison.dense.total_voxels * sizeof(Voxel)) / (1024 * 1024);
        comparison.dense.sparsity_ratio = 1.0f - (static_cast<float>(comparison.dense.active_voxels) / comparison.dense.total_voxels);

        std::cout << "Dense processing complete: " << denseVoxelChanges.size() << " changes detected\n";
    }

    // Declare sparse objects outside conditional for scope access
    SparseVoxelEngine sparseEngine;
    std::optional<SparseVoxelEngine::Scene> sparseScene1Opt, sparseScene2Opt;
    std::vector<VoxelMotionEngine::ChangeVoxel> sparseVoxelChanges;
    
    if (!preferDense) {
        std::cout << "\n=== SPARSE VOXEL ENGINE (NEW) ===\n";
        
        // New sparse implementation
        sparseScene1Opt.emplace(cameras1);
        sparseScene2Opt.emplace(cameras2);
        auto& sparseScene1 = *sparseScene1Opt;
        auto& sparseScene2 = *sparseScene2Opt;

        auto sparseRayStart = std::chrono::high_resolution_clock::now();
        sparseScene1.calculateRayIntersections(topPercentage);
        sparseScene2.calculateRayIntersections(topPercentage);
        auto sparseRayEnd = std::chrono::high_resolution_clock::now();

        auto& sparseVoxelGrid1 = sparseScene1.getVoxelGrid();
        auto& sparseVoxelGrid2 = sparseScene2.getVoxelGrid();

        // Enhanced motion processing with sparse grids
        auto sparseMotionStart = std::chrono::high_resolution_clock::now();
        SparseVoxelMotionExt::SparseVoxelMotionEngine sparseMotionEngine;
        sparseVoxelChanges = sparseMotionEngine.computeSparseVoxelChanges(sparseVoxelGrid1, sparseVoxelGrid2, 0.05f, 99.0f);
        auto sparseMotionEnd = std::chrono::high_resolution_clock::now();
        
        // Also save legacy format for compatibility
        // sparseMotionEngine.saveSparseChangeGrid(sparseVoxelChanges, sparseVoxelGrid1, "voxel_changes.json");

        // Record sparse metrics
    comparison.sparse.ray_processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(sparseRayEnd - sparseRayStart);
    comparison.sparse.motion_processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(sparseMotionEnd - sparseMotionStart);
    comparison.sparse.total_voxels = sparseVoxelGrid1.getMaxVoxelCount();
    comparison.sparse.active_voxels = sparseVoxelGrid1.getActiveVoxelCount();
    comparison.sparse.memory_usage_mb = (comparison.sparse.active_voxels * (sizeof(Voxel) + sizeof(size_t))) / (1024 * 1024);
    comparison.sparse.sparsity_ratio = sparseVoxelGrid1.getSparsityRatio();

        std::cout << "Sparse processing complete: " << sparseVoxelChanges.size() << " changes detected\n";
    } else {
        std::cout << "\n=== SPARSE VOXEL ENGINE (SKIPPED) ===\n";
        std::cout << "Sparse engine skipped - adaptive algorithm selected dense for better performance\n";
        
        // Use dense results for sparse metrics to show comparison
        comparison.sparse.ray_processing_time = comparison.dense.ray_processing_time;
        comparison.sparse.motion_processing_time = comparison.dense.motion_processing_time;
        comparison.sparse.total_voxels = comparison.dense.total_voxels;
        comparison.sparse.active_voxels = comparison.dense.active_voxels;
        comparison.sparse.memory_usage_mb = comparison.dense.memory_usage_mb;
        comparison.sparse.sparsity_ratio = comparison.dense.sparsity_ratio;
    }

    // Export unified JSON data (always run regardless of algorithm selection)
    std::cout << "\n=== CLEAN JSON OUTPUT (ML/Matlab Ready) ===\n";
    if (sparseScene1Opt.has_value() && sparseScene2Opt.has_value()) {
        // We have sparse data - use the sparse exporter
        auto& sparseVoxelGrid1 = sparseScene1Opt->getVoxelGrid();
        auto& sparseVoxelGrid2 = sparseScene2Opt->getVoxelGrid();
        UnifiedVoxelExporter::exportUnifiedScene(
            sparseVoxelGrid1, sparseVoxelGrid2, sparseVoxelChanges, 
            cameras1, "unified_scene_data.json", targets2, targetNames2, 25
        );
    } else if (denseGrid1Opt.has_value() && denseGrid2Opt.has_value()) {
        // We have dense data - use the dense exporter (converts to sparse format internally)
        UnifiedVoxelExporter::exportUnifiedScene(
            *denseGrid1Opt, *denseGrid2Opt, denseVoxelChanges,
            cameras1, "unified_scene_data.json", targets2, targetNames2, 25
        );
    } else {
        std::cout << "No voxel data available for export.\n";
    }

    // Performance analysis and timing
    auto programEnd = std::chrono::high_resolution_clock::now();
    auto scenarioLoadTime = std::chrono::duration_cast<std::chrono::milliseconds>(scenarioLoadEnd - scenarioLoadStart);
    auto totalDuration = std::chrono::duration_cast<std::chrono::milliseconds>(programEnd - programStart);

    std::cout << "\n=== TIMING BREAKDOWN ===\n";
    std::cout << "Scenario loading: " << scenarioLoadTime.count() << "ms\n";
    
    if (preferDense) {
        comparison.printComparison();
    } else {
        std::cout << "Sparse ray intersection: " << comparison.sparse.ray_processing_time.count() << "ms\n";
        std::cout << "Sparse motion analysis: " << comparison.sparse.motion_processing_time.count() << "ms\n";
        std::cout << "Memory usage: " << comparison.sparse.memory_usage_mb << "MB\n";
        std::cout << "Sparsity: " << (comparison.sparse.sparsity_ratio * 100) << "% empty space\n";
    }
    
    std::cout << "Total runtime: " << totalDuration.count() << "ms (" 
              << std::fixed << std::setprecision(2) << (totalDuration.count() / 1000.0) << "s)\n";

    // Additional sparse-specific information
    if (!preferDense) {
        std::cout << "\n=== SPARSE GRID ANALYSIS ===\n";
        if (sparseScene1Opt.has_value()) {
            sparseScene1Opt->printSceneInfo();
        }
    }
    
    return 0;
}