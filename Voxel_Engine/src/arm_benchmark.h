#pragma once

#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

/**
 * @brief ARM/Raspberry Pi specific benchmark utilities
 */
namespace ARMBenchmark {
    
    struct SystemInfo {
        int cpu_cores;
        bool is_arm;
        bool has_neon;
        bool is_low_power;
        std::string cpu_name;
        
        static SystemInfo detect() {
            SystemInfo info;
            info.cpu_cores = std::thread::hardware_concurrency();
            info.is_low_power = (info.cpu_cores <= 4);
            
            #ifdef __ARM_ARCH
            info.is_arm = true;
            #else
            info.is_arm = false;
            #endif
            
            #ifdef __ARM_NEON
            info.has_neon = true;
            #else
            info.has_neon = false;
            #endif
            
            // Try to detect Raspberry Pi specifically
            info.cpu_name = "Unknown";
            
            #ifdef __arm__
            // Read /proc/cpuinfo to identify Pi model
            std::ifstream cpuinfo("/proc/cpuinfo");
            if (cpuinfo.is_open()) {
                std::string line;
                while (std::getline(cpuinfo, line)) {
                    if (line.find("Hardware") != std::string::npos && line.find("BCM") != std::string::npos) {
                        if (info.cpu_cores == 4) {
                            // Check for Pi 3B vs Pi 4 by revision or model
                            cpuinfo.clear();
                            cpuinfo.seekg(0);
                            while (std::getline(cpuinfo, line)) {
                                if (line.find("Model") != std::string::npos) {
                                    if (line.find("Raspberry Pi 3") != std::string::npos) {
                                        info.cpu_name = "Raspberry Pi 3B";
                                        break;
                                    } else if (line.find("Raspberry Pi 4") != std::string::npos) {
                                        info.cpu_name = "Raspberry Pi 4";
                                        break;
                                    }
                                }
                            }
                            if (info.cpu_name == "Unknown") {
                                info.cpu_name = "Raspberry Pi 3/4 (4-core)";
                            }
                        } else if (info.cpu_cores == 1) {
                            info.cpu_name = "Raspberry Pi Zero/1";
                        } else {
                            info.cpu_name = "Raspberry Pi (detected)";
                        }
                        break;
                    }
                }
                cpuinfo.close();
            } else if (info.cpu_cores == 4 && info.is_low_power) {
                info.cpu_name = "Raspberry Pi (likely 3B/4)";
            }
            #endif
            
            return info;
        }
        
        void print() const {
            std::cout << "\n=== SYSTEM INFORMATION ===\n";
            std::cout << "CPU Cores: " << cpu_cores << "\n";
            std::cout << "Architecture: " << (is_arm ? "ARM" : "x86/x64") << "\n";
            std::cout << "NEON SIMD: " << (has_neon ? "Available" : "Not available") << "\n";
            std::cout << "Power Class: " << (is_low_power ? "Low-power device" : "High-performance") << "\n";
            std::cout << "CPU Name: " << cpu_name << "\n";
        }
    };
    
    /**
     * @brief Simple memory bandwidth test for cache optimization
     */
    class MemoryBandwidthTest {
    public:
        static double testSequentialAccess(size_t size_mb = 16) {
            const size_t size = size_mb * 1024 * 1024;
            std::vector<int> data(size / sizeof(int));
            
            auto start = std::chrono::high_resolution_clock::now();
            
            // Sequential access pattern
            volatile int sum = 0;
            for (size_t i = 0; i < data.size(); ++i) {
                sum += data[i];
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            
            double bandwidth_gb_s = (static_cast<double>(size) / (1024*1024*1024)) / (duration.count() / 1000000.0);
            return bandwidth_gb_s;
        }
        
        static void runTest() {
            std::cout << "\n=== MEMORY BANDWIDTH TEST ===\n";
            double bandwidth = testSequentialAccess(16);
            std::cout << "Sequential read bandwidth: " << bandwidth << " GB/s\n";
            
            if (bandwidth < 2.0) {
                std::cout << "LOW bandwidth detected - applying aggressive ARM optimizations\n";
            } else if (bandwidth < 10.0) {
                std::cout << "MODERATE bandwidth - applying standard optimizations\n";
            } else {
                std::cout << "HIGH bandwidth - using performance-oriented settings\n";
            }
        }
    };
    
    /**
     * @brief Performance recommendations based on detected hardware
     */
    struct PerformanceRecommendations {
        int recommended_threads;
        int recommended_ray_step;
        int recommended_block_size;
        float recommended_top_percentage;
        float recommended_max_distance;
        bool use_reduced_precision;
        
        static PerformanceRecommendations generate(const SystemInfo& info) {
            PerformanceRecommendations rec;
            
            if (info.is_low_power) {
                // Raspberry Pi optimizations
                if (info.cpu_cores == 4 && info.cpu_name.find("Pi") != std::string::npos) {
                    // Raspberry Pi 3B specific optimizations (4 cores, Cortex-A53)
                    rec.recommended_threads = 3;  // Leave one core for system
                    rec.recommended_ray_step = 12; // Very aggressive ray reduction
                    rec.recommended_block_size = 24; // Smaller blocks for L1 cache
                    rec.recommended_top_percentage = 1.0f;  // Only process brightest 1%
                    rec.recommended_max_distance = 50.0f;   // Very short rays
                    rec.use_reduced_precision = true;
                } else {
                    // Generic low-power device
                    rec.recommended_threads = std::max(1, info.cpu_cores - 1);
                    rec.recommended_ray_step = 8;  // Fewer rays
                    rec.recommended_block_size = 32;
                    rec.recommended_top_percentage = 1.5f;  // Very selective
                    rec.recommended_max_distance = 60.0f;   // Shorter rays
                    rec.use_reduced_precision = true;
                }
            } else if (info.cpu_cores <= 8) {
                // Mid-range systems
                rec.recommended_threads = info.cpu_cores;
                rec.recommended_ray_step = 4;
                rec.recommended_block_size = 64;
                rec.recommended_top_percentage = 3.0f;
                rec.recommended_max_distance = 100.0f;
                rec.use_reduced_precision = false;
            } else {
                // High-performance systems
                rec.recommended_threads = info.cpu_cores;
                rec.recommended_ray_step = 2;
                rec.recommended_block_size = 128;
                rec.recommended_top_percentage = 5.0f;
                rec.recommended_max_distance = 200.0f;
                rec.use_reduced_precision = false;
            }
            
            return rec;
        }
        
        void print() const {
            std::cout << "\n=== PERFORMANCE RECOMMENDATIONS ===\n";
            std::cout << "Recommended threads: " << recommended_threads << "\n";
            std::cout << "Ray step size: " << recommended_ray_step << "\n";
            std::cout << "Block size: " << recommended_block_size << "\n";
            std::cout << "Top percentage: " << recommended_top_percentage << "%\n";
            std::cout << "Max ray distance: " << recommended_max_distance << "m\n";
            std::cout << "Reduced precision: " << (use_reduced_precision ? "Yes" : "No") << "\n";
        }
    };
}