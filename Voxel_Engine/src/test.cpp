#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include "PixelMotion.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------
// Load a grayscale image into a vector<uint8_t>
// ---------------------------------------------------------
std::vector<uint8_t> loadImage(const std::string& path, int& width, int& height) {
    int channels;
    uint8_t* data = stbi_load(path.c_str(), &width, &height, &channels, 1); // force 1 channel (grayscale)
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    std::vector<uint8_t> image(data, data + width * height);
    stbi_image_free(data);
    return image;
}

// ---------------------------------------------------------
// Save a grayscale image from a vector<uint8_t>
// ---------------------------------------------------------
void saveImage(const std::string& path, const std::vector<uint8_t>& data, int width, int height) {
    if (!stbi_write_png(path.c_str(), width, height, 1, data.data(), width)) {
        throw std::runtime_error("Failed to write image: " + path);
    }
}

// ---------------------------------------------------------
// Program entry point
// ---------------------------------------------------------
int main(int argc, char** argv) {
    try {
        auto totalStart = std::chrono::high_resolution_clock::now();
        
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " <image1> <image2> <output>\n";
            return 1;
        }

        const std::string imagePath1 = argv[1];
        const std::string imagePath2 = argv[2];
        const std::string outputPath = argv[3];

        int width1, height1;
        int width2, height2;

        // Load both grayscale images
        std::cout << "Loading images...\n";
        auto loadStart = std::chrono::high_resolution_clock::now();
        auto img1 = loadImage(imagePath1, width1, height1);
        auto img2 = loadImage(imagePath2, width2, height2);
        auto loadEnd = std::chrono::high_resolution_clock::now();
        auto loadDuration = std::chrono::duration_cast<std::chrono::microseconds>(loadEnd - loadStart);
        std::cout << "Image loading took: " << loadDuration.count() << " us (" << loadDuration.count() / 1000.0 << " ms)\n";

        if (width1 != width2 || height1 != height2) {
            throw std::runtime_error("Images must have the same dimensions.");
        }

        // Configure motion detection parameters
        PixelMotion::Settings settings;
        settings.motionThreshold = 15.0f;
        settings.useAbsoluteDiff = true;
        settings.normalizeOutput = true;
        settings.contrastBoost = 1.2f; // optional tweak for visibility

        // Compute motion map
        std::cout << "Computing motion map...\n";
        auto motionStart = std::chrono::high_resolution_clock::now();
        PixelMotion motion(width1, height1, settings);
        auto motionMap = motion.computeMotion(img1, img2);
        auto motionEnd = std::chrono::high_resolution_clock::now();
        auto motionDuration = std::chrono::duration_cast<std::chrono::microseconds>(motionEnd - motionStart);
        std::cout << "Motion computation took: " << motionDuration.count() << " us (" << motionDuration.count() / 1000.0 << " ms)\n";

        // Save result
        std::cout << "Saving result...\n";
        auto saveStart = std::chrono::high_resolution_clock::now();
        saveImage(outputPath, motionMap, width1, height1);
        auto saveEnd = std::chrono::high_resolution_clock::now();
        auto saveDuration = std::chrono::duration_cast<std::chrono::microseconds>(saveEnd - saveStart);
        std::cout << "Image saving took: " << saveDuration.count() << " us (" << saveDuration.count() / 1000.0 << " ms)\n";
        
        auto totalEnd = std::chrono::high_resolution_clock::now();
        auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(totalEnd - totalStart);
        std::cout << "Total execution time: " << totalDuration.count() << " us (" << totalDuration.count() / 1000.0 << " ms)\n";
        std::cout << "Motion map saved to " << outputPath << "\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }
}
