#pragma once
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <string>

class ImageDenoise {
public:
    enum class Method {
        MEDIAN,         // Median filter (good for salt & pepper noise)
        GAUSSIAN,       // Gaussian blur (general smoothing)
        BILATERAL,      // Bilateral filter (edge-preserving)
        MORPHOLOGICAL,  // Opening + Closing (removes small artifacts)
        THRESHOLD_CLEAN // Clean up based on neighborhood thresholds
    };

    struct Settings {
        Method method;
        int kernelSize;           // Size of filter kernel (3, 5, 7, etc.)
        float gaussianSigma;      // For Gaussian filter
        float bilateralSigmaColor;   // For bilateral filter
        float bilateralSigmaSpace;   // For bilateral filter
        uint8_t minNeighborCount; // For threshold cleaning
        uint8_t noiseThreshold;   // Pixels below this are considered noise candidates
        
        Settings() 
            : method(Method::MEDIAN)
            , kernelSize(3)
            , gaussianSigma(1.0f)
            , bilateralSigmaColor(50.0f)
            , bilateralSigmaSpace(50.0f)
            , minNeighborCount(2)
            , noiseThreshold(30)
        {}
    };

    ImageDenoise(int width, int height, const Settings& settings = Settings());
    
    std::vector<uint8_t> denoise(const std::vector<uint8_t>& inputImage);
    
    // Individual denoising methods
    std::vector<uint8_t> medianFilter(const std::vector<uint8_t>& inputImage, int kernelSize = 3);
    std::vector<uint8_t> gaussianBlur(const std::vector<uint8_t>& inputImage, int kernelSize = 5, float sigma = 1.0f);
    std::vector<uint8_t> bilateralFilter(const std::vector<uint8_t>& inputImage, int kernelSize = 5, float sigmaColor = 50.0f, float sigmaSpace = 50.0f);
    std::vector<uint8_t> morphologicalClean(const std::vector<uint8_t>& inputImage, int kernelSize = 3);
    std::vector<uint8_t> thresholdClean(const std::vector<uint8_t>& inputImage, uint8_t noiseThreshold = 30, uint8_t minNeighborCount = 2);

private:
    int width;
    int height;
    Settings settings;
    
    // Helper functions
    inline bool isValidPixel(int x, int y) const {
        return x >= 0 && x < width && y >= 0 && y < height;
    }
    
    inline size_t getIndex(int x, int y) const {
        return static_cast<size_t>(y * width + x);
    }
    
    uint8_t getPixelSafe(const std::vector<uint8_t>& image, int x, int y) const;
    std::vector<uint8_t> getNeighborhood(const std::vector<uint8_t>& image, int centerX, int centerY, int kernelSize) const;
    float gaussianWeight(int dx, int dy, float sigma) const;
    uint8_t morphologicalErode(const std::vector<uint8_t>& image, int centerX, int centerY, int kernelSize) const;
    uint8_t morphologicalDilate(const std::vector<uint8_t>& image, int centerX, int centerY, int kernelSize) const;
    
    // Image saving functionality
    void saveImageAsJPEG(const std::vector<uint8_t>& image, const std::string& filename, int quality = 90) const;
};