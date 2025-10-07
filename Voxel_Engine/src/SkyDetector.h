#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

class SkyDetector {
private:
    // Cached kernels to avoid recreation
    cv::Mat rectKernel_;
    cv::Mat ellipseKernel_;
    cv::Mat sharpenKernel_;
    
    // Cache for reusable matrices - OpenCV Mat operations are generally thread-safe
    mutable cv::Mat tempMat1_, tempMat2_, tempMat3_;
    
    // Internal helper methods
    cv::Mat findContours(const cv::Mat& thresholdImg, int nrow, int ncol);
    double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2);
    void initializeKernels();
    cv::Mat detectSkyMaskInternal(const cv::Mat& img);
    
public:
    SkyDetector();
    
    // New interface using uint8_t vectors
    // Input: RGB image data as uint8_t vector (width * height * 3)
    // Output: Binary mask as uint8_t vector (width * height, values 0 or 255)
    std::vector<uint8_t> detectSkyMask(const std::vector<uint8_t>& imageData, 
                                       int width, int height, int channels = 3);
    
    // Convenience method for file processing
    std::vector<uint8_t> detectSkyMaskFromFile(const std::string& imagePath, 
                                               int& outWidth, int& outHeight);
};