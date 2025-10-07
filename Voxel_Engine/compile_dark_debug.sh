#!/bin/bash

# Dark Image Debug Compile Script
# Special build for troubleshooting very dark surveillance camera images

echo "=== DARK IMAGE DEBUG BUILD ==="
echo "Optimized for surveillance camera footage analysis"

# Basic settings
CXX="g++"
OUTPUT="main_dark_debug"
ARCH=$(uname -m)

# Debug and optimization flags for dark image analysis
OPTIMIZATION="-O2"
DEBUG_FLAGS="-g -DDARK_IMAGE_DEBUG"
PERFORMANCE_FLAGS="-ffast-math -funroll-loops"

# Architecture-specific flags
case "$ARCH" in
    aarch64|arm64)
        ARCH_FLAGS="-march=armv8-a -mtune=cortex-a53"
        DEFINES="-DARM_NEON -D__ARM_NEON -DNEON_AVAILABLE=1"
        echo "ARM 64-bit mode (Pi 3B/4 64-bit)"
        ;;
    armv7l)
        ARCH_FLAGS="-march=armv7-a -mtune=cortex-a53 -mfpu=neon-vfpv4 -mfloat-abi=hard"
        DEFINES="-DARM_NEON -D__ARM_NEON -DNEON_AVAILABLE=1"
        echo "ARM 32-bit mode (Pi 3B 32-bit)"
        ;;
    *)
        ARCH_FLAGS="-march=native -mtune=native"
        DEFINES="-DSSE_AVAILABLE=1 -DNEON_AVAILABLE=0"
        echo "x86/x64 mode"
        ;;
esac

# Dark image specific defines
DEFINES="$DEFINES -DDARK_IMAGE_MODE -DNDEBUG -DVERBOSE_BRIGHTNESS"

# OpenMP
OPENMP_FLAGS="-fopenmp"

# Libraries
if pkg-config --exists opencv4; then
    OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4)
else
    OPENCV_FLAGS=$(pkg-config --cflags --libs opencv)
fi

EXTERNAL_LIBS="-ltbb -lm -lpthread"

# Source files
SOURCES="src/main.cpp src/PixelMotion.cpp src/ImageDenoise.cpp src/SkyDetector.cpp src/VoxelEngine.cpp src/Scenario.cpp src/Camera.cpp src/Target.cpp src/VoxelMotion.cpp"

# Build command
ALL_FLAGS="$ARCH_FLAGS $OPTIMIZATION $DEBUG_FLAGS $PERFORMANCE_FLAGS $OPENMP_FLAGS $DEFINES"

echo ""
echo "Dark Image Debug Settings:"
echo "- Architecture: $ARCH"
echo "- Dark image optimizations: Enabled"
echo "- Verbose brightness analysis: Enabled"
echo "- Debug symbols: Enabled"
echo "- Output: $OUTPUT"
echo ""

echo "Compiling..."
set -x

$CXX -std=c++17 $ALL_FLAGS $SOURCES -o $OUTPUT $OPENCV_FLAGS $EXTERNAL_LIBS

RESULT=$?
set +x

if [ $RESULT -eq 0 ] && [ -f "$OUTPUT" ]; then
    echo ""
    echo "=== SUCCESS ==="
    echo "Dark image debug build created: $OUTPUT"
    ls -lh $OUTPUT
    
    echo ""
    echo "Features enabled:"
    echo "- Enhanced dark image processing"
    echo "- Verbose brightness histogram analysis"  
    echo "- Adaptive threshold calculation"
    echo "- Debug symbols for troubleshooting"
    echo ""
    echo "Usage:"
    echo "  ./$OUTPUT /path/to/scenario.json"
    echo ""
    echo "This build will:"
    echo "- Show detailed brightness analysis"
    echo "- Use more aggressive thresholds for dark images"
    echo "- Process more pixels from low-brightness ranges"
    echo "- Provide better diagnostic output"
    
else
    echo ""
    echo "=== BUILD FAILED ==="
    exit 1
fi