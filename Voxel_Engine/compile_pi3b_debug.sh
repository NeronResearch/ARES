#!/bin/bash

# Raspberry Pi 3B Debug Compile Script
# Safer compilation with debug symbols for troubleshooting

echo "=== Raspberry Pi 3B Debug Compilation ==="
echo "Adding debug symbols and safety checks..."

# Detect architecture mode
ARCH=$(uname -m)
echo "Architecture: $ARCH"

# Pi 3B debug flags
CXX="g++"
OUTPUT="main_pi3b_debug"

# Architecture-specific optimizations for Pi 3B
case "$ARCH" in
    aarch64|arm64)
        echo "64-bit ARM debug mode"
        ARCH_FLAGS="-march=armv8-a -mtune=cortex-a53"
        DEFINES="-DNDEBUG -DARM_NEON -D__ARM_NEON -DNEON_AVAILABLE=1 -DRASPBERRY_PI_3B -DAARCH64 -DDEBUG_ARM"
        ;;
    armv7l)
        echo "32-bit ARM debug mode"
        ARCH_FLAGS="-march=armv7-a -mtune=cortex-a53 -mfpu=neon-vfpv4 -mfloat-abi=hard"
        DEFINES="-DNDEBUG -DARM_NEON -D__ARM_NEON -DNEON_AVAILABLE=1 -DRASPBERRY_PI_3B -DARMV7L -DDEBUG_ARM"
        ;;
    *)
        echo "Unknown architecture, using safe defaults"
        ARCH_FLAGS="-march=native"
        DEFINES="-DNDEBUG -DARM_NEON -D__ARM_NEON -DNEON_AVAILABLE=1 -DRASPBERRY_PI_3B -DDEBUG_ARM"
        ;;
esac

# Debug-friendly optimization (less aggressive)
OPTIMIZATION="-O1"  # Gentler optimization
PERFORMANCE_FLAGS="-ffast-math -funroll-loops"
CACHE_FLAGS="-ffunction-sections -fdata-sections"
DEBUG_FLAGS="-g -ggdb -fno-omit-frame-pointer"  # Full debug info
SAFETY_FLAGS="-fstack-protector-strong -D_FORTIFY_SOURCE=2"

# Conservative OpenMP
OPENMP_FLAGS="-fopenmp"

# Link flags
LINKER_FLAGS="-Wl,--gc-sections"

# Check dependencies
if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    echo "ERROR: OpenCV not found"
    exit 1
fi

# Get OpenCV flags
if pkg-config --exists opencv4; then
    OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4)
else
    OPENCV_FLAGS=$(pkg-config --cflags --libs opencv)
fi

# Source files
SOURCES="src/main.cpp src/PixelMotion.cpp src/ImageDenoise.cpp src/SkyDetector.cpp src/VoxelEngine.cpp src/Scenario.cpp src/Camera.cpp src/Target.cpp src/VoxelMotion.cpp"

# Build final command
ALL_FLAGS="$ARCH_FLAGS $OPTIMIZATION $PERFORMANCE_FLAGS $CACHE_FLAGS $DEBUG_FLAGS $SAFETY_FLAGS $OPENMP_FLAGS $DEFINES"
ALL_LIBS="$OPENCV_FLAGS -ltbb -lm -lpthread"

echo ""
echo "Debug Compilation Settings:"
echo "- Optimization: O1 (debug-friendly)"
echo "- Debug symbols: Full"
echo "- Stack protection: Enabled"
echo "- Output: $OUTPUT"
echo ""

echo "Compiling debug version..."
set -x

$CXX -std=c++17 $ALL_FLAGS $SOURCES -o $OUTPUT $LINKER_FLAGS $ALL_LIBS

RESULT=$?
set +x

if [ $RESULT -eq 0 ] && [ -f "$OUTPUT" ]; then
    echo ""
    echo "=== DEBUG BUILD SUCCESS ==="
    echo "Debug binary created: $OUTPUT"
    ls -lh $OUTPUT
    
    echo ""
    echo "Debug Features:"
    echo "- Full debug symbols for gdb"
    echo "- Stack protection enabled"
    echo "- Conservative optimization"
    echo "- ARM safety checks enabled"
    echo ""
    echo "To debug segfaults:"
    echo "  gdb ./$OUTPUT"
    echo "  (gdb) run /path/to/scenario.json"
    echo "  (gdb) bt  # for backtrace when it crashes"
    
else
    echo ""
    echo "=== DEBUG BUILD FAILED ==="
    exit 1
fi