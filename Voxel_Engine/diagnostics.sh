#!/bin/bash

# ARES Compilation Diagnostics
# Run this to check your system before compiling

echo "=== ARES Compilation Diagnostics ==="
echo ""

echo "System Information:"
echo "- OS: $(uname -a)"
echo "- Architecture: $(uname -m)"
echo "- Shell: $SHELL"
echo "- Bash Version: $BASH_VERSION"
echo ""

echo "Hardware Detection:"
if [ -f /proc/cpuinfo ]; then
    echo "- CPU Model: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
    echo "- Hardware: $(grep 'Hardware' /proc/cpuinfo | cut -d':' -f2 | xargs 2>/dev/null || echo 'Not available')"
    echo "- CPU Cores: $(nproc)"
    
    if grep -q "Raspberry Pi" /proc/cpuinfo; then
        echo "- Raspberry Pi: YES"
        PI_MODEL=$(grep 'Model' /proc/cpuinfo | cut -d':' -f2 | xargs)
        echo "- Model: $PI_MODEL"
    else
        echo "- Raspberry Pi: NO"
    fi
else
    echo "- /proc/cpuinfo not available"
fi
echo ""

echo "Compiler Information:"
if command -v g++ >/dev/null 2>&1; then
    echo "- g++ version: $(g++ --version | head -1)"
    echo "- g++ path: $(which g++)"
    
    # Test ARM support
    echo "- Testing ARM compilation support..."
    echo 'int main(){return 0;}' > /tmp/test_arm.cpp
    
    if g++ -march=native /tmp/test_arm.cpp -o /tmp/test_arm 2>/dev/null; then
        echo "  ✓ Native compilation works"
    else
        echo "  ✗ Native compilation failed"
    fi
    
    # Test NEON support
    if uname -m | grep -q arm; then
        echo "- Testing NEON SIMD support..."
        cat > /tmp/test_neon.cpp << 'EOF'
#ifdef __ARM_NEON
#include <arm_neon.h>
int main() { 
    float32x4_t v = vdupq_n_f32(1.0f);
    return 0; 
}
#else
int main() { return 1; }
#endif
EOF
        
        if g++ -march=native /tmp/test_neon.cpp -o /tmp/test_neon 2>/dev/null; then
            echo "  ✓ NEON SIMD support available"
        else
            echo "  ✗ NEON SIMD support not available"
        fi
    fi
    
    rm -f /tmp/test_*.cpp /tmp/test_arm /tmp/test_neon
else
    echo "- g++ not found!"
fi
echo ""

echo "Required Libraries:"
for lib in opencv4 opencv tbb; do
    if pkg-config --exists $lib 2>/dev/null; then
        echo "- $lib: ✓ $(pkg-config --modversion $lib)"
    else
        echo "- $lib: ✗ Not found"
    fi
done

# Check for OpenMP
if echo '#include <omp.h>' | g++ -fopenmp -x c++ - -o /tmp/test_omp 2>/dev/null; then
    echo "- OpenMP: ✓ Available"
    rm -f /tmp/test_omp
else
    echo "- OpenMP: ✗ Not available"
fi
echo ""

echo "Performance Recommendations:"
ARCH=$(uname -m)
CORES=$(nproc)

case "$ARCH" in
    aarch64|arm64)
        echo "- Detected ARM 64-bit (modern Raspberry Pi)"
        echo "- Recommended flags: -march=armv8-a+crc -mtune=cortex-a72"
        echo "- NEON SIMD should be available"
        ;;
    armv7l)
        echo "- Detected ARM 32-bit (Raspberry Pi 3/4 in 32-bit mode)"
        echo "- Recommended flags: -march=armv7-a -mfpu=neon-vfpv4"
        echo "- NEON SIMD should be available"
        ;;
    armv6l)
        echo "- Detected ARM v6 (Raspberry Pi Zero/1)"
        echo "- Recommended flags: -march=armv6+fp -mfpu=vfp"
        echo "- No NEON support - will be slower"
        ;;
    *)
        echo "- Detected x86/x64 architecture"
        echo "- Use standard optimizations"
        ;;
esac

if [ "$CORES" -le 2 ]; then
    echo "- Low core count ($CORES) - use conservative parallel settings"
elif [ "$CORES" -le 4 ]; then
    echo "- Moderate core count ($CORES) - standard parallel settings"
else
    echo "- High core count ($CORES) - aggressive parallel settings"
fi

echo ""
echo "To compile with optimizations, run:"
echo "  bash ./compile.sh"
echo "or"
echo "  ./compile_arm.sh"