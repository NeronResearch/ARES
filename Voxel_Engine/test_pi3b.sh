#!/bin/bash

# Quick Pi 3B Architecture Test

echo "=== Raspberry Pi 3B Architecture Detection ==="
echo ""

echo "System Information:"
echo "- Architecture: $(uname -m)"
echo "- Kernel: $(uname -r)"
echo "- OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2 | tr -d '\"' 2>/dev/null || echo 'Unknown')"
echo ""

if [ -f /proc/cpuinfo ]; then
    echo "CPU Information:"
    echo "- Model: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs 2>/dev/null || echo 'Not available')"
    echo "- Hardware: $(grep 'Hardware' /proc/cpuinfo | cut -d':' -f2 | xargs 2>/dev/null || echo 'Not available')"
    echo "- Cores: $(nproc)"
    
    if grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
        echo "- Raspberry Pi: YES"
        PI_MODEL=$(grep 'Model' /proc/cpuinfo | cut -d':' -f2 | xargs 2>/dev/null || echo 'Model not found')
        echo "- Model: $PI_MODEL"
    fi
    echo ""
fi

echo "Compiler Test:"
ARCH=$(uname -m)
case "$ARCH" in
    aarch64|arm64)
        echo "- Mode: 64-bit ARM (aarch64)"
        echo "- Recommended flags: -march=armv8-a -mtune=cortex-a53"
        TEST_FLAGS="-march=armv8-a -mtune=cortex-a53"
        ;;
    armv7l)
        echo "- Mode: 32-bit ARM (armv7l)"
        echo "- Recommended flags: -march=armv7-a -mtune=cortex-a53 -mfpu=neon-vfpv4"
        TEST_FLAGS="-march=armv7-a -mtune=cortex-a53 -mfpu=neon-vfpv4 -mfloat-abi=hard"
        ;;
    *)
        echo "- Mode: Unknown ($ARCH)"
        TEST_FLAGS="-march=native"
        ;;
esac

# Test compilation
echo ""
echo "Testing compiler flags..."
echo 'int main(){return 0;}' > /tmp/pi3b_test.cpp

if g++ $TEST_FLAGS /tmp/pi3b_test.cpp -o /tmp/pi3b_test 2>/dev/null; then
    echo "✓ Compiler flags work correctly"
    rm -f /tmp/pi3b_test
else
    echo "✗ Compiler flags failed - will use generic settings"
fi

rm -f /tmp/pi3b_test.cpp

echo ""
echo "Ready to compile with Pi 3B optimizations!"
echo "Run: ./compile_pi3b.sh"