#!/bin/bash

# ARES Compile Wrapper - Ensures bash execution
# Run this with: ./compile_arm.sh

# Make sure we're using bash, not sh
if [ -z "$BASH_VERSION" ]; then
    echo "Switching to bash for proper ARM detection..."
    exec bash "$0" "$@"
fi

# Check if we're on Raspberry Pi
echo "=== Raspberry Pi ARM Compile Script ==="
echo "Architecture: $(uname -m)"
echo "OS: $(uname -o)"

if [ -f /proc/cpuinfo ]; then
    echo "CPU Info:"
    grep -E "(model name|Hardware|Revision)" /proc/cpuinfo | head -3
fi

echo ""

# Run the main compile script with bash
bash ./compile.sh