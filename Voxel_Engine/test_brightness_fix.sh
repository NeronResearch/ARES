#!/bin/bash

# Quick test of the brightness threshold fix for Pi 3B

echo "=== Testing Pi 3B Brightness Threshold Fix ==="

# Recompile with the fixes
echo "Recompiling with brightness threshold fixes..."
if ./compile_pi3b.sh > compile_output.log 2>&1; then
    echo "✓ Compilation successful"
else
    echo "✗ Compilation failed, check compile_output.log"
    exit 1
fi

# Test with a scenario
SCENARIO="/home/connor/Code/ARES/Scenarios/Scenario3/scenario.json"

if [ ! -f "$SCENARIO" ]; then
    echo "✗ Scenario file not found: $SCENARIO"
    echo "Please update the path to your scenario file"
    exit 1
fi

echo ""
echo "Testing with scenario: $SCENARIO"
echo "Looking for brightness threshold improvements..."
echo ""

# Run and capture output
echo "Running optimized version..."
timeout 60s ./main_pi3b_optimized "$SCENARIO" 2>&1 | tee test_output.log

# Analyze results
echo ""
echo "=== ANALYSIS ==="

if grep -q "ARM-processed 0 rays" test_output.log; then
    echo "✗ Still processing 0 rays - brightness threshold too restrictive"
    
    echo ""
    echo "Histogram data:"
    grep -E "Histogram analysis|Cumulative count|Brightness distribution|ARM-optimized brightness threshold" test_output.log
    
    echo ""
    echo "Suggestions:"
    echo "1. Images may be very dark - consider increasing topPercentage"
    echo "2. Check if pixel brightness calculation is working correctly"
    echo "3. Try debug version: ./compile_pi3b_debug.sh"
    
elif grep -q "ARM-processed [1-9]" test_output.log; then
    RAYS=$(grep "ARM-processed" test_output.log | grep -o '[0-9]\+' | tail -1)
    echo "✓ Processing $RAYS rays - brightness threshold fix working!"
    
    if grep -q "Sparse processing complete: [1-9]" test_output.log; then
        CHANGES=$(grep "Sparse processing complete" test_output.log | grep -o '[0-9]\+' | tail -1)
        echo "✓ Detected $CHANGES voxel changes - full pipeline working!"
    else
        echo "⚠ Processing rays but no voxel changes detected"
        echo "  This might be normal if there's no motion between frames"
    fi
    
else
    echo "? Unexpected output - check test_output.log for details"
fi

echo ""
echo "Full output saved to: test_output.log"
echo "Compile log saved to: compile_output.log"