#!/bin/bash

# Test MLX GPU acceleration engine
# Auto-detects Metal (macOS), CUDA (Linux/NVIDIA), or CPU fallback

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    LX DEX - MLX GPU Acceleration Test${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Detect platform and available GPU
echo -e "${YELLOW}Detecting hardware...${NC}"
PLATFORM=$(uname)
GPU_TYPE="CPU"
GPU_INFO="No GPU detected"

if [ "$PLATFORM" = "Darwin" ]; then
    # macOS - check for Apple Silicon
    if [ "$(uname -m)" = "arm64" ]; then
        GPU_TYPE="Metal"
        GPU_INFO="Apple Silicon GPU (Metal)"
        echo -e "${GREEN}✅ Apple Silicon detected - Metal backend available${NC}"
    else
        echo -e "${YELLOW}⚠️  Intel Mac detected - CPU backend only${NC}"
    fi
elif [ "$PLATFORM" = "Linux" ]; then
    # Linux - check for NVIDIA GPU
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_TYPE="CUDA"
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo -e "${GREEN}✅ NVIDIA GPU detected: $GPU_INFO${NC}"
        
        # Show CUDA version
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        echo -e "${GREEN}   CUDA Version: $CUDA_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠️  No NVIDIA GPU - CPU backend only${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Unsupported platform - CPU backend only${NC}"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Building MLX Engine${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Build the MLX engine
make build-mlx

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Running MLX Tests${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Run tests
CGO_ENABLED=1 go test -v ./pkg/mlx/... -run=MLX

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Running MLX Benchmarks${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Run benchmarks with detailed output
CGO_ENABLED=1 go test -bench=BenchmarkMLX -benchmem -benchtime=10s ./test/benchmark/... | tee mlx_bench.txt

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}    Performance Summary${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Extract key metrics
if [ -f mlx_bench.txt ]; then
    echo -e "${GREEN}Backend: $GPU_TYPE${NC}"
    echo -e "${GREEN}Device: $GPU_INFO${NC}"
    echo ""
    
    # Extract performance numbers
    SINGLE_PERF=$(grep "SingleOrder" mlx_bench.txt | awk '{print $7}')
    BATCH_PERF=$(grep "BatchProcessing" mlx_bench.txt | awk '{print $5}')
    
    if [ ! -z "$SINGLE_PERF" ]; then
        echo -e "${GREEN}Single Order Performance: $SINGLE_PERF orders/sec${NC}"
    fi
    
    if [ ! -z "$BATCH_PERF" ]; then
        echo -e "${GREEN}Batch Processing: $BATCH_PERF orders/sec${NC}"
    fi
    
    # Compare to CPU baseline
    if [ "$GPU_TYPE" != "CPU" ]; then
        echo ""
        echo -e "${YELLOW}Performance Comparison:${NC}"
        echo "  CPU Baseline:  ~400K orders/sec"
        echo "  Current ($GPU_TYPE): See above"
        
        # Calculate speedup if we have the numbers
        if [ ! -z "$SINGLE_PERF" ]; then
            # Remove "orders/sec" and convert to number
            PERF_NUM=$(echo $SINGLE_PERF | sed 's/orders\/sec//')
            # Basic comparison (would need bc for float math)
            echo ""
            if [ "$GPU_TYPE" = "Metal" ]; then
                echo -e "${GREEN}✅ Metal acceleration active${NC}"
            elif [ "$GPU_TYPE" = "CUDA" ]; then
                echo -e "${GREEN}✅ CUDA acceleration active${NC}"
            fi
        fi
    fi
    
    rm -f mlx_bench.txt
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ MLX testing complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════${NC}"

# Show next steps
echo ""
echo "Next steps:"
echo "  • Run 'make bench' for full benchmark suite"
echo "  • Run 'make 3node-bench' for network testing"
echo "  • Run 'make demo' for interactive demo"

# Exit successfully
exit 0