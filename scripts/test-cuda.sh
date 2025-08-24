#!/bin/bash
# Test script for Linux/CUDA systems
# Run this on your Linux box with NVIDIA GPU

set -e

echo "==================================="
echo "LX DEX CUDA Testing Script"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check for NVIDIA GPU
echo -e "${YELLOW}Checking for NVIDIA GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}nvidia-smi not found. Please install NVIDIA drivers.${NC}"
    exit 1
fi

nvidia-smi
echo ""

# Check for CUDA
echo -e "${YELLOW}Checking CUDA installation...${NC}"
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}CUDA not found. Please install CUDA toolkit.${NC}"
    echo "Download from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

nvcc --version
echo ""

# Check Go installation
echo -e "${YELLOW}Checking Go installation...${NC}"
if ! command -v go &> /dev/null; then
    echo -e "${RED}Go not found. Installing Go 1.21...${NC}"
    wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz
    sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz
    export PATH=$PATH:/usr/local/go/bin
    echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
fi

go version
echo ""

# Build C++ MLX engine with CUDA support
echo -e "${YELLOW}Building MLX engine with CUDA support...${NC}"
cd bridge

# Clean previous builds
rm -f *.o *.so *.a

# Compile with CUDA support
g++ -std=c++17 -O3 -fPIC \
    -DHAS_CUDA \
    -I/usr/local/cuda/include \
    -c mlx_engine.cpp -o mlx_engine.o

# Create shared library
g++ -shared -o libmlx_engine.so mlx_engine.o \
    -L/usr/local/cuda/lib64 -lcudart -lcuda

# Copy to test directory
cp libmlx_engine.so ../pkg/mlx/

echo -e "${GREEN}MLX engine built successfully!${NC}"
echo ""

# Run tests
echo -e "${YELLOW}Running MLX tests with CUDA...${NC}"
cd ..
export CGO_ENABLED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Basic tests
echo -e "${YELLOW}Running unit tests...${NC}"
go test -v ./pkg/mlx/...

# Benchmarks
echo -e "${YELLOW}Running benchmarks (this may take a while)...${NC}"
go test -bench=. -benchtime=30s ./pkg/mlx/... | tee cuda_benchmark.txt

# Performance summary
echo ""
echo -e "${GREEN}==================================="
echo "Performance Summary"
echo "===================================${NC}"
grep -E "Benchmark.*ns/op|Benchmark.*ops/sec" cuda_benchmark.txt || true

# Compare with CPU baseline
echo ""
echo -e "${YELLOW}Running CPU-only baseline for comparison...${NC}"
cd bridge
g++ -std=c++17 -O3 -fPIC -c mlx_engine.cpp -o mlx_engine_cpu.o
g++ -shared -o libmlx_engine.so mlx_engine_cpu.o -lstdc++
cp libmlx_engine.so ../pkg/mlx/
cd ..

go test -bench=MLX -benchtime=10s ./pkg/mlx/... | tee cpu_benchmark.txt

echo ""
echo -e "${GREEN}==================================="
echo "CPU vs CUDA Comparison"
echo "===================================${NC}"
echo "CPU Performance:"
grep "BenchmarkMLXMatching" cpu_benchmark.txt || true
echo ""
echo "CUDA Performance:"
grep "BenchmarkMLXMatching" cuda_benchmark.txt || true

# Calculate speedup
if [ -f cuda_benchmark.txt ] && [ -f cpu_benchmark.txt ]; then
    CPU_NS=$(grep "BenchmarkMLXMatching" cpu_benchmark.txt | awk '{print $3}')
    CUDA_NS=$(grep "BenchmarkMLXMatching" cuda_benchmark.txt | awk '{print $3}')
    
    if [ ! -z "$CPU_NS" ] && [ ! -z "$CUDA_NS" ]; then
        SPEEDUP=$(echo "scale=2; $CPU_NS / $CUDA_NS" | bc)
        echo ""
        echo -e "${GREEN}CUDA Speedup: ${SPEEDUP}x faster than CPU${NC}"
    fi
fi

echo ""
echo -e "${GREEN}Testing complete!${NC}"