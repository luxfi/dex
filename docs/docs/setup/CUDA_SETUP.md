# CUDA Setup and Testing Guide for LX DEX

## Overview

LX DEX supports GPU acceleration through the MLX engine, which automatically detects and uses:
- **CUDA** on NVIDIA GPUs (Linux/Windows)
- **Metal** on Apple Silicon (macOS)
- **CPU** fallback when no GPU is available

## Performance Expectations

| Backend | Hardware | Expected Performance |
|---------|----------|---------------------|
| CUDA | RTX 4090 | 20-50M orders/sec |
| CUDA | RTX 3090 | 10-20M orders/sec |
| CUDA | A100 | 50-100M orders/sec |
| Metal | M1 Max | 5-10M orders/sec |
| Metal | M2 Ultra | 10-20M orders/sec |
| CPU | AMD EPYC | 100K-500K orders/sec |

## Prerequisites for CUDA

### 1. NVIDIA GPU
- Compute Capability 6.0+ (GTX 10xx series or newer)
- Recommended: RTX 30xx/40xx series or datacenter GPUs (A100, H100)

### 2. NVIDIA Drivers
```bash
# Check if installed
nvidia-smi

# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535

# RHEL/CentOS/Fedora
sudo dnf install nvidia-driver
```

### 3. CUDA Toolkit
```bash
# Download CUDA 12.3 (recommended)
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
sudo sh cuda_12.3.0_545.23.06_linux.run

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify
nvcc --version
```

## Quick Test on Linux/CUDA

### Method 1: Using Test Script (Recommended)
```bash
# Clone the repository
git clone https://github.com/luxfi/dex.git
cd dex

# Run CUDA test script
./scripts/test-cuda.sh
```

### Method 2: Manual Testing
```bash
# 1. Build MLX engine with CUDA
cd bridge
g++ -std=c++17 -O3 -fPIC -DHAS_CUDA \
    -I/usr/local/cuda/include \
    -c mlx_engine.cpp -o mlx_engine.o

g++ -shared -o libmlx_engine.so mlx_engine.o \
    -L/usr/local/cuda/lib64 -lcudart -lcuda

# 2. Copy library
cp libmlx_engine.so ../pkg/mlx/

# 3. Run tests
cd ..
export CGO_ENABLED=1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

go test -v ./pkg/mlx/...
```

### Method 3: Using Docker
```bash
# Build CUDA Docker image
docker build -f Dockerfile.cuda -t lux-dex:cuda .

# Run with GPU support
docker run --gpus all lux-dex:cuda

# Run tests in container
docker run --gpus all lux-dex:cuda go test -v ./pkg/mlx/...
```

## Verifying CUDA Detection

When CUDA is properly detected, you should see:
```
CUDA backend detected
MLX Engine initialized with: NVIDIA RTX 4090 (CUDA)
```

If CUDA is not detected, it will fall back to CPU:
```
Using CPU backend (no GPU detected)
MLX Engine initialized with: CPU (No GPU acceleration)
```

## Benchmarking

### Run Full Benchmark Suite
```bash
# Basic benchmark
go test -bench=MLX -benchtime=10s ./pkg/mlx/...

# Comprehensive benchmark with memory stats
go test -bench=. -benchmem -benchtime=30s ./pkg/mlx/... | tee benchmark.txt

# Compare CUDA vs CPU
./scripts/test-cuda.sh  # This automatically runs both and compares
```

### Expected Benchmark Results

**NVIDIA RTX 4090:**
```
BenchmarkMLXMatching-32    15000000    75 ns/op    13.3M matches/sec
```

**NVIDIA A100:**
```
BenchmarkMLXMatching-32    30000000    35 ns/op    28.5M matches/sec
```

**CPU Baseline (AMD EPYC):**
```
BenchmarkMLXMatching-32    500000     2500 ns/op   400K matches/sec
```

## CI/CD Integration

### GitHub Actions
The repository includes GitHub Actions workflows that:
1. Test with and without CGO on multiple OS/Go versions
2. Run CUDA tests in Docker containers
3. Benchmark and check for performance regressions

### GitLab CI
```yaml
test-cuda:
  image: nvidia/cuda:12.3.0-devel-ubuntu22.04
  tags:
    - gpu
  script:
    - ./scripts/test-cuda.sh
```

### Jenkins
```groovy
pipeline {
    agent {
        docker {
            image 'nvidia/cuda:12.3.0-devel-ubuntu22.04'
            args '--gpus all'
        }
    }
    stages {
        stage('Test CUDA') {
            steps {
                sh './scripts/test-cuda.sh'
            }
        }
    }
}
```

## Troubleshooting

### Issue: CUDA not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
ls /usr/local/cuda/lib64/libcudart.so

# Check environment variables
echo $LD_LIBRARY_PATH
echo $PATH
```

### Issue: Library not found
```bash
# Add library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:./pkg/mlx:$LD_LIBRARY_PATH

# Or copy library to system path
sudo cp bridge/libmlx_engine.so /usr/local/lib/
sudo ldconfig
```

### Issue: Compilation errors
```bash
# Ensure you have C++17 support
g++ --version  # Should be 7.0+

# Install required packages
sudo apt install build-essential pkg-config
```

### Issue: Performance not as expected
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Check thermal throttling
nvidia-smi -q -d TEMPERATURE

# Set GPU to maximum performance
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 350  # Set power limit (W)
```

## Multi-GPU Support (Future)

The MLX engine currently uses single GPU. Multi-GPU support is planned:
```cpp
// Future API
mlx_engine_set_device(engine, device_id);
mlx_engine_enable_multi_gpu(engine, num_gpus);
```

## Cloud Provider Setup

### AWS (p3/p4/g5 instances)
```bash
# Launch p3.2xlarge with Deep Learning AMI
aws ec2 run-instances --image-id ami-xxxxxx --instance-type p3.2xlarge

# CUDA is pre-installed, just run:
./scripts/test-cuda.sh
```

### Google Cloud (A100/T4/V100)
```bash
# Create instance with GPU
gcloud compute instances create lx-dex-gpu \
    --accelerator type=nvidia-tesla-a100,count=1 \
    --maintenance-policy TERMINATE \
    --image-family ubuntu-2204-lts \
    --image-project ubuntu-os-cloud

# Install CUDA driver
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-drivers
```

### Azure (NC/ND/NV series)
```bash
# Use GPU-optimized VM
az vm create \
    --resource-group myResourceGroup \
    --name lx-dex-gpu \
    --image Ubuntu2204 \
    --size Standard_NC6s_v3

# NVIDIA drivers included, install CUDA toolkit
./scripts/test-cuda.sh
```

## Performance Tuning

### 1. Batch Size Optimization
```go
// Adjust batch size for your GPU
const OptimalBatchSize = 10000  // For RTX 4090
const OptimalBatchSize = 50000  // For A100
```

### 2. Memory Pool Pre-allocation
```cpp
// In mlx_engine.cpp
static constexpr size_t POOL_SIZE = 16ULL * 1024 * 1024 * 1024; // 16GB
```

### 3. Kernel Launch Configuration
```cpp
// Optimize for your GPU
dim3 blocks(256);   // For consumer GPUs
dim3 blocks(1024);  // For datacenter GPUs
```

## Contact and Support

- GitHub Issues: https://github.com/luxfi/dex/issues
- Documentation: https://docs.lux.network/dex
- Discord: https://discord.gg/lux

## License

The MLX engine and CUDA integration are part of the LX DEX project under the MIT License.