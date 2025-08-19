# Testing LX DEX on Linux/CUDA

## Quick Start

### On your Linux box with NVIDIA GPU:

```bash
# Clone the repository
git clone https://github.com/luxfi/dex.git
cd dex

# Run automated CUDA test
make test-cuda

# Or use the script directly
./scripts/test-cuda.sh
```

## What the Test Does

1. **Detects NVIDIA GPU** - Uses `nvidia-smi` to verify GPU presence
2. **Checks CUDA** - Ensures CUDA toolkit is installed
3. **Builds MLX Engine** - Compiles with CUDA support
4. **Runs Tests** - Executes unit tests with GPU acceleration
5. **Benchmarks** - Measures performance (30 second run)
6. **Compares** - Shows CUDA vs CPU performance difference

## Expected Output

```
===================================
LX DEX CUDA Testing Script
===================================
Checking for NVIDIA GPU...
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.3   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|   0  NVIDIA RTX 4090     Off  | 00000000:01:00.0 Off |                  N/A |
| 30%   45C    P8    25W / 450W |      0MiB / 24576MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

Building MLX engine with CUDA support...
MLX engine built successfully!

Running MLX tests with CUDA...
=== RUN   TestMLXEngine
CUDA backend detected
MLX Engine initialized with: NVIDIA RTX 4090 (CUDA)
--- PASS: TestMLXEngine (0.00s)

Running benchmarks...
BenchmarkMLXMatching-32    20000000    65 ns/op    15.4M matches/sec

===================================
Performance Summary
===================================
CUDA Performance: 65 ns/op (15.4M matches/sec)
CPU Performance: 2500 ns/op (400K matches/sec)

CUDA Speedup: 38.46x faster than CPU
```

## Requirements

- Linux (Ubuntu 20.04+ recommended)
- NVIDIA GPU (GTX 10xx series or newer)
- CUDA 11.8+ (12.3 recommended)
- Go 1.21+
- gcc/g++ 7.0+

## Docker Alternative

If you don't want to install CUDA locally:

```bash
# Build and run in Docker with GPU support
make docker-cuda

# Or manually
docker build -f Dockerfile.cuda -t lux-dex:cuda .
docker run --gpus all lux-dex:cuda
```

## CI Integration

The repository includes GitHub Actions that automatically:
- Test on NVIDIA GPUs using self-hosted runners
- Run in CUDA Docker containers
- Compare performance across different GPU types
- Check for performance regressions

## Troubleshooting

### "nvidia-smi not found"
Install NVIDIA drivers:
```bash
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot
```

### "CUDA not found"
Install CUDA toolkit:
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
```

### "Library not found"
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Performance Tuning

For maximum performance:

1. **Set GPU to P0 state**:
```bash
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 450  # Set power limit to max
```

2. **Disable GPU boost** (for consistent benchmarks):
```bash
sudo nvidia-smi -lgc 2100  # Lock GPU clock
```

3. **Use larger batches**:
Edit `pkg/mlx/mlx_test.go` and increase order count from 100 to 10000

## Support

- Full documentation: [docs/CUDA_SETUP.md](docs/CUDA_SETUP.md)
- GitHub Issues: https://github.com/luxfi/dex/issues
- Discord: https://discord.gg/lux