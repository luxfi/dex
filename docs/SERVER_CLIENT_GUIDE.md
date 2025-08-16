# 🖥️ Server & Client Commands Guide

## Quick Start

The LX Engine now includes convenient commands to start servers and clients individually. Here's how to use them:

## Available Commands

### 🚀 Servers

```bash
# Start DEX server (Pure Go) on port 50051
make dex-server

# Start DEX server (Hybrid C++) on port 50051 - Higher performance
make dex-server-hybrid  

# Start Gateway server on port 8080
make gateway-server

# Start LX exchange server on port 5555
make zmq-server
```

### 💹 Clients

```bash
# Start mega trader with 1000 concurrent traders
make mega-trader

# Start LX trader client
make zmq-trader

# Start C++ FIX trader client
make fix-trader-client

# Start FIX message generator (streaming mode)
make fix-generator
```

## Usage Examples

### Example 1: Run DEX Server with Mega Traders

**Terminal 1 - Start Server:**
```bash
make dex-server
# Server starts on port 50051
```

**Terminal 2 - Start Traders:**
```bash
make mega-trader
# Starts 1000 traders connecting to localhost:50051
```

### Example 2: LX Network Test

**Terminal 1 - Start Exchange:**
```bash
make zmq-server
# Exchange starts on port 5555
```

**Terminal 2 - Start Trader:**
```bash
make zmq-trader
# Trader connects to localhost:5555
```

### Example 3: High-Performance Testing

**Terminal 1 - Start Hybrid Server:**
```bash
make dex-server-hybrid
# High-performance C++ hybrid server on port 50051
```

**Terminal 2 - Run Benchmark:**
```bash
make mega-trader
# Tests with 1000 traders
```

## Custom Parameters

You can also run the binaries directly with custom parameters:

### DEX Server Options
```bash
backend/bin/lx-dex -port 50051 -log-level debug
```

### Mega Traders Options
```bash
backend/bin/mega-traders \
  -traders 5000 \        # Number of traders
  -rate 100 \            # Orders per second per trader
  -duration 60s \        # Test duration
  -grpc localhost:50051  # Server address
```

### LX Trader Options
```bash
backend/bin/zmq-trader \
  -exchange tcp://localhost:5555 \
  -id trader1 \
  -rate 1000             # Orders per second
```

### FIX Trader Options
```bash
backend/bin/fix-trader \
  localhost \            # Server host
  9876 \                 # Server port
  100 \                  # Number of traders
  10                     # Duration in seconds
```

## Performance Tips

1. **For Maximum Performance**: Use `make dex-server-hybrid` (C++ optimized)
2. **For Testing**: Start with `make dex-server` (Pure Go, easier debugging)
3. **For Network Testing**: Use LX components (`zmq-server` + `zmq-trader`)
4. **For FIX Protocol**: Use `fix-generator` to create test messages

## Monitoring

While running servers and clients, you can monitor performance:

```bash
# In another terminal
make monitor
```

## Quick Test Everything

```bash
# Terminal 1
make dex-server

# Terminal 2
make mega-trader

# You should see:
# - Server handling orders
# - Traders submitting orders
# - Performance metrics
```

## Troubleshooting

If a command fails:

1. **Build first**: `make build`
2. **Check ports**: Ensure ports (50051, 5555, 8080) are free
3. **Kill existing**: `pkill lx-dex` or `pkill zmq-exchange`

## Complete Command List

Run `make help` to see all available commands including:
- Server commands
- Client commands  
- Benchmark commands
- Build commands

---

**Pro Tip**: The ultra-fast FIX engine achieves 5.21M msgs/sec! Run `make bench-ultra` to test it.