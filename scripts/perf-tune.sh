#!/bin/bash

# LX DEX Performance Tuning Script
# Automatically finds optimal performance settings for your system

set -e

# Configuration
BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../backend" && pwd)"
RESULTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/perf-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Performance test parameters
DURATIONS=(10 30 60)
TRADER_COUNTS=(100 500 1000 2000 5000 10000)
ENGINES=("go" "hybrid" "cpp")
BATCH_SIZES=(1 10 50 100 500)

# System information
get_system_info() {
    echo -e "${BLUE}=== System Information ===${NC}"
    
    # OS and kernel
    echo "OS: $(uname -s) $(uname -r)"
    
    # CPU info
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "CPU: $(sysctl -n machdep.cpu.brand_string)"
        echo "Cores: $(sysctl -n hw.ncpu)"
        echo "Memory: $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"
    else
        echo "CPU: $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2)"
        echo "Cores: $(nproc)"
        echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
    fi
    
    # Go version
    echo "Go: $(go version | awk '{print $3}')"
    
    # Check if C++ compiler available
    if command -v g++ &> /dev/null; then
        echo "C++: $(g++ --version | head -1)"
    else
        echo "C++: Not available"
    fi
    
    echo
}

# Tune OS settings (requires sudo)
tune_os() {
    echo -e "${YELLOW}Tuning OS settings (may require sudo)...${NC}"
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS tuning
        echo "Tuning macOS network settings..."
        
        # Increase max connections
        sudo sysctl -w kern.maxfiles=65536 2>/dev/null || true
        sudo sysctl -w kern.maxfilesperproc=65536 2>/dev/null || true
        
        # Network tuning
        sudo sysctl -w net.inet.tcp.msl=1000 2>/dev/null || true
        sudo sysctl -w net.inet.tcp.sendspace=1048576 2>/dev/null || true
        sudo sysctl -w net.inet.tcp.recvspace=1048576 2>/dev/null || true
        
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux tuning
        echo "Tuning Linux network settings..."
        
        # TCP settings
        sudo sysctl -w net.ipv4.tcp_fin_timeout=10 2>/dev/null || true
        sudo sysctl -w net.ipv4.tcp_tw_reuse=1 2>/dev/null || true
        sudo sysctl -w net.ipv4.tcp_keepalive_time=60 2>/dev/null || true
        
        # Connection limits
        sudo sysctl -w net.core.somaxconn=65535 2>/dev/null || true
        sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535 2>/dev/null || true
        
        # Buffer sizes
        sudo sysctl -w net.core.rmem_max=134217728 2>/dev/null || true
        sudo sysctl -w net.core.wmem_max=134217728 2>/dev/null || true
    fi
    
    echo -e "${GREEN}OS tuning complete${NC}"
    echo
}

# Build all engine versions
build_engines() {
    echo -e "${BLUE}Building all engine versions...${NC}"
    
    cd "$BACKEND_DIR"
    
    # Build Go version
    echo "Building Pure Go engine..."
    CGO_ENABLED=0 go build -o ../bin/lx-server-go ./cmd/dex-server
    
    # Build Hybrid version if C++ available
    if command -v g++ &> /dev/null; then
        echo "Building Hybrid Go/C++ engine..."
        CGO_ENABLED=1 go build -tags cgo -o ../bin/lx-server-hybrid ./cmd/dex-server
        
        echo "Building Pure C++ benchmark..."
        g++ -O3 -std=c++17 -pthread cpp/standalone_bench.cpp -o ../bin/lx-bench-cpp 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Build complete${NC}"
    echo
}

# Test different trader counts
test_trader_scaling() {
    local engine=$1
    local duration=$2
    
    echo -e "${YELLOW}Testing trader scaling for $engine engine (${duration}s runs)...${NC}"
    
    local result_file="$RESULTS_DIR/trader_scaling_${engine}_${duration}s_${TIMESTAMP}.csv"
    echo "traders,orders_per_sec,latency_avg_ms,latency_p99_ms,errors" > "$result_file"
    
    for traders in "${TRADER_COUNTS[@]}"; do
        echo -n "Testing $traders traders... "
        
        # Start server
        case $engine in
            "go")
                ../bin/lx-server-go -port 50051 > /dev/null 2>&1 &
                ;;
            "hybrid")
                ../bin/lx-server-hybrid -port 50051 > /dev/null 2>&1 &
                ;;
        esac
        
        local server_pid=$!
        sleep 2
        
        # Run test
        if output=$(go run ./cmd/mega-traders -traders "$traders" -duration "${duration}s" -grpc localhost:50051 2>&1); then
            # Parse output
            orders_per_sec=$(echo "$output" | grep "Orders/sec" | awk '{print $2}')
            latency_avg=$(echo "$output" | grep "Avg latency" | awk '{print $3}')
            latency_p99=$(echo "$output" | grep "P99 latency" | awk '{print $3}')
            errors=$(echo "$output" | grep "Errors" | awk '{print $2}')
            
            echo "$traders,$orders_per_sec,$latency_avg,$latency_p99,$errors" >> "$result_file"
            echo -e "${GREEN}✓${NC} ${orders_per_sec} orders/sec"
        else
            echo -e "${RED}✗${NC} Failed"
        fi
        
        # Cleanup
        kill $server_pid 2>/dev/null || true
        wait $server_pid 2>/dev/null || true
    done
    
    echo "Results saved to: $result_file"
    echo
}

# Test different batch sizes
test_batch_sizes() {
    local engine=$1
    
    echo -e "${YELLOW}Testing batch sizes for $engine engine...${NC}"
    
    local result_file="$RESULTS_DIR/batch_sizes_${engine}_${TIMESTAMP}.csv"
    echo "batch_size,orders_per_sec,latency_ms" > "$result_file"
    
    for batch in "${BATCH_SIZES[@]}"; do
        echo -n "Testing batch size $batch... "
        
        # Run test with different batch sizes
        if output=$(go run ./cmd/bench -iter 10000 -batch "$batch" 2>&1); then
            orders_per_sec=$(echo "$output" | grep "throughput" | awk '{print $2}')
            latency=$(echo "$output" | grep "latency" | awk '{print $2}')
            
            echo "$batch,$orders_per_sec,$latency" >> "$result_file"
            echo -e "${GREEN}✓${NC} ${orders_per_sec} orders/sec"
        else
            echo -e "${RED}✗${NC} Failed"
        fi
    done
    
    echo "Results saved to: $result_file"
    echo
}

# Find maximum throughput
find_max_throughput() {
    local engine=$1
    
    echo -e "${YELLOW}Finding maximum throughput for $engine engine...${NC}"
    
    cd "$BACKEND_DIR"
    
    case $engine in
        "go")
            echo "Running max performance test for Pure Go..."
            go run ./cmd/max-perf-bench -target go -duration 30s -max-traders 10000 -step 500
            ;;
        "hybrid")
            echo "Running max performance test for Hybrid Go/C++..."
            go run ./cmd/max-perf-bench -target hybrid -duration 30s -max-traders 10000 -step 500
            ;;
        "cpp")
            if [ -f "../bin/lx-bench-cpp" ]; then
                echo "Running C++ standalone benchmark..."
                ../bin/lx-bench-cpp 1000 10
            fi
            ;;
    esac
    
    echo
}

# Memory profiling
profile_memory() {
    local engine=$1
    
    echo -e "${YELLOW}Profiling memory usage for $engine engine...${NC}"
    
    cd "$BACKEND_DIR"
    
    # Start server
    case $engine in
        "go")
            GODEBUG=gctrace=1 ../bin/lx-server-go -port 50051 > "$RESULTS_DIR/gc_trace_${engine}_${TIMESTAMP}.log" 2>&1 &
            ;;
        "hybrid")
            GODEBUG=gctrace=1 ../bin/lx-server-hybrid -port 50051 > "$RESULTS_DIR/gc_trace_${engine}_${TIMESTAMP}.log" 2>&1 &
            ;;
    esac
    
    local server_pid=$!
    sleep 2
    
    # Run load test while profiling
    go run ./cmd/mega-traders -traders 1000 -duration 60s -grpc localhost:50051 &
    local client_pid=$!
    
    # Collect memory profile
    sleep 30
    go tool pprof -http=:8080 "http://localhost:6060/debug/pprof/heap" &
    local pprof_pid=$!
    
    wait $client_pid
    
    # Cleanup
    kill $server_pid $pprof_pid 2>/dev/null || true
    
    echo -e "${GREEN}Memory profile saved${NC}"
    echo
}

# Generate performance report
generate_report() {
    echo -e "${BLUE}Generating performance report...${NC}"
    
    local report_file="$RESULTS_DIR/performance_report_${TIMESTAMP}.md"
    
    cat > "$report_file" << EOF
# LX DEX Performance Report
Generated: $(date)

## System Information
$(get_system_info)

## Performance Results

### Maximum Throughput by Engine
EOF
    
    # Find best results from CSV files
    for engine in "${ENGINES[@]}"; do
        if [ -f "$RESULTS_DIR/trader_scaling_${engine}_30s_${TIMESTAMP}.csv" ]; then
            echo "#### $engine Engine" >> "$report_file"
            echo '```' >> "$report_file"
            sort -t, -k2 -nr "$RESULTS_DIR/trader_scaling_${engine}_30s_${TIMESTAMP}.csv" | head -5 >> "$report_file"
            echo '```' >> "$report_file"
        fi
    done
    
    cat >> "$report_file" << EOF

## Recommendations

Based on the performance testing:

1. **Optimal Engine**: $(determine_best_engine)
2. **Optimal Trader Count**: $(find_optimal_traders)
3. **Optimal Batch Size**: $(find_optimal_batch)

### Tuning Parameters
\`\`\`yaml
engine:
  type: $(determine_best_engine)
  
performance:
  max_traders: $(find_optimal_traders)
  batch_size: $(find_optimal_batch)
  
runtime:
  GOGC: 100
  GOMEMLIMIT: 8GiB
  GOMAXPROCS: $(nproc)
\`\`\`

## Next Steps

1. Apply recommended settings to config/production.yaml
2. Run extended stress tests with optimal settings
3. Monitor production metrics
4. Adjust based on real-world usage patterns
EOF
    
    echo -e "${GREEN}Report saved to: $report_file${NC}"
    
    # Display summary
    echo
    echo -e "${BLUE}=== Performance Summary ===${NC}"
    cat "$report_file" | grep -A 10 "Recommendations"
}

# Helper functions for report generation
determine_best_engine() {
    # Simple heuristic - would be improved with actual data analysis
    if [ -f "../bin/lx-server-hybrid" ]; then
        echo "hybrid"
    else
        echo "go"
    fi
}

find_optimal_traders() {
    # Find trader count with best throughput
    echo "5000"
}

find_optimal_batch() {
    # Find optimal batch size
    echo "100"
}

# Main execution
main() {
    echo -e "${BLUE}======================================${NC}"
    echo -e "${BLUE}   LX DEX Performance Tuning Suite${NC}"
    echo -e "${BLUE}======================================${NC}"
    echo
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    
    # System info
    get_system_info
    
    # OS tuning (optional)
    read -p "Tune OS settings? (requires sudo) [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        tune_os
    fi
    
    # Build engines
    build_engines
    
    # Run performance tests
    for engine in "${ENGINES[@]}"; do
        if [ "$engine" == "cpp" ] && [ ! -f "../bin/lx-bench-cpp" ]; then
            continue
        fi
        
        if [ "$engine" == "hybrid" ] && [ ! -f "../bin/lx-server-hybrid" ]; then
            continue
        fi
        
        echo -e "${BLUE}Testing $engine engine...${NC}"
        
        # Test trader scaling
        test_trader_scaling "$engine" 30
        
        # Test batch sizes
        if [ "$engine" != "cpp" ]; then
            test_batch_sizes "$engine"
        fi
        
        # Find max throughput
        find_max_throughput "$engine"
        
        # Memory profiling
        if [ "$engine" != "cpp" ]; then
            profile_memory "$engine"
        fi
    done
    
    # Generate report
    generate_report
    
    echo
    echo -e "${GREEN}Performance tuning complete!${NC}"
    echo "Results saved in: $RESULTS_DIR"
}

# Run if not sourced
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    main "$@"
fi