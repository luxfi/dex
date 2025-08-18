#!/bin/bash

# ZMQ Binary FIX Benchmark Test Script
# Runs multi-node consensus with load testing

set -e

# Configuration
NODE1_HOST=${NODE1_HOST:-"192.168.1.100"}
NODE2_HOST=${NODE2_HOST:-"192.168.1.101"}
LOAD1_HOST=${LOAD1_HOST:-"192.168.1.102"}
LOAD2_HOST=${LOAD2_HOST:-"192.168.1.103"}

# Build the benchmark
echo "Building ZMQ benchmark..."
cd ../cmd/zmq-benchmark
go build -o zmq-benchmark

# Function to run on remote host
run_remote() {
    local host=$1
    local cmd=$2
    echo "Running on $host: $cmd"
    ssh $host "cd ~/lx/dex/backend/cmd/zmq-benchmark && $cmd" &
}

# Function to run locally
run_local() {
    local cmd=$1
    echo "Running locally: $cmd"
    $cmd &
}

# Clean up previous runs
cleanup() {
    echo "Cleaning up..."
    pkill -f zmq-benchmark || true
    rm -rf ./badger-node-* || true
    ssh $NODE2_HOST "pkill -f zmq-benchmark || true; rm -rf ~/lx/dex/backend/cmd/zmq-benchmark/badger-node-*" || true
}

# Test 1: Basic throughput test (single node)
test_basic_throughput() {
    echo "=== Test 1: Basic Throughput Test ==="
    
    # Start consumer
    ./zmq-benchmark -mode consumer -endpoint "tcp://0.0.0.0:5555" -consumers 4 -duration 30s &
    CONSUMER_PID=$!
    
    sleep 2
    
    # Start producer
    ./zmq-benchmark -mode producer -endpoint "tcp://127.0.0.1:5555" -producers 4 -rate 1000000 -batch 100 -duration 30s
    
    wait $CONSUMER_PID
    echo "Test 1 complete"
}

# Test 2: Two-node consensus with BadgerDB
test_consensus() {
    echo "=== Test 2: Two-Node Consensus Test ==="
    
    # Start Node 1 (local)
    echo "Starting Node 1..."
    ./zmq-benchmark -mode consensus -node 1 -endpoint "tcp://0.0.0.0:5555" \
        "tcp://$NODE2_HOST:6002" &
    NODE1_PID=$!
    
    # Start Node 2 (remote)
    echo "Starting Node 2..."
    run_remote $NODE2_HOST "./zmq-benchmark -mode consensus -node 2 -endpoint 'tcp://0.0.0.0:5555' 'tcp://$NODE1_HOST:6001'"
    
    sleep 5
    
    # Start load generators
    echo "Starting load generators..."
    run_remote $LOAD1_HOST "./zmq-benchmark -mode producer -endpoint 'tcp://$NODE1_HOST:5555' -producers 8 -rate 500000 -duration 60s"
    run_remote $LOAD2_HOST "./zmq-benchmark -mode producer -endpoint 'tcp://$NODE2_HOST:5555' -producers 8 -rate 500000 -duration 60s"
    
    # Wait for test to complete
    sleep 65
    
    # Stop nodes
    kill $NODE1_PID || true
    ssh $NODE2_HOST "pkill -f zmq-benchmark" || true
    
    echo "Test 2 complete"
}

# Test 3: Maximum throughput test (pummel mode)
test_pummel() {
    echo "=== Test 3: Pummel Test (Maximum Load) ==="
    
    # Start relay/proxy for load distribution
    ./zmq-benchmark -mode relay -endpoint "tcp://0.0.0.0:5555" &
    RELAY_PID=$!
    
    sleep 2
    
    # Start multiple consumers
    for i in {1..8}; do
        ./zmq-benchmark -mode consumer -endpoint "tcp://127.0.0.1:5556" -consumers 1 -duration 60s &
    done
    
    sleep 2
    
    # Start load generators from all machines
    echo "Starting maximum load from all machines..."
    run_local "./zmq-benchmark -mode producer -endpoint 'tcp://127.0.0.1:5555' -producers 16 -rate 2000000 -batch 1000 -duration 60s"
    run_remote $LOAD1_HOST "./zmq-benchmark -mode producer -endpoint 'tcp://$NODE1_HOST:5555' -producers 16 -rate 2000000 -batch 1000 -duration 60s"
    run_remote $LOAD2_HOST "./zmq-benchmark -mode producer -endpoint 'tcp://$NODE1_HOST:5555' -producers 16 -rate 2000000 -batch 1000 -duration 60s"
    
    # Wait for test
    sleep 65
    
    # Cleanup
    kill $RELAY_PID || true
    pkill -f "zmq-benchmark.*consumer" || true
    
    echo "Test 3 complete"
}

# Test 4: Latency measurement under load
test_latency() {
    echo "=== Test 4: Latency Test ==="
    
    # Start consumer with latency tracking
    ./zmq-benchmark -mode consumer -endpoint "tcp://0.0.0.0:5555" -consumers 4 -latency -duration 30s &
    CONSUMER_PID=$!
    
    sleep 2
    
    # Start producer with controlled rate
    ./zmq-benchmark -mode producer -endpoint "tcp://127.0.0.1:5555" -producers 2 -rate 100000 -batch 10 -duration 30s
    
    wait $CONSUMER_PID
    echo "Test 4 complete"
}

# Main execution
main() {
    echo "Starting ZMQ Binary FIX Benchmark Suite"
    echo "======================================="
    echo "Node 1: $NODE1_HOST (local)"
    echo "Node 2: $NODE2_HOST"
    echo "Load 1: $LOAD1_HOST"
    echo "Load 2: $LOAD2_HOST"
    echo ""
    
    # Cleanup first
    cleanup
    
    # Run tests based on argument
    case "${1:-all}" in
        basic)
            test_basic_throughput
            ;;
        consensus)
            test_consensus
            ;;
        pummel)
            test_pummel
            ;;
        latency)
            test_latency
            ;;
        all)
            test_basic_throughput
            echo ""
            sleep 5
            test_latency
            echo ""
            sleep 5
            test_consensus
            echo ""
            sleep 5
            test_pummel
            ;;
        *)
            echo "Usage: $0 [basic|consensus|pummel|latency|all]"
            exit 1
            ;;
    esac
    
    echo ""
    echo "All tests complete!"
    
    # Final cleanup
    cleanup
}

# Run main function
main "$@"