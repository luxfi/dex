// ZeroMQ Network Benchmark - Tests real network throughput
package main

import (
    "flag"
    "fmt"
    "log"
    "os/exec"
    "strings"
    "time"
)

func main() {
    var (
        mode      = flag.String("mode", "local", "Benchmark mode: local, manual")
        traders   = flag.Int("traders", 100, "Number of traders")
        rate      = flag.Int("rate", 1000, "Orders per second per trader")
        duration  = flag.Duration("duration", 30*time.Second, "Test duration")
    )
    flag.Parse()

    fmt.Println("🔬 ZeroMQ Network Benchmark")
    fmt.Println("===========================")
    
    switch *mode {
    case "local":
        runLocalTest(*traders, *rate, *duration)
    case "manual":
        printManualInstructions(*traders, *rate, *duration)
    }
}

func runLocalTest(traders int, rate int, duration time.Duration) {
    fmt.Println("\n📊 Running LOCAL test (same machine)")
    fmt.Printf("Traders: %d\n", traders)
    fmt.Printf("Rate per trader: %d orders/sec\n", rate)
    fmt.Printf("Total target: %d orders/sec\n", traders*rate)
    fmt.Printf("Duration: %v\n", duration)
    fmt.Println(strings.Repeat("-", 50))
    
    // Build first
    fmt.Println("Building ZMQ tools...")
    exec.Command("go", "build", "-o", "../zmq-exchange/zmq-exchange", "../zmq-exchange").Run()
    exec.Command("go", "build", "-o", "../zmq-trader/zmq-trader", "../zmq-trader").Run()
    
    // Start exchange
    fmt.Println("Starting exchange server...")
    exchangeCmd := exec.Command("../zmq-exchange/zmq-exchange", "-bind", "tcp://*:5555")
    if err := exchangeCmd.Start(); err != nil {
        log.Fatalf("Failed to start exchange: %v", err)
    }
    defer exchangeCmd.Process.Kill()
    
    time.Sleep(2 * time.Second)
    
    // Run trader
    fmt.Println("Starting traders...")
    traderCmd := exec.Command("../zmq-trader/zmq-trader",
        "-server", "tcp://localhost:5555",
        "-traders", fmt.Sprintf("%d", traders),
        "-rate", fmt.Sprintf("%d", rate),
        "-duration", duration.String())
    
    output, err := traderCmd.CombinedOutput()
    if err != nil {
        log.Printf("Trader error: %v", err)
    }
    
    fmt.Println("\n" + string(output))
}

func printManualInstructions(traders int, rate int, duration time.Duration) {
    fmt.Println("\n📋 Manual Distributed Test Instructions")
    fmt.Println(strings.Repeat("=", 50))
    
    fmt.Println("\n1️⃣ BUILD on all nodes:")
    fmt.Println("   go build ./cmd/zmq-exchange")
    fmt.Println("   go build ./cmd/zmq-trader")
    
    fmt.Println("\n2️⃣ On EXCHANGE node (e.g., 10.0.0.10):")
    fmt.Println("   ./zmq-exchange -bind 'tcp://*:5555'")
    
    fmt.Println("\n3️⃣ On TRADER nodes:")
    fmt.Printf("   ./zmq-trader -server 'tcp://10.0.0.10:5555' -traders %d -rate %d -duration %v\n",
        traders, rate, duration)
    
    fmt.Println("\n4️⃣ For multiple trader nodes, divide traders:")
    fmt.Printf("   Node1: ./zmq-trader -server 'tcp://10.0.0.10:5555' -traders %d -rate %d\n",
        traders/2, rate)
    fmt.Printf("   Node2: ./zmq-trader -server 'tcp://10.0.0.10:5555' -traders %d -rate %d\n",
        traders/2, rate)
    
    fmt.Println("\n⚙️ Network Tuning (run on all nodes):")
    fmt.Println("   # Increase TCP buffers")
    fmt.Println("   sudo sysctl -w net.core.rmem_max=134217728")
    fmt.Println("   sudo sysctl -w net.core.wmem_max=134217728")
    fmt.Println("   sudo sysctl -w net.ipv4.tcp_rmem='4096 87380 134217728'")
    fmt.Println("   sudo sysctl -w net.ipv4.tcp_wmem='4096 65536 134217728'")
    
    fmt.Println("\n📊 Expected Performance on 10Gbps:")
    fmt.Println("   • With 100-byte messages: ~12.5M messages/sec theoretical")
    fmt.Println("   • With 200-byte orders: ~6.25M orders/sec theoretical")
    fmt.Println("   • Realistic with overhead: 2-4M orders/sec")
    
    fmt.Println("\n🎯 To saturate 10Gbps:")
    fmt.Printf("   • Need ~50,000 traders at %d orders/sec each\n", rate)
    fmt.Println("   • Or batch messages for higher efficiency")
    fmt.Println("   • Use multiple exchange instances (sharding)")
}
