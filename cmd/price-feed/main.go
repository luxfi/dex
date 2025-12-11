// Command price-feed runs a local price aggregator with all sources.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/luxfi/dex/pkg/price"
)

func main() {
	var (
		interval = flag.Duration("interval", 500*time.Millisecond, "display interval")
		symbols  = flag.String("symbols", "LUX-USDC,LUX-USD,ETH-USD,BTC-USD", "symbols to watch")
		sources  = flag.String("sources", "all", "sources: all,xchain,achain,cchain,pyth,chainlink")
	)
	flag.Parse()

	fmt.Println("LX DEX Price Feed")
	fmt.Println("=================")
	fmt.Println()

	// Create oracle
	oracle := price.NewOracle()

	sourceList := strings.Split(*sources, ",")
	useAll := *sources == "all"

	// X-Chain - native DEX (highest priority)
	if useAll || contains(sourceList, "xchain") {
		xchain := price.NewXChainSource(
			"http://localhost:9650/ext/bc/X",
			"ws://localhost:9650/ext/bc/X/ws",
		)
		if err := xchain.Start(); err != nil {
			log.Printf("Warning: X-Chain start failed: %v", err)
		} else {
			oracle.AddSource("x-chain", xchain)
			fmt.Println("✓ X-Chain source started (native DEX)")
		}
	}

	// A-Chain - attestations
	if useAll || contains(sourceList, "achain") {
		achain := price.NewAChainSource(
			"http://localhost:9650/ext/bc/A",
			"ws://localhost:9650/ext/bc/A/ws",
		)
		if err := achain.Start(); err != nil {
			log.Printf("Warning: A-Chain start failed: %v", err)
		} else {
			oracle.AddSource("a-chain", achain)
			fmt.Println("✓ A-Chain source started (attestations)")
		}
	}

	// C-Chain - AMM pools
	if useAll || contains(sourceList, "cchain") {
		cchain := price.NewCChainSource(
			"http://localhost:9650/ext/bc/C/rpc",
			"ws://localhost:9650/ext/bc/C/ws",
		)
		if err := cchain.Start(); err != nil {
			log.Printf("Warning: C-Chain start failed: %v", err)
		} else {
			oracle.AddSource("c-chain", cchain)
			fmt.Println("✓ C-Chain source started (AMM pools)")
		}
	}

	// Pyth Network
	if useAll || contains(sourceList, "pyth") {
		pyth := price.NewPythSource(
			"wss://hermes.pyth.network/ws",
			"https://hermes.pyth.network",
		)
		if err := pyth.Connect(); err != nil {
			log.Printf("Warning: Pyth connect failed: %v (will use HTTP fallback)", err)
		}
		oracle.AddSource("pyth", pyth)
		fmt.Println("✓ Pyth source added (real-time oracle)")
	}

	// Chainlink
	if useAll || contains(sourceList, "chainlink") {
		chainlink := price.NewChainlinkSource()
		if err := chainlink.Start(); err != nil {
			log.Printf("Warning: Chainlink start failed: %v", err)
		} else {
			oracle.AddSource("chainlink", chainlink)
			fmt.Println("✓ Chainlink source started (polling oracle)")
		}
	}

	fmt.Println()

	// Parse and watch symbols
	syms := strings.Split(*symbols, ",")
	for _, sym := range syms {
		oracle.Watch(strings.TrimSpace(sym))
	}

	// Start oracle
	if err := oracle.Start(); err != nil {
		log.Fatalf("Failed to start oracle: %v", err)
	}

	fmt.Println("Symbols:", strings.Join(syms, ", "))
	fmt.Println()
	fmt.Println("Press Ctrl+C to stop")
	fmt.Println()

	// Signal handling
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		fmt.Println("\nShutting down...")
		cancel()
	}()

	// Wait for initial data
	fmt.Println("Waiting for price data...")
	time.Sleep(time.Second)

	// Display loop
	ticker := time.NewTicker(*interval)
	defer ticker.Stop()

	fmt.Printf("\n%-15s %-12s %-12s %-12s %-10s %-10s %s\n",
		"SYMBOL", "PRICE", "TWAP", "VWAP", "CONF", "SOURCE", "AGE")
	fmt.Println(strings.Repeat("-", 90))

	for {
		select {
		case <-ctx.Done():
			oracle.Stop()
			fmt.Println("Stopped.")
			return

		case <-ticker.C:
			display(oracle, syms)
		}
	}
}

func display(oracle *price.Oracle, symbols []string) {
	for _, sym := range symbols {
		data, err := oracle.Data(sym)
		if err != nil {
			fmt.Printf("%-15s %-12s\n", sym, "--")
			continue
		}

		age := time.Since(data.Timestamp)
		ageStr := fmt.Sprintf("%dms", age.Milliseconds())
		if age > time.Second {
			ageStr = fmt.Sprintf("%.1fs", age.Seconds())
		}

		twap := oracle.TWAP(sym)
		vwap := oracle.VWAP(sym)

		stale := ""
		if data.Stale {
			stale = " [STALE]"
		}

		fmt.Printf("%-15s $%-11.4f $%-11.4f $%-11.4f %-10.2f %-10s %s%s\n",
			data.Symbol,
			data.Price,
			twap,
			vwap,
			data.Confidence,
			data.Source,
			ageStr,
			stale,
		)
	}
	fmt.Println()
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if strings.TrimSpace(v) == s {
			return true
		}
	}
	return false
}
