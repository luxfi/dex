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
		interval   = flag.Duration("interval", 500*time.Millisecond, "display interval")
		symbols    = flag.String("symbols", "LUX-USDC,LZOO-USDC,LETH-USDC,ZOO-USDC", "symbols to watch")
		sources    = flag.String("sources", "all", "sources: all,xchain,achain,cchain,zoo,pyth,chainlink")
		noVerifier = flag.Bool("no-verify", false, "disable Q-Chain quantum finality verification")
	)
	flag.Parse()

	fmt.Println("LX Price Feed")
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

	// Zoo Chain - Zoo Labs DEX
	if useAll || contains(sourceList, "zoo") {
		zoo := price.NewZooChainSource(
			"http://localhost:9651/rpc",
			"ws://localhost:9651/ws",
		)
		if err := zoo.Start(); err != nil {
			log.Printf("Warning: Zoo Chain start failed: %v", err)
		} else {
			oracle.AddSource("zoo-chain", zoo)
			fmt.Println("✓ Zoo Chain source started (Zoo Labs DEX)")
		}
	}

	// Q-Chain - Quantum Finality Verifier (enabled by default)
	// This is NOT a price source - it verifies finality of prices from other sources
	if !*noVerifier {
		qchain := price.NewQChainVerifier(
			"http://localhost:9650/ext/bc/Q",
			"ws://localhost:9650/ext/bc/Q/ws",
		)
		if err := qchain.Start(); err != nil {
			log.Printf("Warning: Q-Chain start failed: %v", err)
		} else {
			oracle.SetVerifier(qchain)
			fmt.Println("✓ Q-Chain verifier attached (quantum finality)")
		}
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

	// Show header based on whether verifier is attached
	if oracle.Verifier() != nil {
		fmt.Printf("\n%-15s %-12s %-12s %-12s %-10s %-10s %-10s %s\n",
			"SYMBOL", "PRICE", "TWAP", "VWAP", "CONF", "SOURCE", "FINALITY", "AGE")
	} else {
		fmt.Printf("\n%-15s %-12s %-12s %-12s %-10s %-10s %s\n",
			"SYMBOL", "PRICE", "TWAP", "VWAP", "CONF", "SOURCE", "AGE")
	}
	fmt.Println(strings.Repeat("-", 100))

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
	hasVerifier := oracle.Verifier() != nil

	for _, sym := range symbols {
		// Use VerifiedPrice to get finality status if verifier attached
		verified, err := oracle.VerifiedPrice(sym)
		if err != nil {
			fmt.Printf("%-15s %-12s\n", sym, "--")
			continue
		}
		data := verified.Data

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

		if hasVerifier {
			// Show finality status
			finalityStr := "PENDING"
			if verified.Finalized {
				finalityStr = "QUANTUM"
			}

			fmt.Printf("%-15s $%-11.4f $%-11.4f $%-11.4f %-10.2f %-10s %-10s %s%s\n",
				data.Symbol,
				data.Price,
				twap,
				vwap,
				data.Confidence,
				data.Source,
				finalityStr,
				ageStr,
				stale,
			)
		} else {
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
	}

	// Show finality latency if verifier attached
	if hasVerifier {
		latency := oracle.Verifier().FinalityLatency()
		fmt.Printf("Quantum finality latency: %v\n", latency)
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
