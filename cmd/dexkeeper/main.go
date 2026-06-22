// Command dexkeeper is the off-chain KEEPER that drives the on-chain 0x9999 two-phase
// atomic DEX seam so a DEXFill lands on the C-Chain.
//
// It watches the C-Chain for IntentSubmitted events emitted by the 0x9999 settlement
// precompile (Phase-A intent: a maker swap with DI01 hookData), and for each intent
// submits — to the dexvm's dex.submitTx RPC — an ImportTx (consume the staged C->D atomic
// object, fund the D order) plus a settling RelayOrderTx (clob_submit, collateral-bound).
// The dexvm proposer relays the order once and exports a D->C proceeds object; the keeper
// then drives the Phase-B settlement swap (DS01 hookData) that consumes it and emits
// DEXFill. The keeper holds NO custody authority — value is locked in shared memory by C
// before the keeper sees it, and the D->C object pays its recorded owner regardless.
//
// Modes:
//   - seed     : seed maker liquidity on the venue (one resting ask) and exit.
//   - phase-a  : seed liquidity + send ONE Phase-A intent swap from the maker; print the
//     derived intentID + C tx hash and exit. (The watch loop, run separately or
//     via -mode watch, then drives it to settlement.)
//   - watch    : run the keeper loop, watching IntentSubmitted and driving each intent.
//
// The maker key is derived from the BIP-39 mnemonic in LUX_MNEMONIC at m/44'/9000'/0'/0/i,
// scanning for the index whose EVM address equals -maker (a KMS-synced secret supplies the
// mnemonic at deploy).
//
// Usage:
//
//	LUX_MNEMONIC="..." dexkeeper \
//	  -mode watch \
//	  -c-rpc   https://api.lux-dev.network/ext/bc/C/rpc \
//	  -d-rpc   http://luxd-0.luxd-headless.lux-devnet.svc.cluster.local:9650/ext/bc/<D>/rpc \
//	  -d-chain-id <D> \
//	  -venue   dchain-venue.lux-devnet.svc.cluster.local:9099 \
//	  -maker   0x9011E888251AB053B7bD1cdB598Db4f9DEd94714
package main

import (
	"context"
	"flag"
	"fmt"
	"math/big"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/luxfi/genesis/pkg/genesis"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
)

// defaultMaker is the devnet treasury / maker address (genesis.TreasuryAddress).
const defaultMaker = "0x9011E888251AB053B7bD1cdB598Db4f9DEd94714"

func main() {
	var (
		mode      = flag.String("mode", "watch", "watch | phase-a | seed")
		cRPC      = flag.String("c-rpc", "", "C-Chain EVM JSON-RPC (e.g. https://api.lux-dev.network/ext/bc/C/rpc)")
		dRPC      = flag.String("d-rpc", "", "dexvm RPC base URL (…/ext/bc/<D-chain-id>/rpc)")
		dChainID  = flag.String("d-chain-id", "", "D-Chain (dexvm) blockchain id (CB58)")
		venueAddr = flag.String("venue", "", "D-Chain venue ZAP address (host:port) for seeding maker liquidity")
		makerHex  = flag.String("maker", defaultMaker, "maker EVM address to derive from LUX_MNEMONIC")
		chainID   = flag.Int64("chain-id", 0, "C-Chain EVM chainID (queried via eth_chainId if 0)")
		networkID = flag.Uint("network-id", 0, "Lux primary network id for intent-id derivation (defaults to chain-id)")
		symbol    = flag.String("symbol", "LUX/LUSD", "market symbol to seed/route (seed + phase-a modes)")
		amountIn  = flag.Uint64("amount-in", 1, "Phase-A taker input amount (integer asset units)")
		askPrice  = flag.Float64("ask-price", 10.0, "maker ask price to seed (quote per base)")
		askSize   = flag.Float64("ask-size", 1.0, "maker ask size to seed (base units)")
		fromBlock = flag.Uint64("from-block", 0, "watch: first C-Chain block to scan (0 = current head)")
	)
	flag.Parse()

	mnemonic := os.Getenv(genesis.MnemonicEnvVar)
	if mnemonic == "" {
		fatalf("%s not set (the maker key is derived from it; a KMS-synced secret supplies it at deploy)", genesis.MnemonicEnvVar)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	switch *mode {
	case "seed":
		runSeed(ctx, *venueAddr, *symbol, *askPrice, *askSize)
	case "phase-a":
		runPhaseA(ctx, mnemonic, keeperFlags(cRPC, dRPC, dChainID, venueAddr, makerHex, chainID, networkID), *venueAddr, *symbol, *amountIn, *askPrice, *askSize)
	case "watch":
		runWatch(ctx, mnemonic, keeperFlags(cRPC, dRPC, dChainID, venueAddr, makerHex, chainID, networkID), *fromBlock)
	default:
		fatalf("unknown -mode %q (want watch | phase-a | seed)", *mode)
	}
}

// keeperFlags assembles the resolved keeperConfig from the raw flags, validating the
// required ones. ChainID stays nil when -chain-id is 0 so newKeeper queries eth_chainId.
func keeperFlags(cRPC, dRPC, dChainID, venueAddr, makerHex *string, chainID *int64, networkID *uint) keeperConfig {
	if *cRPC == "" {
		fatalf("-c-rpc required")
	}
	if *dRPC == "" {
		fatalf("-d-rpc required")
	}
	if *dChainID == "" {
		fatalf("-d-chain-id required")
	}
	dID, err := ids.FromString(*dChainID)
	if err != nil {
		fatalf("invalid -d-chain-id %q: %v", *dChainID, err)
	}
	cfg := keeperConfig{
		CChainRPC: *cRPC,
		DRPC:      *dRPC,
		DChainID:  dID,
		VenueAddr: *venueAddr,
		Maker:     mustAddress(*makerHex),
		NetworkID: uint32(*networkID),
	}
	if *chainID != 0 {
		cfg.ChainID = big.NewInt(*chainID)
	}
	return cfg
}

// runSeed seeds one maker book on the venue and exits.
func runSeed(ctx context.Context, venueAddr, symbol string, askPrice, askSize float64) {
	if venueAddr == "" {
		fatalf("-venue required for seed mode")
	}
	cli, closeCli, err := dialVenue(ctx, venueAddr)
	if err != nil {
		fatalf("dial venue: %v", err)
	}
	defer closeCli()
	spec := seedSpecFor(symbol, "mk", askPrice, askSize)
	pool, err := cli.seedMakerLiquidity(spec)
	if err != nil {
		fatalf("seed maker liquidity: %v", err)
	}
	log.Info("seeded maker liquidity", "symbol", symbol, "pool", fmt.Sprintf("%x", pool),
		"maker", spec.Maker, "ask", fmt.Sprintf("%g@%g", askSize, askPrice))
}

// runPhaseA seeds liquidity, then sends ONE Phase-A intent swap from the maker and prints
// the derived intentID + C tx hash.
func runPhaseA(ctx context.Context, mnemonic string, cfg keeperConfig, venueAddr, symbol string, amountIn uint64, askPrice, askSize float64) {
	if venueAddr == "" {
		fatalf("-venue required for phase-a mode (to seed the book the intent fills against)")
	}
	cli, closeCli, err := dialVenue(ctx, venueAddr)
	if err != nil {
		fatalf("dial venue: %v", err)
	}
	spec := seedSpecFor(symbol, "mk", askPrice, askSize)
	if _, err := cli.seedMakerLiquidity(spec); err != nil {
		closeCli()
		fatalf("seed maker liquidity: %v", err)
	}
	closeCli()

	k, err := newKeeper(ctx, cfg, mnemonic)
	if err != nil {
		fatalf("new keeper: %v", err)
	}
	log.Info("maker key derived", "address", k.maker.Address, "index", k.maker.Index, "chainID", k.cfg.ChainID)

	pool := poolIDForSymbol(symbol)
	marketID := pool // identity market on the devnet proof
	assetIn := spec.Quote
	pk := poolKeyForMarket(marketID)
	nonce := uint64(time.Now().UnixNano())
	deadline := uint64(time.Now().Add(1 * time.Hour).Unix())

	intentID, txHash, err := k.sendPhaseAIntent(ctx, pk, marketID, assetIn, false /*zeroForOne*/, amountIn, nonce, deadline)
	if err != nil {
		fatalf("send Phase-A intent: %v", err)
	}
	log.Info("Phase-A intent sent",
		"intentID(derived)", intentID,
		"txHash", txHash,
		"nonce", nonce,
		"amountIn", amountIn,
		"market", fmt.Sprintf("%x", marketID))
	fmt.Printf("PHASE-A: intentID(derived)=%s txHash=%s nonce=%d\n", intentID, txHash.Hex(), nonce)
}

// runWatch runs the keeper loop. fromBlock 0 resolves to the current head at boot.
func runWatch(ctx context.Context, mnemonic string, cfg keeperConfig, fromBlock uint64) {
	k, err := newKeeper(ctx, cfg, mnemonic)
	if err != nil {
		fatalf("new keeper: %v", err)
	}
	log.Info("maker key derived", "address", k.maker.Address, "index", k.maker.Index, "chainID", k.cfg.ChainID)
	if fromBlock == 0 {
		head, err := k.cc.BlockNumber(ctx)
		if err != nil {
			fatalf("BlockNumber: %v", err)
		}
		fromBlock = head
	}
	if err := k.watch(ctx, fromBlock); err != nil && ctx.Err() == nil {
		fatalf("watch loop: %v", err)
	}
	log.Info("keeper stopped")
}

// seedSpecFor builds a single-ask seed spec for a symbol. base/quote follow the venue
// convention used by cmd/fourpath-live (a "LUX"-tagged base, a "LUSD"-tagged quote), so a
// keeper-seeded book is compatible with the proven live-venue client.
func seedSpecFor(symbol, makerPrefix string, askPrice, askSize float64) seedSpec {
	return seedSpec{
		Symbol:   symbol,
		Base:     assetID(0x4c5558),     // "LUX"
		Quote:    assetID(0x4c55534400), // "LUSD"
		Maker:    makerPrefix,
		AskSize:  askSize,
		AskPrice: askPrice,
	}
}

// mustAddress parses a 0x-prefixed 20-byte EVM address or exits. The maker flag is an
// operator-supplied boundary value, so it is validated here rather than trusted.
func mustAddress(s string) common.Address {
	if !common.IsHexAddress(s) {
		fatalf("invalid -maker address %q (want 0x + 40 hex)", s)
	}
	return common.HexToAddress(s)
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "dexkeeper: "+format+"\n", args...)
	os.Exit(1)
}
