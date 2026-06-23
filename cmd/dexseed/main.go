// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command dexseed is the CANONICAL real-market seeder for the live native D-Chain
// DEX. It seeds REAL markets onto the in-luxd CLOB ingestion seam (POST
// /ext/bc/D/dex/clob_* with FROZEN zapwire frames) using REAL, consensus-native asset
// identities — dexcore.DeriveAssetID(networkID, C-chainID, kind, canonicalRef) — for
// every side of every market.
//
// CONTRAST WITH cmd/clobverify: clobverify is a LOCAL-ONLY SYNTHETIC harness. It seeds
// an ASCII-of-symbol poolID and an all-zero / label-shaped asset (e.g. "LUX/LUSD" ->
// 0x4c55582f4c555344...). If those synthetic frames reach a shared D-Chain validator
// set they persist into the replication snapshot and BRICK the native VM on the next
// boot (assertOrderUserCoverage refuses a resting order on a market with no real asset
// row). clobverify is therefore gated by internal/localguard to loopback only.
//
// dexseed is DIFFERENT and SAFE against a shared cluster: it NEVER builds an
// ascii-of-symbol id. Every asset it opens is admitted ONLY if its (kind, ref) derives
// a well-formed canonical AssetID via dexcore.DeriveAssetID — the SAME identity the
// chain's resolver computes — and the chain's own permissionless resolver additionally
// proves each side is backed by live on-chain code (the EXTCODESIZE reality gate)
// before binding the market. A synthetic/malformed ref is refused HERE, at frame-build
// time, before a single frame is sent. So dexseed needs no localguard opt-in; its
// safety is structural, not a loopback fence.
//
// END-TO-END SMOKE PATH (one real market):
//
//  1. clob_open_market(poolID, baseID, quoteID)        bind the REAL (base,quote)
//  2. clob_deposit(maker, baseID, amount, ref)         fund the maker's base balance
//  3. clob_place(poolID, SELL, askPrice, askSize, maker)  rest a maker limit ask
//  4. clob_get_markets / clob_get_book?market=<poolID> read back & confirm
//
// EXAMPLE (devnet, one real LUX/<ERC20> market against a single validator):
//
//	dexseed \
//	  -node http://localhost:9650 \
//	  -network-id 3 \
//	  -cchain-id Bm8X7THQtS2txLrQWTDXN8a4JDuhP4KGybUE754LLSiVFLQ7v \
//	  -quote-erc20 0x<20-byte-token-address> \
//	  -maker maker
//
// The base is EVM_NATIVE LUX (the EVM zero-address native marker). The quote is the
// real ERC-20 at -quote-erc20 (it MUST be a token actually deployed on the C-Chain;
// the chain's reality gate rejects a non-deployed address and the open is refused with
// HTTP 502). Omitting -quote-erc20 prints what to pass and exits — there is no
// synthetic fallback by design.
package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/ids"
)

// Read endpoint sub-path keys. These mirror pkg/dchain.MethodGetMarkets / MethodGetBook
// — the external HTTP routes the native VM registers under /ext/bc/D/dex/. They are
// declared here (not imported) so the seeder reads the JSON schema exactly as an
// external consumer (the indexer) would, never coupling to the VM package — the same
// discipline cmd/clobverify uses for the read surface.
const (
	methodGetMarkets = "clob_get_markets"
	methodGetBook    = "clob_get_book"
)

// httpClient is a short-timeout client. A write blocks in the VM's submitTx until the
// order's block is accepted, so the timeout must comfortably exceed one consensus
// round; 30s is the same bound cmd/clobverify uses.
var httpClient = &http.Client{Timeout: 30 * time.Second}

// chainHead mirrors pkg/dchain.chainHead — the accepted-block anchor every read body
// carries.
type chainHead struct {
	Height       uint64 `json:"height"`
	LastAccepted string `json:"lastAccepted"`
	Root         string `json:"root"`
}

// marketJSON mirrors pkg/dchain.marketJSON (one row of clob_get_markets).
type marketJSON struct {
	PoolID      string  `json:"poolId"`
	Symbol      string  `json:"symbol"`
	AssetsBound bool    `json:"assetsBound"`
	Orders      int     `json:"orders"`
	Remaining   float64 `json:"remaining"`
	BestBid     float64 `json:"bestBid"`
	BestAsk     float64 `json:"bestAsk"`
}

// getMarketsResp mirrors pkg/dchain.getMarketsResp (clob_get_markets body).
type getMarketsResp struct {
	chainHead
	Count   int          `json:"count"`
	Markets []marketJSON `json:"markets"`
}

// levelJSON mirrors pkg/dchain.levelJSON (one aggregated book level).
type levelJSON struct {
	Price float64 `json:"price"`
	Size  float64 `json:"size"`
	Count int     `json:"count"`
}

// getBookResp mirrors pkg/dchain.getBookResp (clob_get_book body).
type getBookResp struct {
	chainHead
	Market    string      `json:"market"`
	Orders    int         `json:"orders"`
	Remaining float64     `json:"remaining"`
	BestBid   float64     `json:"bestBid"`
	BestAsk   float64     `json:"bestAsk"`
	Bids      []levelJSON `json:"bids"`
	Asks      []levelJSON `json:"asks"`
}

func die(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "FAIL: "+format+"\n", args...)
	os.Exit(1)
}

// postFrame POSTs a FROZEN zapwire frame to <base>/ext/bc/D/dex/<method> as
// application/octet-stream and returns the raw response bytes (the zapwire ack / fills
// / balance frame). base is http://host:port (no trailing slash). This is the exact
// posting style cmd/clobverify uses, pointed at the same in-luxd ingestion seam.
func postFrame(base, method string, payload []byte) ([]byte, error) {
	url := base + "/ext/bc/D/dex/" + method
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/octet-stream")
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: HTTP %d: %s", method, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return body, nil
}

// getJSON GETs <base>/ext/bc/D/dex/<method>[?query] and decodes the JSON body.
func getJSON(base, method, query string, out any) error {
	url := base + "/ext/bc/D/dex/" + method
	if query != "" {
		url += "?" + query
	}
	resp, err := httpClient.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: HTTP %d: %s", method, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	return json.Unmarshal(body, out)
}

// parseERC20Addr parses a 0x-prefixed (or bare) 20-byte hex token address. A
// non-20-byte or non-hex value is an error — there is NO synthetic fallback.
func parseERC20Addr(s string) ([]byte, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return nil, fmt.Errorf("empty token address")
	}
	if len(s) >= 2 && (s[0:2] == "0x" || s[0:2] == "0X") {
		s = s[2:]
	}
	b, err := hex.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("not hex: %w", err)
	}
	if len(b) != 20 {
		return nil, fmt.Errorf("ERC-20 address must be 20 bytes, got %d", len(b))
	}
	return b, nil
}

func main() {
	var (
		nodeURL    = flag.String("node", "", "single validator base URL, e.g. http://localhost:9650 (REQUIRED)")
		networkID  = flag.Uint("network-id", 3, "Lux network id the assets live under (3=devnet; mainnet=1, testnet=2)")
		cChainID   = flag.String("cchain-id", "", "C-Chain blockchain id (the sourceChainID for EVM_NATIVE/ERC20 asset derivation) (REQUIRED)")
		quoteERC20 = flag.String("quote-erc20", "", "REAL ERC-20 token contract address (0x...20 bytes) to use as the market quote; the base is EVM_NATIVE LUX")
		maker      = flag.String("maker", "maker", "maker CLOB user handle (the 16-byte on-wire user, NOT an EVM key)")
		askPrice   = flag.Float64("ask-price", 1.0, "resting maker ask price (quote per base)")
		askSize    = flag.Float64("ask-size", 1.0, "resting maker ask size (base whole units)")
		settle     = flag.Duration("settle", 2*time.Second, "wait before reading back so the order's block is accepted")
	)
	flag.Parse()

	node := strings.TrimRight(strings.TrimSpace(*nodeURL), "/")
	if node == "" {
		die("-node is required (single validator base URL, e.g. http://localhost:9650)")
	}
	if *cChainID == "" {
		die("-cchain-id is required (the C-Chain blockchain id; asset ids are derived under it)")
	}
	cid, err := ids.FromString(*cChainID)
	if err != nil {
		die("-cchain-id %q: %v", *cChainID, err)
	}
	if *networkID == 0 || *networkID > 0xffffffff {
		die("-network-id %d out of range for uint32", *networkID)
	}
	netID := uint32(*networkID)

	if *quoteERC20 == "" {
		die("-quote-erc20 is required: a REAL ERC-20 token address (0x...) to quote LUX against.\n" +
			"      dexseed NEVER fabricates a synthetic quote (that bricks the chain). Pass a token\n" +
			"      actually deployed on the C-Chain, e.g. -quote-erc20 0x<20-byte-token-address>.")
	}
	quoteAddr, err := parseERC20Addr(*quoteERC20)
	if err != nil {
		die("-quote-erc20 %q: %v", *quoteERC20, err)
	}

	base := nativeSpec()          // EVM_NATIVE LUX (the chain's own coin)
	quote := erc20Spec(quoteAddr) // REAL ERC-20 quote

	// Build the frozen frames + validated plan. This step REFUSES any synthetic /
	// malformed asset ref (the only ones it would accept derive a well-formed canonical
	// AssetID), so we never send a poison frame.
	depositAmount := depositAmountFor(*askSize)
	// The deposit ref is derived after we know the poolID/baseID, so derive a provisional
	// plan first to learn the ids, then thread the real ref. realMarketFrames is pure and
	// cheap, so we call it twice: once to learn baseID, once with the bound ref.
	_, _, _, plan0, err := realMarketFrames(netID, cid, base, quote, *maker, *askPrice, *askSize, depositAmount, [32]byte{})
	if err != nil {
		die("derive real market (asset admission): %v", err)
	}
	depRef := depositRefFor(plan0.PoolID, plan0.BaseID, depositAmount)
	openMarket, deposit, place, plan, err := realMarketFrames(netID, cid, base, quote, *maker, *askPrice, *askSize, depositAmount, depRef)
	if err != nil {
		die("derive real market (asset admission): %v", err)
	}

	fmt.Printf("=== dexseed: REAL-market seeder ===\n")
	fmt.Printf("  node        : %s\n", node)
	fmt.Printf("  network-id  : %d\n", netID)
	fmt.Printf("  C-Chain id  : %s\n", cid)
	fmt.Printf("  base (LUX)  : kind=%s id=%s\n", base.Kind, plan.BaseID)
	fmt.Printf("  quote(ERC20): kind=%s addr=0x%s id=%s\n", quote.Kind, hex.EncodeToString(quoteAddr), plan.QuoteID)
	fmt.Printf("  poolID      : 0x%s (canonical, NON-ascii)\n", hex.EncodeToString(plan.PoolID[:]))
	fmt.Printf("  maker       : %q  ask %.6f x %.6f  deposit base=%d\n", *maker, *askPrice, *askSize, depositAmount)

	mustPost := func(step, method string, payload []byte) []byte {
		b, err := postFrame(node, method, payload)
		if err != nil {
			die("%s (%s): %v", step, method, err)
		}
		return b
	}

	// ---- 1/3 open the REAL market (bind base+quote canonical ids) ----
	mustPost("[1/3] open-market", zapwire.MethodOpenMarket, openMarket)
	fmt.Printf("[1/3] clob_open_market: bound base+quote canonical ids (real-asset market)\n")

	// ---- 2/3 fund the maker's base balance ----
	depResp := mustPost("[2/3] deposit", zapwire.MethodDeposit, deposit)
	depStatus, credited, derr := zapwire.DecodeBalanceResp(depResp)
	if derr != nil {
		die("[2/3] deposit: decode balance resp: %v (resp=%x)", derr, depResp)
	}
	fmt.Printf("[2/3] clob_deposit: maker base credited=%d (status=%d, ref=0x%s…)\n",
		credited, depStatus, hex.EncodeToString(depRef[:4]))

	// ---- 3/3 rest the maker limit ask ----
	askResp := mustPost("[3/3] place", zapwire.MethodPlace, place)
	orderID, ackStatus, aerr := zapwire.DecodeAck(askResp)
	if aerr != nil {
		die("[3/3] place: decode ack: %v (resp=%x)", aerr, askResp)
	}
	if ackStatus != zapwire.StatusPlaced {
		die("[3/3] place: ask NOT placed (status=%d, resp=%x) — the maker base deposit may be insufficient to lock the ask",
			ackStatus, askResp)
	}
	fmt.Printf("[3/3] clob_place: maker ask RESTING (orderId=%d, status=placed)\n", orderID)

	// ---- read back: confirm the market + resting order landed ----
	if *settle > 0 {
		time.Sleep(*settle)
	}
	fmt.Printf("\n=== read-back: clob_get_markets + clob_get_book ===\n")

	var mkts getMarketsResp
	if err := getJSON(node, methodGetMarkets, "", &mkts); err != nil {
		die("read clob_get_markets: %v", err)
	}
	poolHex := hex.EncodeToString(plan.PoolID[:])
	var seen *marketJSON
	for i := range mkts.Markets {
		if strings.EqualFold(strings.TrimPrefix(mkts.Markets[i].PoolID, "0x"), poolHex) {
			seen = &mkts.Markets[i]
			break
		}
	}
	if seen == nil {
		die("clob_get_markets (height=%d, count=%d) does NOT list our poolID 0x%s — market did not commit",
			mkts.Height, mkts.Count, poolHex)
	}
	fmt.Printf("  market: poolId=0x%s assetsBound=%v orders=%d bestAsk=%.6f (head height=%d)\n",
		poolHex, seen.AssetsBound, seen.Orders, seen.BestAsk, mkts.Height)
	if !seen.AssetsBound {
		die("market is listed but assetsBound=false — the (base,quote) binding did not persist")
	}

	var book getBookResp
	if err := getJSON(node, methodGetBook, "market=0x"+poolHex, &book); err != nil {
		die("read clob_get_book: %v", err)
	}
	if book.Orders < 1 || len(book.Asks) < 1 {
		die("clob_get_book shows NO resting ask (orders=%d, asks=%d) — the resting order did not land",
			book.Orders, len(book.Asks))
	}
	fmt.Printf("  book  : orders=%d remaining=%.6f bestAsk=%.6f (top ask: px=%.6f size=%.6f count=%d)\n",
		book.Orders, book.Remaining, book.BestAsk, book.Asks[0].Price, book.Asks[0].Size, book.Asks[0].Count)

	fmt.Printf("\nSEEDED: REAL market 0x%s is live on the native D-Chain with a resting maker ask.\n", poolHex)
	fmt.Printf("        base=%s quote=%s — both canonical real-asset ids (NOT ascii-of-symbol).\n", plan.BaseID, plan.QuoteID)
}

// depositAmountFor returns a maker base deposit comfortably covering the resting ask
// lock. A SELL of askSize base units locks floor(askSize) base; we credit a generous
// whole-unit cushion so the place always funds. (The native D-Chain credits and locks
// in whole base units on the deposit/place rails.)
func depositAmountFor(askSize float64) uint64 {
	amt := uint64(askSize) + 2
	if amt < 2 {
		amt = 2
	}
	return amt
}
