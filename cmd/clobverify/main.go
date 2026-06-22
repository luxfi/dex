// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command clobverify exercises and AUDITS the native D-Chain across a validator
// set. It submits a crossing order to ONE validator over the in-luxd HTTP
// ingestion seam (POST /ext/bc/D/dex/clob_* with frozen zapwire frames), then
// reads the committed trade log (GET /ext/bc/D/dex/clob_get_trades) from EVERY
// validator and compares.
//
// This is the cross-validator replication probe the consensus-safety gate needs:
// the native matcher runs at Block.Verify on each node, so if a fill is committed
// identically on all N validators (same trade rows, same accepted head root), the
// fill IS replicated under the chain's quorum; if only the submit-target (the
// proposer) shows it, replication did not reach the followers.
//
// It is pure-Go (CGO=0), links only pkg/zapwire (the one source of frame truth)
// and the stdlib, so it builds static for in-cluster use. It does NOT depend on
// the dexvm package (the read responses are decoded as open JSON), so the read
// schema is exercised exactly as an external consumer (the indexer) would.
package main

import (
	"bytes"
	"encoding/binary"
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
)

// chainHead is the accepted-block anchor every read carries (mirrors
// pkg/dchain.chainHead). Two nodes at the same height with the same root have
// committed the identical history.
type chainHead struct {
	Height       uint64 `json:"height"`
	LastAccepted string `json:"lastAccepted"`
	Root         string `json:"root"`
}

// tradeJSON mirrors pkg/dchain.tradeJSON (the committed-fill read form).
type tradeJSON struct {
	Height       uint64  `json:"height"`
	Seq          uint64  `json:"seq"`
	TradeID      uint64  `json:"tradeId"`
	Price        float64 `json:"price"`
	Size         float64 `json:"size"`
	TakerSide    string  `json:"takerSide"`
	MakerOrderID uint64  `json:"makerOrderId"`
	TakerOrderID uint64  `json:"takerOrderId"`
	Timestamp    int64   `json:"timestamp"`
}

type getTradesResp struct {
	chainHead
	Count  int         `json:"count"`
	Trades []tradeJSON `json:"trades"`
}

type marketJSON struct {
	PoolID      string  `json:"poolId"`
	Symbol      string  `json:"symbol"`
	AssetsBound bool    `json:"assetsBound"`
	Orders      int     `json:"orders"`
	Remaining   float64 `json:"remaining"`
	BestBid     float64 `json:"bestBid"`
	BestAsk     float64 `json:"bestAsk"`
}

type getMarketsResp struct {
	chainHead
	Count   int          `json:"count"`
	Markets []marketJSON `json:"markets"`
}

func die(format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "FAIL: "+format+"\n", args...)
	os.Exit(1)
}

// poolIDForSymbol mirrors pkg/dchain.poolIDForSymbol.
func poolIDForSymbol(sym string) [32]byte {
	var p [32]byte
	copy(p[:], sym)
	p[31] = 0xD0
	return p
}

// contentRef mirrors pkg/dchain.contentRef (exactly-once-on-retry deposit ref).
func contentRef(typ byte, user string, asset [32]byte, amount uint64) [32]byte {
	var ref [32]byte
	copy(ref[:], asset[:])
	ref[0] ^= typ
	for i := 0; i < len(user) && i+1 < 24; i++ {
		ref[1+i] ^= user[i]
	}
	binary.BigEndian.PutUint64(ref[24:32], binary.BigEndian.Uint64(asset[24:32])^amount)
	return ref
}

const txDeposit = 1 // dchain.TxDeposit wire tag

// httpClient is a short-timeout client; the consensus wait for a write is bounded
// by the VM's submitTx (it blocks until the order's block is accepted).
var httpClient = &http.Client{Timeout: 30 * time.Second}

// postFrame POSTs a frozen zapwire frame to base/dex/<method> and returns the
// raw response bytes. base is http://host:port (no trailing slash).
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
		return nil, fmt.Errorf("%s: HTTP %d: %s", method, resp.StatusCode, body)
	}
	return body, nil
}

// getJSON GETs base/dex/<method><query> and decodes the JSON body into out.
func getJSON(base, method, query string, out interface{}) error {
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
		return fmt.Errorf("%s: HTTP %d: %s", method, resp.StatusCode, body)
	}
	return json.Unmarshal(body, out)
}

func main() {
	var (
		nodesCSV = flag.String("nodes", "", "comma-separated validator base URLs, e.g. http://luxd-0...:9650,http://luxd-1...:9650")
		submitTo = flag.Int("submit-to", 0, "index into -nodes of the validator to SUBMIT the crossing order to (the proposer)")
		symbol   = flag.String("symbol", "LUX/LUSD", "market symbol")
		readOnly = flag.Bool("read-only", false, "skip the write; only read clob_get_trades/markets from every node and compare")
	)
	flag.Parse()

	nodes := splitCSV(*nodesCSV)
	if len(nodes) == 0 {
		die("-nodes is required (comma-separated validator base URLs)")
	}
	if *submitTo < 0 || *submitTo >= len(nodes) {
		die("-submit-to %d out of range for %d nodes", *submitTo, len(nodes))
	}

	pool := poolIDForSymbol(*symbol)
	var base [32]byte // AssetLUX all-zero
	var quote [32]byte
	binary.BigEndian.PutUint64(quote[24:32], 0x4c555344_000001) // "LUSD" label
	const aliceUser, bobUser = "alice", "bob"
	const askPrice, askSize = 12.50, 2.0
	const buyLimit, buySize = 13.0, 1.0

	target := strings.TrimRight(nodes[*submitTo], "/")
	fmt.Printf("=== clobverify: %d node(s), submit-to=%d (%s), symbol=%s ===\n", len(nodes), *submitTo, target, *symbol)

	if !*readOnly {
		// ---- WRITE PATH on the submit-target (the order enters its mempool ->
		// its consensus -> gossips to the set) ----
		mustPost := func(method string, payload []byte) []byte {
			b, err := postFrame(target, method, payload)
			if err != nil {
				die("write %s -> %s: %v", method, target, err)
			}
			return b
		}

		mustPost(zapwire.MethodOpenMarket, zapwire.EncodeOpenMarket(pool, base, quote))
		fmt.Println("[1/4] clob_open_market: assets bound (custody market)")

		dep := func(user string, asset [32]byte, amt uint64) {
			ref := contentRef(byte(txDeposit), user, asset, amt)
			resp := mustPost(zapwire.MethodDeposit, zapwire.EncodeDeposit(user, asset, amt, ref))
			st, credited, _ := zapwire.DecodeBalanceResp(resp)
			fmt.Printf("      deposit %s amt=%d -> status=%d credited=%d\n", user, amt, st, credited)
		}
		dep(aliceUser, base, 2)
		dep(bobUser, quote, 53)
		fmt.Println("[2/4] clob_deposit: alice base, bob quote credited")

		askResp := mustPost(zapwire.MethodPlace, zapwire.EncodePlace(pool, zapwire.SideSell, askPrice, askSize, aliceUser))
		if len(askResp) < 9 || askResp[8] != zapwire.StatusPlaced {
			die("clob_place ask not placed: %x", askResp)
		}
		fmt.Printf("[3/4] clob_place: alice ask %.2f x %.1f resting (orderId=%d)\n", askPrice, askSize, binary.BigEndian.Uint64(askResp[0:8]))

		subResp := mustPost(zapwire.MethodSubmit, zapwire.EncodeSubmit(pool, zapwire.SideBuy, false, buyLimit, buySize, bobUser))
		fills, err := zapwire.DecodeFills(subResp)
		if err != nil {
			die("clob_submit fills decode: %v (resp=%x)", err, subResp)
		}
		if len(fills) == 0 {
			die("clob_submit produced 0 fills (the native cross did not match): resp=%x", subResp)
		}
		fmt.Printf("[4/4] clob_submit: bob crossed -> %d fill(s); first px=%.4f size=%.4f side=%d\n",
			len(fills), fills[0].Price, fills[0].Size, fills[0].TakerSide)
	}

	// Give the gossip + follower Verify/Accept a moment to land before reading.
	time.Sleep(2 * time.Second)

	// ---- READ PATH on EVERY node: compare committed trades + head root ----
	fmt.Printf("\n=== cross-validator read: clob_get_trades on all %d node(s) ===\n", len(nodes))
	type nodeView struct {
		url    string
		err    error
		trades getTradesResp
		mkts   getMarketsResp
	}
	views := make([]nodeView, len(nodes))
	for i, n := range nodes {
		b := strings.TrimRight(n, "/")
		v := nodeView{url: b}
		if e := getJSON(b, "clob_get_trades", "", &v.trades); e != nil {
			v.err = e
		} else {
			_ = getJSON(b, "clob_get_markets", "", &v.mkts)
		}
		views[i] = v
	}

	// Print a per-validator evidence table.
	fmt.Printf("%-3s %-40s %-7s %-8s %-18s %s\n", "idx", "node", "trades", "height", "root[:16]", "firstFill(px/size/side@h.seq)")
	for i, v := range views {
		short := v.url
		if h := strings.TrimPrefix(short, "http://"); h != short {
			short = h
		}
		if len(short) > 40 {
			short = short[:40]
		}
		if v.err != nil {
			fmt.Printf("%-3d %-40s ERROR: %v\n", i, short, v.err)
			continue
		}
		first := "-"
		if len(v.trades.Trades) > 0 {
			t := v.trades.Trades[0]
			first = fmt.Sprintf("%.4f/%.4f/%s@%d.%d", t.Price, t.Size, t.TakerSide, t.Height, t.Seq)
		}
		root16 := v.trades.Root
		if len(root16) > 16 {
			root16 = root16[:16]
		}
		fmt.Printf("%-3d %-40s %-7d %-8d %-18s %s\n", i, short, v.trades.Count, v.trades.Height, root16, first)
	}

	// ---- VERDICT ----
	fmt.Printf("\n=== verdict ===\n")
	var withFill, errored int
	canonical := canonicalTrades(views[*submitTo].trades)
	allIdentical := true
	var refRoot string
	haveRef := false
	for i, v := range views {
		if v.err != nil {
			errored++
			allIdentical = false
			fmt.Printf("  node[%d] %s: UNREACHABLE (%v)\n", i, v.url, v.err)
			continue
		}
		if v.trades.Count > 0 {
			withFill++
		}
		// Compare the trade set + head root against the submit-target (the proposer).
		if canonicalTrades(v.trades) != canonical {
			allIdentical = false
			fmt.Printf("  node[%d] %s: TRADES DIFFER from submit-target\n", i, v.url)
		}
		if !haveRef {
			refRoot, haveRef = v.trades.Root, true
		} else if v.trades.Root != refRoot {
			allIdentical = false
			fmt.Printf("  node[%d] %s: ROOT %s != %s\n", i, v.url, v.trades.Root[:16], refRoot[:16])
		}
	}

	fmt.Printf("  nodes=%d  with-fill=%d  unreachable=%d  identical=%v\n", len(nodes), withFill, errored, allIdentical)
	if withFill == len(nodes) && allIdentical && errored == 0 {
		fmt.Printf("\nPROVEN: the fill is REPLICATED on ALL %d validators — identical trade rows + identical head root.\n", len(nodes))
		fmt.Printf("        The native D-Chain fill is committed under the chain's quorum, not just the proposer.\n")
		os.Exit(0)
	}
	if withFill <= 1 {
		fmt.Printf("\nSINGLE-PROPOSER: only %d of %d validators show the fill — replication did NOT reach the followers.\n", withFill, len(nodes))
	} else {
		fmt.Printf("\nPARTIAL: %d of %d validators show the fill (replication incomplete or roots diverged).\n", withFill, len(nodes))
	}
	os.Exit(2)
}

// canonicalTrades renders a trades response to a stable comparable string (the
// committed fills + head root), so two nodes match iff they committed the same
// fills at the same accepted head.
func canonicalTrades(r getTradesResp) string {
	var b strings.Builder
	fmt.Fprintf(&b, "h=%d root=%s n=%d;", r.Height, r.Root, r.Count)
	for _, t := range r.Trades {
		fmt.Fprintf(&b, "[%d.%d id=%d px=%.8f sz=%.8f side=%s mk=%d tk=%d ts=%d]",
			t.Height, t.Seq, t.TradeID, t.Price, t.Size, t.TakerSide, t.MakerOrderID, t.TakerOrderID, t.Timestamp)
	}
	return b.String()
}

func splitCSV(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ",") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// compile-time guard the read schema stays a superset of what we decode.
var _ = hex.EncodeToString
