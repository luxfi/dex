// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command dexbench is a ZAP load generator and cross-arch determinism checker
// for the standalone D-Chain DEX venue (`dexd run`). It is a CLIENT: it speaks
// the FROZEN dex_* ZAP wire (pkg/zapwire) over github.com/luxfi/rpc, signs every
// money-moving frame exactly as the chains/dexvm proxy and the 0x9010 precompile
// do, and measures the full mempool -> BuildBlock -> Verify -> Accept round trip.
//
// There is no `dexd bench` subcommand in the binary; this is the canonical
// load-gen the testnet harness drives. It is a pure-Go leaf (CGO_ENABLED=0): it
// imports only pkg/zapwire (the frozen wire), github.com/luxfi/crypto (secp256k1
// + keccak), and github.com/luxfi/rpc (the ZAP transport) — never the cgo/GPU
// matcher in pkg/dex, so it builds and runs on any host (the load client need not
// be the venue host).
//
// WIRE/AUTH PARITY — re-defined constants, pinned to source.
// A client cannot import the cgo-tagged pkg/dchain, so (exactly like the proxy
// relay and the precompile adapter) it RE-DEFINES the three small auth pieces.
// They MUST stay byte-identical to pkg/dchain/auth.go + pkg/dchain/tx.go:
//
//   - account fold   : keccak256(accountDomain ‖ scheme[1] ‖ addr[20])[:16]
//     (pkg/dchain/auth.go Account16 / accountDomain)
//   - auth digest    : keccak256(txAuthDomain ‖ scheme[1] ‖ type[1] ‖ nonce[8] ‖ body)
//     (pkg/dchain/auth.go txAuthDigest / txAuthDomain)
//   - auth envelope  : scheme[1] ‖ nonce[8] ‖ sigLen[2] ‖ pubLen[2] ‖ sig ‖ pub
//     (pkg/dchain/auth.go TxAuth.encode)
//   - tx-type bytes  : ensure_market=1 place=2 cancel=3 submit=4 (pkg/dchain/tx.go)
//   - RPC payload    : [frozen body] ‖ [auth envelope]   (handler.go parseTxFrame)
//
// crypto.Keccak256(a,b,...) hashes the concatenation byte-identically to the
// server's hash.ComputeKeccak256Array — same primitive, so the digests match.
//
// MODES:
//
//	dexbench -addrs h1:9099,h1:9100,h2:9099 -conns 32 -duration 30s
//	    open the venues, fan out -conns signed-write workers per venue, and
//	    report throughput + p50/p99/p999 latency over the wire.
//
//	dexbench -verify -addrs amd64box:9099,arm64box:9099
//	    drive the IDENTICAL deterministic order sequence into two venues and
//	    assert byte-identical consensus output (order ids + fills). This is the
//	    cross-arch (amd64⇄arm64) determinism proof over the live boxes.
package main

import (
	"context"
	"crypto/ecdsa"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/luxfi/crypto"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// --- auth constants re-defined from pkg/dchain (see package comment) ----------

const (
	txAuthDomain  = "lux.dchain.tx.auth.v1" // pkg/dchain/auth.go
	accountDomain = "lux.dchain.account.v1" // pkg/dchain/auth.go
	schemeSecp    = byte(0)                 // dex.AuthSecp256k1 (pkg/dex/auth.go)
	txEnsure      = byte(1)                 // TxEnsureMarket (pkg/dchain/tx.go)
	txPlace       = byte(2)                 // TxPlace
	txSubmit      = byte(4)                 // TxSubmit
)

// account is one signing identity: a deterministic secp256k1 key, its derived
// 16-byte settlement account (the zapwire user[16] field), and a monotone nonce.
// Each worker owns its own account so its nonce stream is strictly sequential
// (the consensus gate enforces monotone per-account nonces; one in-flight signed
// write per account keeps the stream gap-free).
type account struct {
	priv  *ecdsa.PrivateKey
	user  string // string(account16[:]) — the wire user field
	nonce uint64
}

// newAccount derives a deterministic identity from a seed label. Deterministic
// derivation is what makes -verify reproducible across hosts: the same label
// yields the same key, account, and signatures on the amd64 and arm64 boxes.
func newAccount(label string) (*account, error) {
	d := crypto.Keccak256([]byte("dexbench.secp.v1"), []byte(label))
	var priv *ecdsa.PrivateKey
	var err error
	for { // re-hash on the vanishingly rare out-of-range scalar (matches tests)
		priv, err = crypto.ToECDSA(d)
		if err == nil {
			break
		}
		d = crypto.Keccak256(d)
	}
	addr := crypto.PubkeyToAddress(priv.PublicKey).Bytes() // 20-byte keccak address
	acc := crypto.Keccak256([]byte(accountDomain), []byte{schemeSecp}, addr)[:zapwire.UserSize]
	return &account{priv: priv, user: string(acc)}, nil
}

// sign returns the auth envelope for (typ, nonce, body) under the account key and
// advances the nonce. The envelope is appended to the frozen body to form the RPC
// payload.
func (a *account) sign(typ byte, body []byte) ([]byte, error) {
	var n [8]byte
	binary.BigEndian.PutUint64(n[:], a.nonce)
	digest := crypto.Keccak256([]byte(txAuthDomain), []byte{schemeSecp}, []byte{typ}, n[:], body)
	sig, err := crypto.Sign(digest, a.priv) // 65-byte r‖s‖v
	if err != nil {
		return nil, err
	}
	env := make([]byte, 0, 1+8+2+2+len(sig))
	env = append(env, schemeSecp)
	env = append(env, n[:]...)
	var l [2]byte
	binary.BigEndian.PutUint16(l[:], uint16(len(sig)))
	env = append(env, l[:]...)
	env = append(env, 0, 0) // pubLen = 0 (secp recovers the key)
	env = append(env, sig...)
	a.nonce++
	out := make([]byte, 0, len(body)+len(env))
	out = append(out, body...)
	return append(out, env...), nil
}

// marketID deterministically maps a market index to a 32-byte poolId. The venue
// accepts any 32-byte poolId; deterministic ids keep -verify reproducible.
func marketID(i int) [32]byte {
	var p [32]byte
	copy(p[:], crypto.Keccak256([]byte(fmt.Sprintf("dexbench.market.%d", i))))
	return p
}

type config struct {
	addrs    []string
	conns    int
	markets  int
	duration time.Duration
	warmup   time.Duration
	op       string
	size     float64
	price    float64
	verify   bool
	dialTO   time.Duration
	callTO   time.Duration
}

func main() {
	var (
		addrs    = flag.String("addrs", "127.0.0.1:9099", "comma-separated venue ZAP endpoints (host:port)")
		conns    = flag.Int("conns", 16, "signed-write worker connections PER endpoint")
		markets  = flag.Int("markets", 4, "distinct markets each worker spreads orders across")
		duration = flag.Duration("duration", 30*time.Second, "measured load duration")
		warmup   = flag.Duration("warmup", 3*time.Second, "warmup excluded from stats")
		op       = flag.String("op", "place", "load op: place (resting write) | cross (place+submit, exercises matcher)")
		size     = flag.Float64("size", 1.0, "order size")
		price    = flag.Float64("price", 100.0, "base price")
		verify   = flag.Bool("verify", false, "determinism mode: drive identical seq into 2 endpoints, compare output")
		dialTO   = flag.Duration("dial-timeout", 10*time.Second, "per-endpoint dial timeout")
		callTO   = flag.Duration("call-timeout", 30*time.Second, "per-call timeout")
	)
	flag.Parse()

	cfg := config{
		addrs:    splitAddrs(*addrs),
		conns:    *conns,
		markets:  *markets,
		duration: *duration,
		warmup:   *warmup,
		op:       *op,
		size:     *size,
		price:    *price,
		verify:   *verify,
		dialTO:   *dialTO,
		callTO:   *callTO,
	}
	if len(cfg.addrs) == 0 {
		fmt.Fprintln(os.Stderr, "dexbench: -addrs is required")
		os.Exit(2)
	}
	if cfg.markets < 1 {
		cfg.markets = 1
	}

	if cfg.verify {
		if err := runVerify(cfg); err != nil {
			fmt.Fprintf(os.Stderr, "dexbench verify: %v\n", err)
			os.Exit(1)
		}
		return
	}
	if err := runLoad(cfg); err != nil {
		fmt.Fprintf(os.Stderr, "dexbench load: %v\n", err)
		os.Exit(1)
	}
}

func splitAddrs(s string) []string {
	var out []string
	for _, a := range strings.Split(s, ",") {
		if a = strings.TrimSpace(a); a != "" {
			out = append(out, a)
		}
	}
	return out
}

// dial opens a ZAP client to addr (the default rpc transport is ZAP).
func dial(ctx context.Context, addr string) (rpc.Client, error) {
	return rpc.Dial(ctx, addr)
}

// ensureMarket creates a market (admin frame: body only, no auth envelope).
func ensureMarket(ctx context.Context, c rpc.Client, pool [32]byte) error {
	resp, err := c.CallRaw(ctx, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(pool))
	if err != nil {
		return err
	}
	if len(resp) < 9 || resp[8] != zapwire.StatusPlaced {
		return fmt.Errorf("ensure_market not placed: %x", resp)
	}
	return nil
}

// placeSell rests a SELL limit order. In a load made only of resting SELLs there
// is never an opposite bid to cross, so every order rests — a clean, sustainable
// unit for measuring the consensus-write + wire round trip.
func placeSell(ctx context.Context, c rpc.Client, a *account, pool [32]byte, price, size float64) (uint64, error) {
	body := zapwire.EncodePlace(pool, zapwire.SideSell, price, size, a.user)
	payload, err := a.sign(txPlace, body)
	if err != nil {
		return 0, err
	}
	resp, err := c.CallRaw(ctx, zapwire.MethodPlace, payload)
	if err != nil {
		return 0, err
	}
	id, status, err := zapwire.DecodeAck(resp)
	if err != nil {
		return 0, err
	}
	if status != zapwire.StatusPlaced {
		return 0, fmt.Errorf("place rejected: status=%d resp=%x", status, resp)
	}
	return id, nil
}

// submitBuy crosses the book with a marketable BUY and returns the fills.
func submitBuy(ctx context.Context, c rpc.Client, a *account, pool [32]byte, limit, size float64) ([]zapwire.Fill, error) {
	body := zapwire.EncodeSubmit(pool, zapwire.SideBuy, false, limit, size, a.user)
	payload, err := a.sign(txSubmit, body)
	if err != nil {
		return nil, err
	}
	resp, err := c.CallRaw(ctx, zapwire.MethodSubmit, payload)
	if err != nil {
		return nil, err
	}
	return zapwire.DecodeFills(resp)
}

// --- load mode ---------------------------------------------------------------

const sampleCap = 1 << 20 // per-worker latency samples retained for percentiles

type workerStat struct {
	ops      uint64
	errs     uint64
	fills    uint64
	latNanos []int64
}

func runLoad(cfg config) error {
	fmt.Printf("dexbench load: endpoints=%v conns/endpoint=%d markets=%d op=%s duration=%s warmup=%s\n",
		cfg.addrs, cfg.conns, cfg.markets, cfg.op, cfg.duration, cfg.warmup)

	type wkr struct {
		addr string
		idx  int
	}
	var workers []wkr
	for _, addr := range cfg.addrs {
		for i := 0; i < cfg.conns; i++ {
			workers = append(workers, wkr{addr: addr, idx: i})
		}
	}

	stats := make([]workerStat, len(workers))
	var started int64
	var measuring int32
	stop := make(chan struct{})
	var wg sync.WaitGroup

	dialCtx, dialCancel := context.WithTimeout(context.Background(), cfg.dialTO)
	defer dialCancel()

	for wi := range workers {
		w := workers[wi]
		st := &stats[wi]
		st.latNanos = make([]int64, 0, 1<<16)

		c, err := dial(dialCtx, w.addr)
		if err != nil {
			return fmt.Errorf("dial %s: %w", w.addr, err)
		}
		acc, err := newAccount(fmt.Sprintf("%s#%d", w.addr, w.idx))
		if err != nil {
			return err
		}
		// Ensure this worker's markets exist (idempotent). For cross mode also seed
		// a deep resting SELL the taker submits against.
		setupCtx, setupCancel := context.WithTimeout(context.Background(), cfg.callTO)
		for m := 0; m < cfg.markets; m++ {
			if err := ensureMarket(setupCtx, c, marketID(m)); err != nil {
				setupCancel()
				return fmt.Errorf("worker %s#%d ensure market %d: %w", w.addr, w.idx, m, err)
			}
		}
		setupCancel()

		wg.Add(1)
		go func() {
			defer wg.Done()
			defer c.Close()
			atomic.AddInt64(&started, 1)
			runWorker(cfg, c, acc, st, &measuring, stop)
		}()
	}

	// Wait for all workers to connect+setup, then run warmup, flip the measuring
	// flag, run the measured window, and stop.
	for atomic.LoadInt64(&started) < int64(len(workers)) {
		time.Sleep(5 * time.Millisecond)
	}
	if cfg.warmup > 0 {
		time.Sleep(cfg.warmup)
	}
	t0 := time.Now()
	atomic.StoreInt32(&measuring, 1)
	time.Sleep(cfg.duration)
	atomic.StoreInt32(&measuring, 0)
	elapsed := time.Since(t0)
	close(stop)
	wg.Wait()

	report(cfg, stats, elapsed)
	return nil
}

// runWorker drives signed writes as fast as the venue accepts them (closed-loop:
// one in-flight signed op per account, so the nonce stream stays sequential).
// Latency is recorded only while measuring==1.
func runWorker(cfg config, c rpc.Client, acc *account, st *workerStat, measuring *int32, stop <-chan struct{}) {
	var maker *account
	if cfg.op == "cross" {
		maker, _ = newAccount("maker:" + acc.user)
		// Seed a deep resting SELL per market so taker BUYs cross it.
		for m := 0; m < cfg.markets; m++ {
			ctx, cancel := context.WithTimeout(context.Background(), cfg.callTO)
			_, _ = placeSell(ctx, c, maker, marketID(m), cfg.price, cfg.size*float64(1<<20))
			cancel()
		}
	}
	mkt := 0
	for {
		select {
		case <-stop:
			return
		default:
		}
		pool := marketID(mkt)
		mkt = (mkt + 1) % cfg.markets
		ctx, cancel := context.WithTimeout(context.Background(), cfg.callTO)
		start := time.Now()
		var opErr error
		var nFills int
		switch cfg.op {
		case "cross":
			fills, err := submitBuy(ctx, c, acc, pool, cfg.price, cfg.size)
			opErr = err
			nFills = len(fills)
			if err == nil && nFills == 0 {
				// Liquidity drained: re-seed and retry next loop.
				_, _ = placeSell(ctx, c, maker, pool, cfg.price, cfg.size*float64(1<<20))
			}
		default: // place
			_, opErr = placeSell(ctx, c, acc, pool, cfg.price+1, cfg.size)
		}
		lat := time.Since(start).Nanoseconds()
		cancel()

		if atomic.LoadInt32(measuring) == 1 {
			if opErr != nil {
				st.errs++
				continue
			}
			st.ops++
			st.fills += uint64(nFills)
			if len(st.latNanos) < sampleCap {
				st.latNanos = append(st.latNanos, lat)
			}
		} else if opErr != nil {
			// pre/post window error: surface once via errs so a broken run is visible
			st.errs++
		}
	}
}

func report(cfg config, stats []workerStat, elapsed time.Duration) {
	var totOps, totErr, totFills uint64
	var lat []int64
	for i := range stats {
		totOps += stats[i].ops
		totErr += stats[i].errs
		totFills += stats[i].fills
		lat = append(lat, stats[i].latNanos...)
	}
	sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })

	thru := float64(totOps) / elapsed.Seconds()
	fmt.Printf("\n=== dexbench RESULT ===\n")
	fmt.Printf("  op             : %s\n", cfg.op)
	fmt.Printf("  endpoints      : %d  (%v)\n", len(cfg.addrs), cfg.addrs)
	fmt.Printf("  workers        : %d  (%d/endpoint)\n", len(cfg.addrs)*cfg.conns, cfg.conns)
	fmt.Printf("  elapsed        : %s\n", elapsed.Round(time.Millisecond))
	fmt.Printf("  ops (acked)    : %d\n", totOps)
	fmt.Printf("  errors         : %d\n", totErr)
	if cfg.op == "cross" {
		fmt.Printf("  fills          : %d\n", totFills)
	}
	fmt.Printf("  throughput     : %.0f ops/sec\n", thru)
	if len(lat) > 0 {
		fmt.Printf("  latency (µs)   : p50=%.1f  p90=%.1f  p99=%.1f  p999=%.1f  max=%.1f  (n=%d)\n",
			pct(lat, 0.50)/1e3, pct(lat, 0.90)/1e3, pct(lat, 0.99)/1e3, pct(lat, 0.999)/1e3,
			float64(lat[len(lat)-1])/1e3, len(lat))
	}
	fmt.Println()
}

// pct returns the p-quantile of a pre-sorted slice (nearest-rank).
func pct(sorted []int64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(p * float64(len(sorted)))
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return float64(sorted[i])
}

// --- verify (cross-arch determinism) mode ------------------------------------

// runVerify drives the IDENTICAL deterministic order sequence into every endpoint
// and asserts byte-identical consensus output. Each standalone venue bootstraps
// the same deterministic genesis and runs the same deterministic VM, so identical
// ordered input MUST yield identical order ids and fills regardless of host arch.
// This is the live amd64⇄arm64 determinism proof.
func runVerify(cfg config) error {
	if len(cfg.addrs) < 2 {
		return fmt.Errorf("verify needs >=2 endpoints (e.g. amd64box:9099,arm64box:9099)")
	}
	fmt.Printf("dexbench verify: driving identical order sequence into %v\n", cfg.addrs)

	ctx, cancel := context.WithTimeout(context.Background(), cfg.callTO*4)
	defer cancel()

	type conn struct {
		addr string
		c    rpc.Client
		// one signing identity per endpoint, derived from the SAME label so the
		// signatures (hence tx ids) are identical across endpoints.
		maker *account
		taker *account
	}
	var conns []*conn
	for _, addr := range cfg.addrs {
		c, err := dial(ctx, addr)
		if err != nil {
			return fmt.Errorf("dial %s: %w", addr, err)
		}
		defer c.Close()
		mk, _ := newAccount("verify.maker")
		tk, _ := newAccount("verify.taker")
		conns = append(conns, &conn{addr: addr, c: c, maker: mk, taker: tk})
	}

	pool := marketID(0)
	// Deterministic two-sided book then a crossing taker — mirrors the e2e.
	type step struct {
		kind  string // "ensure" | "place" | "submit"
		side  uint8
		price float64
		size  float64
	}
	steps := []step{
		{kind: "ensure"},
		{kind: "place", side: zapwire.SideSell, price: 100.0, size: 10.0},
		{kind: "place", side: zapwire.SideSell, price: 101.0, size: 5.0},
		{kind: "place", side: zapwire.SideSell, price: 102.0, size: 7.0},
		{kind: "submit", side: zapwire.SideBuy, price: 102.0, size: 18.0},
	}

	// Collect each endpoint's observable output (order ids + fills) per step.
	type outcome struct {
		ackID uint64
		fills []zapwire.Fill
	}
	results := make([][]outcome, len(conns))

	for ci, cn := range conns {
		for _, s := range steps {
			switch s.kind {
			case "ensure":
				if err := ensureMarket(ctx, cn.c, pool); err != nil {
					return fmt.Errorf("%s ensure: %w", cn.addr, err)
				}
				results[ci] = append(results[ci], outcome{})
			case "place":
				body := zapwire.EncodePlace(pool, s.side, s.price, s.size, cn.maker.user)
				payload, err := cn.maker.sign(txPlace, body)
				if err != nil {
					return err
				}
				resp, err := cn.c.CallRaw(ctx, zapwire.MethodPlace, payload)
				if err != nil {
					return fmt.Errorf("%s place: %w", cn.addr, err)
				}
				id, _, err := zapwire.DecodeAck(resp)
				if err != nil {
					return fmt.Errorf("%s place ack: %w", cn.addr, err)
				}
				results[ci] = append(results[ci], outcome{ackID: id})
			case "submit":
				body := zapwire.EncodeSubmit(pool, s.side, false, s.price, s.size, cn.taker.user)
				payload, err := cn.taker.sign(txSubmit, body)
				if err != nil {
					return err
				}
				resp, err := cn.c.CallRaw(ctx, zapwire.MethodSubmit, payload)
				if err != nil {
					return fmt.Errorf("%s submit: %w", cn.addr, err)
				}
				fills, err := zapwire.DecodeFills(resp)
				if err != nil {
					return fmt.Errorf("%s submit fills: %w", cn.addr, err)
				}
				results[ci] = append(results[ci], outcome{fills: fills})
			}
		}
	}

	// Compare every endpoint against endpoint 0.
	base := results[0]
	mismatch := 0
	for ci := 1; ci < len(results); ci++ {
		for si := range base {
			a, b := base[si], results[ci][si]
			if a.ackID != b.ackID || !fillsEqual(a.fills, b.fills) {
				mismatch++
				fmt.Printf("  MISMATCH step=%d %s: %s=%+v  %s=%+v\n",
					si, steps[si].kind, cfg.addrs[0], a, cfg.addrs[ci], b)
			}
		}
	}

	fmt.Printf("\n=== dexbench VERIFY ===\n")
	fmt.Printf("  endpoints : %v\n", cfg.addrs)
	fmt.Printf("  steps     : %d\n", len(steps))
	// Report the crossing fills observed (same on every endpoint when PASS).
	for si, s := range steps {
		if s.kind == "submit" {
			fmt.Printf("  submit fills @%s : %d fill(s)\n", cfg.addrs[0], len(base[si].fills))
			for fi, f := range base[si].fills {
				fmt.Printf("    fill %d: %.4f @ %.4f (takerSide=%d)\n", fi, f.Size, f.Price, f.TakerSide)
			}
		}
	}
	if mismatch == 0 {
		fmt.Printf("  RESULT    : PASS — byte-identical consensus output across all endpoints (cross-arch determinism holds)\n\n")
		return nil
	}
	return fmt.Errorf("%d mismatched outputs across endpoints (determinism BROKEN)", mismatch)
}

func fillsEqual(a, b []zapwire.Fill) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
