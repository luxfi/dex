// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command dexfill drives ONE controlled crossing into the standalone D-Chain
// venue (`dexd run`) to prove a real fill on a market BOUND to two real 20-byte
// ERC-20 token addresses (NOT a synthetic 32-byte asset id). It is a CLIENT: it
// speaks the FROZEN dex_* ZAP wire (pkg/zapwire) over github.com/luxfi/rpc and
// signs the money-moving frames exactly as dexbench / the chains/dexvm proxy /
// the precompile adapter do (pure-Go leaf, CGO_ENABLED=0).
//
// The market poolId is the canonical V4 id keccak256(abi.encode(poolKey)) — the
// SAME derivation precompile/dex PoolKey.ID() uses — so the venue market id is
// the genuine pool id the C-Chain 0x9999 settlement would map for this pair.
package main

import (
	"bytes"
	"context"
	"crypto/ecdsa"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/luxfi/crypto"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// auth constants — byte-identical to pkg/dchain/auth.go + tx.go (see dexbench).
const (
	txAuthDomain  = "lux.dchain.tx.auth.v1"
	accountDomain = "lux.dchain.account.v1"
	schemeSecp    = byte(0)
	txPlace       = byte(2)
	txSubmit      = byte(4)
)

type account struct {
	priv  *ecdsa.PrivateKey
	user  string
	nonce uint64
}

func newAccount(label string) *account {
	d := crypto.Keccak256([]byte("dexfill.secp.v1"), []byte(label))
	var priv *ecdsa.PrivateKey
	var err error
	for {
		if priv, err = crypto.ToECDSA(d); err == nil {
			break
		}
		d = crypto.Keccak256(d)
	}
	addr := crypto.PubkeyToAddress(priv.PublicKey).Bytes()
	acc := crypto.Keccak256([]byte(accountDomain), []byte{schemeSecp}, addr)[:zapwire.UserSize]
	return &account{priv: priv, user: string(acc)}
}

func (a *account) sign(typ byte, body []byte) ([]byte, error) {
	var n [8]byte
	binary.BigEndian.PutUint64(n[:], a.nonce)
	digest := crypto.Keccak256([]byte(txAuthDomain), []byte{schemeSecp}, []byte{typ}, n[:], body)
	sig, err := crypto.Sign(digest, a.priv)
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
	return append(append([]byte{}, body...), env...), nil
}

func leftPad(addr []byte) [32]byte {
	var w [32]byte
	copy(w[12:], addr)
	return w
}

func mustAddr(s string) []byte {
	s = strings.TrimPrefix(strings.ToLower(s), "0x")
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != 20 {
		fmt.Fprintf(os.Stderr, "dexfill: bad 20-byte address %q\n", s)
		os.Exit(2)
	}
	return b
}

// poolID computes the canonical V4 poolId binding the two REAL ERC-20 addresses
// (sorted currency0<currency1), matching precompile/dex PoolKey.ID():
// keccak256(abi.encode(currency0, currency1, fee, tickSpacing, hooks)).
func poolID(a, b []byte, fee uint32, tick int32, hooks []byte) (id, c0, c1 [32]byte) {
	lo, hi := a, b
	if bytes.Compare(b, a) < 0 {
		lo, hi = b, a
	}
	c0, c1 = leftPad(lo), leftPad(hi)
	var data []byte
	data = append(data, c0[:]...)
	data = append(data, c1[:]...)
	var fb [32]byte
	binary.BigEndian.PutUint32(fb[28:], fee)
	data = append(data, fb[:]...)
	var tb [32]byte
	binary.BigEndian.PutUint32(tb[28:], uint32(tick))
	data = append(data, tb[:]...)
	hb := leftPad(hooks)
	data = append(data, hb[:]...)
	copy(id[:], crypto.Keccak256(data))
	return id, c0, c1
}

func die(what string, err error) {
	fmt.Fprintf(os.Stderr, "dexfill: %s: %v\n", what, err)
	os.Exit(1)
}

func main() {
	addr := flag.String("addr", "127.0.0.1:9099", "venue ZAP endpoint host:port")
	base := flag.String("base", "", "base ERC-20 20-byte address (sold/bought)")
	quote := flag.String("quote", "", "quote ERC-20 20-byte address (priced in)")
	price := flag.Float64("price", 2000.0, "limit price (quote per base)")
	size := flag.Float64("size", 1.5, "order size (base units)")
	flag.Parse()
	if *base == "" || *quote == "" {
		fmt.Fprintln(os.Stderr, "dexfill: -base and -quote (real ERC-20 addresses) are required")
		os.Exit(2)
	}

	bb, qb := mustAddr(*base), mustAddr(*quote)
	const fee, tick = uint32(3000), int32(60)
	hooks := make([]byte, 20)
	pid, c0, c1 := poolID(bb, qb, fee, tick, hooks)

	fmt.Println("=== REAL market binding (20-byte ERC-20 addresses) ===")
	fmt.Printf("  base   : 0x%x   assetID=0x%x\n", bb, leftPad(bb))
	fmt.Printf("  quote  : 0x%x   assetID=0x%x\n", qb, leftPad(qb))
	fmt.Printf("  poolKey: currency0=0x%x currency1=0x%x fee=%d tickSpacing=%d hooks=0x%x\n", c0[12:], c1[12:], fee, tick, hooks)
	fmt.Printf("  poolId : 0x%x  (V4 keccak256(abi.encode(poolKey)))\n", pid)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	c, err := rpc.Dial(ctx, *addr)
	if err != nil {
		die("dial "+*addr, err)
	}
	defer c.Close()

	resp, err := c.CallRaw(ctx, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(pid))
	if err != nil {
		die("ensure_market", err)
	}
	if len(resp) < 9 || resp[8] != zapwire.StatusPlaced {
		die("ensure_market not placed", fmt.Errorf("%x", resp))
	}
	fmt.Println("ensure_market: OK")

	maker, taker := newAccount("dexfill.maker"), newAccount("dexfill.taker")

	pbody := zapwire.EncodePlace(pid, zapwire.SideSell, *price, *size, maker.user)
	ppl, err := maker.sign(txPlace, pbody)
	if err != nil {
		die("sign place", err)
	}
	presp, err := c.CallRaw(ctx, zapwire.MethodPlace, ppl)
	if err != nil {
		die("place", err)
	}
	oid, status, err := zapwire.DecodeAck(presp)
	if err != nil {
		die("place ack", err)
	}
	if status != zapwire.StatusPlaced {
		die("place rejected", fmt.Errorf("status=%d", status))
	}
	fmt.Printf("place  SELL %.4f @ %.4f -> orderId=%d (resting)\n", *size, *price, oid)

	sbody := zapwire.EncodeSubmit(pid, zapwire.SideBuy, false, *price, *size, taker.user)
	spl, err := taker.sign(txSubmit, sbody)
	if err != nil {
		die("sign submit", err)
	}
	sresp, err := c.CallRaw(ctx, zapwire.MethodSubmit, spl)
	if err != nil {
		die("submit", err)
	}
	fills, err := zapwire.DecodeFills(sresp)
	if err != nil {
		die("submit fills", err)
	}
	fmt.Printf("submit BUY  %.4f @ %.4f -> %d fill(s)\n", *size, *price, len(fills))
	var base0, quote0 float64
	for i, f := range fills {
		fmt.Printf("  fill %d: size=%.6f price=%.6f takerSide=%d\n", i, f.Size, f.Price, f.TakerSide)
		base0 += f.Size
		quote0 += f.Size * f.Price
	}
	if len(fills) == 0 {
		fmt.Println("RESULT: NO FILL (no crossing liquidity)")
		os.Exit(1)
	}
	fmt.Printf("RESULT: FILL — base=%.6f quote=%.6f on poolId 0x%x\n", base0, quote0, pid)
}
