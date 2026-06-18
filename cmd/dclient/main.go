// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command dclient is the PATH-B (native ZAP) e2e driver against a live D-Chain
// venue. It speaks the FROZEN clob_* surface over github.com/luxfi/rpc — the
// exact wire frames the 0x9010 EVM precompile and the chains/dexvm proxy use —
// to:
//
//   1. ensure_market LUX/LUSD and LUX/LETH,
//   2. place resting maker liquidity (clob_place),
//   3. cross with a marketable taker (clob_submit) and assert the
//      consensus-computed fills,
//   4. query the resulting book (clob_depth).
//
// Every write crosses the venue's full match -> Verify -> Accept consensus path;
// the fills returned are what the chain re-derived, not a local mutation.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"time"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

func poolID(sym string) [32]byte {
	var p [32]byte
	copy(p[:], sym)
	p[31] = 0xD0
	return p
}

func depth(ctx context.Context, c *rpc.ZAPConn, sym string) (orders int, remaining, bestBid, bestAsk float64, found bool, err error) {
	id := poolID(sym)
	resp, e := c.Call(ctx, "clob_depth", id[:])
	if e != nil {
		return 0, 0, 0, 0, false, e
	}
	if len(resp) < 29 {
		return 0, 0, 0, 0, false, fmt.Errorf("short depth resp %d", len(resp))
	}
	orders = int(binary.BigEndian.Uint32(resp[0:4]))
	remaining = math.Float64frombits(binary.BigEndian.Uint64(resp[4:12]))
	bestBid = math.Float64frombits(binary.BigEndian.Uint64(resp[12:20]))
	bestAsk = math.Float64frombits(binary.BigEndian.Uint64(resp[20:28]))
	found = resp[28] == 1
	return
}

func ensure(ctx context.Context, c *rpc.ZAPConn, sym string) error {
	id := poolID(sym)
	resp, err := c.Call(ctx, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(id))
	if err != nil {
		return err
	}
	_, status, derr := zapwire.DecodeAck(resp)
	if derr != nil {
		return derr
	}
	if status == zapwire.StatusRejected {
		return fmt.Errorf("ensure_market %s rejected", sym)
	}
	return nil
}

func place(ctx context.Context, c *rpc.ZAPConn, sym string, side uint8, price, size float64, user string) (uint64, error) {
	id := poolID(sym)
	resp, err := c.Call(ctx, zapwire.MethodPlace, zapwire.EncodePlace(id, side, price, size, user))
	if err != nil {
		return 0, err
	}
	orderID, status, derr := zapwire.DecodeAck(resp)
	if derr != nil {
		return 0, derr
	}
	if status == zapwire.StatusRejected {
		return 0, fmt.Errorf("place %s rejected", sym)
	}
	return orderID, nil
}

func submit(ctx context.Context, c *rpc.ZAPConn, sym string, side uint8, isMarket bool, limit, size float64) ([]zapwire.Fill, error) {
	id := poolID(sym)
	resp, err := c.Call(ctx, zapwire.MethodSubmit, zapwire.EncodeSubmit(id, side, isMarket, limit, size, ""))
	if err != nil {
		return nil, err
	}
	return zapwire.DecodeFills(resp)
}

func main() {
	addr := flag.String("addr", "127.0.0.1:9099", "venue ZAP address")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	conn, err := rpc.ZAPDial(ctx, *addr)
	if err != nil {
		fmt.Printf("FAIL ZAPDial(%s): %v\n", *addr, err)
		os.Exit(1)
	}
	defer conn.Close()
	fmt.Printf("PATH-B dialed venue %s\n", *addr)

	// 1. ensure both markets.
	for _, sym := range []string{"LUX/LUSD", "LUX/LETH"} {
		if err := ensure(ctx, conn, sym); err != nil {
			fmt.Printf("FAIL ensure %s: %v\n", sym, err)
			os.Exit(1)
		}
		fmt.Printf("ensure_market %-9s OK\n", sym)
	}

	// 2. seed resting liquidity.
	//    LUX/LUSD: asks 10@100, 5@101 ; bids 8@99, 12@98  (4 orders / 35.0).
	//    LUX/LETH: asks 4@0.05, 6@0.051 ; bid 3@0.049     (3 orders / 13.0).
	const sell, buy = uint8(1), uint8(0)
	seed := []struct {
		sym   string
		side  uint8
		price float64
		size  float64
	}{
		{"LUX/LUSD", sell, 100, 10}, {"LUX/LUSD", sell, 101, 5},
		{"LUX/LUSD", buy, 99, 8}, {"LUX/LUSD", buy, 98, 12},
		{"LUX/LETH", sell, 0.05, 4}, {"LUX/LETH", sell, 0.051, 6},
		{"LUX/LETH", buy, 0.049, 3},
	}
	for _, s := range seed {
		oid, err := place(ctx, conn, s.sym, s.side, s.price, s.size, "makerA")
		if err != nil {
			fmt.Printf("FAIL place %s %v@%v: %v\n", s.sym, s.size, s.price, err)
			os.Exit(1)
		}
		fmt.Printf("place %-9s side=%d %5.3f@%-7.3f -> orderID=%d\n", s.sym, s.side, s.size, s.price, oid)
	}

	// Report seeded books.
	for _, sym := range []string{"LUX/LUSD", "LUX/LETH"} {
		n, rem, bb, ba, _, err := depth(ctx, conn, sym)
		if err != nil {
			fmt.Printf("FAIL depth %s: %v\n", sym, err)
			os.Exit(1)
		}
		fmt.Printf("BOOK   %-9s seeded: orders=%d remaining=%.3f bestBid=%.4f bestAsk=%.4f\n", sym, n, rem, bb, ba)
	}

	// 3. cross LUX/LUSD with a marketable BUY 12 @ limit 101.
	//    Expect 2 fills: 10@100 then 2@101 (price-time priority, multi-level VWAP).
	fills, err := submit(ctx, conn, "LUX/LUSD", buy, false, 101, 12)
	if err != nil {
		fmt.Printf("FAIL submit LUX/LUSD: %v\n", err)
		os.Exit(1)
	}
	var filledSz, notional float64
	for i, f := range fills {
		filledSz += f.Size
		notional += f.Size * f.Price
		fmt.Printf("FILL   LUX/LUSD #%d size=%.3f price=%.4f side=%d\n", i, f.Size, f.Price, f.TakerSide)
	}
	fmt.Printf("CROSS  LUX/LUSD buy 12@101 -> %d fills, filled=%.3f notional=%.3f vwap=%.4f\n",
		len(fills), filledSz, notional, notional/filledSz)

	// 4. cross LUX/LETH with a marketable SELL 5 @ limit 0.049.
	//    Expect 1 fill: 3@0.049 (the only bid); IOC residual (2) dropped.
	fills2, err := submit(ctx, conn, "LUX/LETH", sell, false, 0.049, 5)
	if err != nil {
		fmt.Printf("FAIL submit LUX/LETH: %v\n", err)
		os.Exit(1)
	}
	var filledSz2 float64
	for i, f := range fills2 {
		filledSz2 += f.Size
		fmt.Printf("FILL   LUX/LETH #%d size=%.3f price=%.4f side=%d\n", i, f.Size, f.Price, f.TakerSide)
	}
	fmt.Printf("CROSS  LUX/LETH sell 5@0.049 -> %d fills, filled=%.3f (IOC residual dropped)\n", len(fills2), filledSz2)

	// 5. final book state.
	for _, sym := range []string{"LUX/LUSD", "LUX/LETH"} {
		n, rem, bb, ba, _, err := depth(ctx, conn, sym)
		if err != nil {
			fmt.Printf("FAIL depth %s: %v\n", sym, err)
			os.Exit(1)
		}
		fmt.Printf("BOOK   %-9s post-cross: orders=%d remaining=%.3f bestBid=%.4f bestAsk=%.4f\n", sym, n, rem, bb, ba)
	}

	// Assertions (consensus-computed).
	okAll := true
	check := func(name string, got, want float64) {
		if math.Abs(got-want) > 1e-9 {
			fmt.Printf("ASSERT FAIL %s: got %.4f want %.4f\n", name, got, want)
			okAll = false
		} else {
			fmt.Printf("ASSERT OK   %s = %.4f\n", name, got)
		}
	}
	check("LUSD fills count", float64(len(fills)), 2)
	check("LUSD filled size", filledSz, 12)
	check("LUSD VWAP", notional/filledSz, (10*100+2*101)/12.0)
	check("LETH fills count", float64(len(fills2)), 1)
	check("LETH filled size", filledSz2, 3)
	nL, remL, _, _, _, _ := depth(ctx, conn, "LUX/LUSD")
	check("LUSD post-cross resting", remL, 23.0) // 3@101 ask + 8@99 + 12@98 bids
	_ = nL
	nE, remE, _, _, _, _ := depth(ctx, conn, "LUX/LETH")
	check("LETH post-cross resting", remE, 10.0) // 4@0.05 + 6@0.051 asks (bid fully consumed)
	_ = nE

	if okAll {
		fmt.Println("PATH-B RESULT: PASS (native ZAP e2e: seed -> cross -> consensus fills -> settled book)")
	} else {
		fmt.Println("PATH-B RESULT: FAIL")
		os.Exit(1)
	}
}
