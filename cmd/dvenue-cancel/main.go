// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command dvenue-cancel is the LIVE cancel-hard-gate e2e against a running
// D-Chain venue. It proves the cancel custody semantic: a PLACE locks available
// into reserved; a CANCEL unlocks EXACTLY the unfilled reserve back to available
// (never more, never the filled part). It also proves a PARTIAL-fill-then-cancel
// releases ONLY the still-resting remainder.
package main

import (
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

const (
	assetLUX  uint64 = 0x4c5558_00000001
	assetLUSD uint64 = 0x4c555344_000001
)

func pool() [32]byte {
	var p [32]byte
	copy(p[:], "LUX/LUSD-CANCEL")
	p[31] = 0xCA
	return p
}

func main() {
	addr := flag.String("addr", "127.0.0.1:9099", "venue ZAP addr")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	c, err := rpc.ZAPDial(ctx, *addr)
	if err != nil {
		fmt.Printf("FAIL dial: %v\n", err)
		os.Exit(1)
	}
	defer c.Close()

	ok := true
	check := func(name string, got, want uint64) {
		if got != want {
			fmt.Printf("ASSERT FAIL %s: got %d want %d\n", name, got, want)
			ok = false
		} else {
			fmt.Printf("ASSERT OK   %s = %d\n", name, got)
		}
	}
	bal := func(user string, asset uint64) (uint64, uint64) {
		req := make([]byte, zapwire.UserSize+zapwire.AssetIDSize)
		copy(req, user)
		binary.BigEndian.PutUint64(req[zapwire.UserSize:], asset)
		r, e := c.Call(ctx, "clob_balance", req)
		if e != nil || len(r) < 16 {
			return 0, 0
		}
		return binary.BigEndian.Uint64(r[0:8]), binary.BigEndian.Uint64(r[8:16])
	}

	p := pool()
	const maker, taker = "cancMaker", "cancTaker"

	// open + deposit.
	if r, e := c.Call(ctx, zapwire.MethodOpenMarket, zapwire.EncodeOpenMarket(p, assetLUX, assetLUSD)); e != nil {
		fmt.Printf("FAIL open: %v\n", e)
		os.Exit(1)
	} else if _, st, _ := zapwire.DecodeAck(r); st == zapwire.StatusRejected {
		fmt.Println("FAIL open rejected")
		os.Exit(1)
	}
	c.Call(ctx, zapwire.MethodDeposit, zapwire.EncodeDeposit(maker, assetLUX, 100))
	c.Call(ctx, zapwire.MethodDeposit, zapwire.EncodeDeposit(taker, assetLUSD, 1000))
	fmt.Println("setup: open LUX/LUSD-CANCEL, deposit maker 100 LUX + taker 1000 LUSD")

	// ---- CASE 1: plain place -> cancel returns the FULL lock ----
	fmt.Println("\n--- CASE 1: place SELL 30 LUX @ 5, then CANCEL (full unlock) ---")
	r1, _ := c.Call(ctx, zapwire.MethodPlace, zapwire.EncodePlace(p, zapwire.SideSell, 5.0, 30.0, maker))
	oid1, st1, _ := zapwire.DecodeAck(r1)
	if st1 == zapwire.StatusRejected {
		fmt.Println("FAIL place rejected")
		os.Exit(1)
	}
	a, l := bal(maker, assetLUX)
	fmt.Printf("after place: maker LUX avail=%d locked=%d (orderID=%d)\n", a, l, oid1)
	check("avail after place", a, 70)
	check("locked after place", l, 30)

	rc, _ := c.Call(ctx, zapwire.MethodCancel, zapwire.EncodeCancel(p, oid1))
	_, stc, _ := zapwire.DecodeAck(rc)
	if stc == zapwire.StatusRejected {
		fmt.Println("FAIL cancel rejected")
		os.Exit(1)
	}
	a, l = bal(maker, assetLUX)
	fmt.Printf("after cancel: maker LUX avail=%d locked=%d\n", a, l)
	check("avail after cancel (full unlock)", a, 100)
	check("locked after cancel", l, 0)

	// ---- CASE 2: partial fill, then cancel releases ONLY the remainder ----
	fmt.Println("\n--- CASE 2: place SELL 40 LUX @ 5, taker BUYS 10, cancel rest (release only 30) ---")
	r2, _ := c.Call(ctx, zapwire.MethodPlace, zapwire.EncodePlace(p, zapwire.SideSell, 5.0, 40.0, maker))
	oid2, st2, _ := zapwire.DecodeAck(r2)
	if st2 == zapwire.StatusRejected {
		fmt.Println("FAIL place2 rejected")
		os.Exit(1)
	}
	a, l = bal(maker, assetLUX)
	fmt.Printf("after place2: maker LUX avail=%d locked=%d\n", a, l)
	check("avail after place2", a, 60)
	check("locked after place2", l, 40)

	// Taker buys 10 LUX @ 5 -> 50 LUSD, crossing the maker's resting sell.
	fills, _ := zapwire.DecodeFills(mustCall(ctx, c, zapwire.MethodSubmit, zapwire.EncodeSubmit(p, zapwire.SideBuy, false, 5.0, 10.0, taker)))
	fmt.Printf("taker bought %d fill(s)\n", len(fills))
	a, l = bal(maker, assetLUX)
	mq, _ := bal(maker, assetLUSD)
	fmt.Printf("after partial fill: maker LUX avail=%d locked=%d, maker LUSD avail=%d\n", a, l, mq)
	check("maker LUX locked after partial fill", l, 30) // 40 - 10 filled
	check("maker LUSD received", mq, 50)

	// Cancel the remainder: should unlock ONLY the still-resting 30 LUX.
	rc2, _ := c.Call(ctx, zapwire.MethodCancel, zapwire.EncodeCancel(p, oid2))
	_, stc2, _ := zapwire.DecodeAck(rc2)
	if stc2 == zapwire.StatusRejected {
		fmt.Println("FAIL cancel2 rejected")
		os.Exit(1)
	}
	a, l = bal(maker, assetLUX)
	fmt.Printf("after cancel2: maker LUX avail=%d locked=%d\n", a, l)
	check("maker LUX locked after cancel remainder", l, 0)
	check("maker LUX avail after cancel remainder", a, 90) // 60 + 30 unlocked (10 went to taker)

	// Conservation: maker LUX (90 avail) + taker LUX (10) == 100 deposited.
	ta, _ := bal(taker, assetLUX)
	check("LUX conserved (maker avail + taker)", a+ta, 100)

	if ok {
		fmt.Println("\nCANCEL-GATE RESULT: PASS (full unlock + partial-fill-then-cancel releases only the remainder, conserving)")
	} else {
		fmt.Println("\nCANCEL-GATE RESULT: FAIL")
		os.Exit(1)
	}
}

func mustCall(ctx context.Context, c *rpc.ZAPConn, method string, payload []byte) []byte {
	r, e := c.Call(ctx, method, payload)
	if e != nil {
		fmt.Printf("FAIL %s: %v\n", method, e)
		os.Exit(1)
	}
	return r
}
