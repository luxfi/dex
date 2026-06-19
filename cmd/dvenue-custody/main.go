// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Command dvenue-custody is the LIVE custody-model e2e driver against a running
// D-Chain venue. It proves "the money lives in the order book": funds DEPOSIT
// into the D-Chain (clob_deposit — the funds-in leg of the atomic rail), live as
// per-account available balance the book draws from, an order LOCKS available
// into reserved, a marketable cross SETTLES ENTIRELY inside D-Chain consensus
// (balances move maker<->taker in the ledger), and the proceeds WITHDRAW back out
// (clob_withdraw — funds-out). Every write crosses the venue's full match ->
// Verify -> Accept consensus path; balances are read from the committed ledger.
//
// The acceptance criterion is WEI-EXACT CONSERVATION for a NON-NATIVE asset:
// total deposited == total withdrawn, and the MAKER receives exactly what the
// TAKER paid (the leg the prior taker-only settle model dropped).
package main

import (
	"context"
	"crypto/rand"
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// Asset handles for the live market (8-byte ledger handles). LUX = base, LUSD =
// quote. These are the D-Chain asset identities (the EVM maps a 20-byte currency
// to such a handle; here we use them directly).
const (
	assetLUX  uint64 = 0x4c5558_00000001
	assetLUSD uint64 = 0x4c555344_000001
)

func custodyPool() [32]byte {
	var p [32]byte
	copy(p[:], "LUX/LUSD-CUSTODY")
	p[31] = 0xC0
	return p
}

func openMarket(ctx context.Context, c *rpc.ZAPConn, pool [32]byte, base, quote uint64) error {
	resp, err := c.Call(ctx, zapwire.MethodOpenMarket, zapwire.EncodeOpenMarket(pool, base, quote))
	if err != nil {
		return err
	}
	_, status, derr := zapwire.DecodeAck(resp)
	if derr != nil {
		return derr
	}
	if status == zapwire.StatusRejected {
		return fmt.Errorf("open_market rejected")
	}
	return nil
}

func deposit(ctx context.Context, c *rpc.ZAPConn, user string, asset, amount uint64) (uint64, error) {
	resp, err := c.Call(ctx, zapwire.MethodDeposit, zapwire.EncodeDeposit(user, asset, amount, freshRef()))
	if err != nil {
		return 0, err
	}
	_, credited, derr := zapwire.DecodeBalanceResp(resp)
	return credited, derr
}

func withdraw(ctx context.Context, c *rpc.ZAPConn, user string, asset, want uint64) (uint64, error) {
	resp, err := c.Call(ctx, zapwire.MethodWithdraw, zapwire.EncodeWithdraw(user, asset, want, freshRef()))
	if err != nil {
		return 0, err
	}
	_, realized, derr := zapwire.DecodeBalanceResp(resp)
	return realized, derr
}

// freshRef returns a unique 32-byte idempotency ref for a genuinely-distinct
// custody op. This live CLI issues each deposit/withdraw as its own operation (not
// a retry), so each carries a fresh ref — distinct on the d-chain seen: index,
// never deduped against a prior op. (Production callers supply the originating EVM
// txHash / atomic import id; a live operator CLI has no such identity, so a CSPRNG
// nonce is the correct unique reference.)
func freshRef() [32]byte {
	var r [32]byte
	if _, err := rand.Read(r[:]); err != nil {
		panic("dvenue-custody: CSPRNG failure deriving custody ref: " + err.Error())
	}
	return r
}

func balance(ctx context.Context, c *rpc.ZAPConn, user string, asset uint64) (available, locked uint64, err error) {
	req := make([]byte, zapwire.UserSize+zapwire.AssetIDSize)
	copy(req[0:zapwire.UserSize], user)
	binary.BigEndian.PutUint64(req[zapwire.UserSize:], asset)
	resp, e := c.Call(ctx, "clob_balance", req)
	if e != nil {
		return 0, 0, e
	}
	if len(resp) < 16 {
		return 0, 0, fmt.Errorf("short balance resp %d", len(resp))
	}
	return binary.BigEndian.Uint64(resp[0:8]), binary.BigEndian.Uint64(resp[8:16]), nil
}

func place(ctx context.Context, c *rpc.ZAPConn, pool [32]byte, side uint8, price, size float64, user string) (uint64, uint8, error) {
	resp, err := c.Call(ctx, zapwire.MethodPlace, zapwire.EncodePlace(pool, side, price, size, user))
	if err != nil {
		return 0, 0, err
	}
	oid, status, derr := zapwire.DecodeAck(resp)
	return oid, status, derr
}

func submit(ctx context.Context, c *rpc.ZAPConn, pool [32]byte, side uint8, isMarket bool, limit, size float64, user string) ([]zapwire.Fill, error) {
	resp, err := c.Call(ctx, zapwire.MethodSubmit, zapwire.EncodeSubmit(pool, side, isMarket, limit, size, user))
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
	fmt.Printf("CUSTODY e2e dialed venue %s\n", *addr)

	const maker, taker = "custMakerA", "custTakerX"
	pool := custodyPool()
	okAll := true
	check := func(name string, got, want uint64) {
		if got != want {
			fmt.Printf("ASSERT FAIL %s: got %d want %d\n", name, got, want)
			okAll = false
		} else {
			fmt.Printf("ASSERT OK   %s = %d\n", name, got)
		}
	}

	// 1. open market (bind base/quote assets) so orders can be value-checked.
	if err := openMarket(ctx, conn, pool, assetLUX, assetLUSD); err != nil {
		fmt.Printf("FAIL open_market: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("open_market LUX/LUSD-CUSTODY (base=LUX, quote=LUSD) OK")

	// 2. DEPOSIT funds INTO the D-Chain: maker 100 LUX, taker 1000 LUSD.
	cMaker, err := deposit(ctx, conn, maker, assetLUX, 100)
	if err != nil {
		fmt.Printf("FAIL deposit maker: %v\n", err)
		os.Exit(1)
	}
	cTaker, err := deposit(ctx, conn, taker, assetLUSD, 1000)
	if err != nil {
		fmt.Printf("FAIL deposit taker: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("DEPOSIT maker 100 LUX (credited %d), taker 1000 LUSD (credited %d)\n", cMaker, cTaker)
	check("deposit maker LUX credited", cMaker, 100)
	check("deposit taker LUSD credited", cTaker, 1000)

	// Confirm the deposits credited available balance (the book can draw from it).
	mLUXa, mLUXl, _ := balance(ctx, conn, maker, assetLUX)
	tLUSDa, tLUSDl, _ := balance(ctx, conn, taker, assetLUSD)
	fmt.Printf("BAL after deposit: maker LUX avail=%d locked=%d | taker LUSD avail=%d locked=%d\n", mLUXa, mLUXl, tLUSDa, tLUSDl)
	check("maker LUX available post-deposit", mLUXa, 100)
	check("taker LUSD available post-deposit", tLUSDa, 1000)

	// 3. maker PLACES a resting SELL of 10 LUX @ 5 (locks 10 LUX from available).
	oid, status, err := place(ctx, conn, pool, zapwire.SideSell, 5.0, 10.0, maker)
	if err != nil || status == zapwire.StatusRejected {
		fmt.Printf("FAIL place maker sell: err=%v status=%d\n", err, status)
		os.Exit(1)
	}
	fmt.Printf("PLACE maker SELL 10 LUX @ 5 -> orderID=%d\n", oid)
	mLUXa, mLUXl, _ = balance(ctx, conn, maker, assetLUX)
	fmt.Printf("BAL after place: maker LUX avail=%d locked=%d\n", mLUXa, mLUXl)
	check("maker LUX available post-place", mLUXa, 90)
	check("maker LUX locked post-place", mLUXl, 10)

	// 4. taker SUBMITS a marketable BUY of 10 LUX, limit 5 (locks 50 LUSD), CROSSING.
	fills, err := submit(ctx, conn, pool, zapwire.SideBuy, false, 5.0, 10.0, taker)
	if err != nil {
		fmt.Printf("FAIL submit taker buy: %v\n", err)
		os.Exit(1)
	}
	var filledBase, notional float64
	for i, f := range fills {
		filledBase += f.Size
		notional += f.Size * f.Price
		fmt.Printf("FILL #%d size=%.3f price=%.4f side=%d\n", i, f.Size, f.Price, f.TakerSide)
	}
	fmt.Printf("CROSS taker BUY 10 LUX @ 5 -> %d fills, filled=%.3f notional=%.3f (settled IN the D-Chain ledger)\n",
		len(fills), filledBase, notional)

	// 5. THE MAKER LEG (what the old taker-only model dropped): the maker received
	//    the taker's 50 LUSD into available, and its locked LUX moved to the taker.
	mLUXa, mLUXl, _ = balance(ctx, conn, maker, assetLUX)
	mLUSDa, _, _ := balance(ctx, conn, maker, assetLUSD)
	tLUXa, _, _ := balance(ctx, conn, taker, assetLUX)
	tLUSDa, tLUSDl, _ = balance(ctx, conn, taker, assetLUSD)
	fmt.Printf("BAL after cross: maker LUX avail=%d locked=%d, maker LUSD avail=%d | taker LUX avail=%d, taker LUSD avail=%d locked=%d\n",
		mLUXa, mLUXl, mLUSDa, tLUXa, tLUSDa, tLUSDl)
	check("maker LUX available post-cross", mLUXa, 90)   // 10 locked spent to taker
	check("maker LUX locked post-cross", mLUXl, 0)       // fully consumed
	check("maker LUSD received (the taker's payment)", mLUSDa, 50)
	check("taker LUX received", tLUXa, 10)
	check("taker LUSD available post-cross", tLUSDa, 950) // 1000 - 50 spent
	check("taker LUSD locked post-cross", tLUSDl, 0)      // lock fully spent (50==lock)

	// 6. CONSERVATION across the trade: every unit still present somewhere.
	check("LUX conserved (maker+taker available)", mLUXa+tLUXa, 100)
	check("LUSD conserved (maker+taker available)", mLUSDa+tLUSDa, 1000)

	// 7. WITHDRAW everything back out; realized == available, conserving to the unit.
	var totalOut uint64
	w := func(user string, asset, want uint64) uint64 {
		r, e := withdraw(ctx, conn, user, asset, want)
		if e != nil {
			fmt.Printf("FAIL withdraw %s asset=%#x: %v\n", user, asset, e)
			os.Exit(1)
		}
		return r
	}
	totalOut += w(maker, assetLUX, 90)
	totalOut += w(maker, assetLUSD, 50)
	totalOut += w(taker, assetLUX, 10)
	totalOut += w(taker, assetLUSD, 950)
	fmt.Printf("WITHDRAW total realized = %d\n", totalOut)
	check("total withdrawn == total deposited", totalOut, 100+1000)

	// 8. ledger drained to zero for both parties.
	mLUXa, _, _ = balance(ctx, conn, maker, assetLUX)
	mLUSDa, _, _ = balance(ctx, conn, maker, assetLUSD)
	tLUXa, _, _ = balance(ctx, conn, taker, assetLUX)
	tLUSDa, _, _ = balance(ctx, conn, taker, assetLUSD)
	check("maker LUX drained", mLUXa, 0)
	check("maker LUSD drained", mLUSDa, 0)
	check("taker LUX drained", tLUXa, 0)
	check("taker LUSD drained", tLUSDa, 0)

	if okAll {
		fmt.Println("CUSTODY RESULT: PASS (deposit -> book -> cross -> settle-in-ledger -> withdraw, WEI-CONSERVING for a non-native asset)")
	} else {
		fmt.Println("CUSTODY RESULT: FAIL")
		os.Exit(1)
	}
}
