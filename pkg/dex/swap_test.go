// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"errors"
	"testing"
)

// swap_test.go proves the shared core's swap path over REAL assets: a native maker
// rests on the book, a taker swaps through the router, the fill settles value-
// exactly, and conservation + ownership hold. No synthetic ids, no stubs.

// TestSwap_SellIntoRestingBid is the canonical product flow: a maker rests a BUY
// (bid) for base, a taker SELLs base into it through the router. The taker's base
// is debited, the maker's quote is paid to the taker, the fill is value-exact, and
// the total ledger is conserved across the whole trade.
func TestSwap_SellIntoRestingBid(t *testing.T) {
	db := newStore()
	// Market LETH (base) / LUSD (quote), identified by REAL token addresses.
	pid := seedMarket(t, db, 1, tokLETH, tokLUSD)

	maker := account(0xAA)
	taker := account(0xBB)
	// Maker deposits 1,000,000 LUSD and rests a BUY of 100 LETH @ 50 LUSD/LETH
	// (locks 100*50 = 5000 LUSD).
	seedDeposit(t, db, maker, tokLUSD, 1_000_000)
	restOrder(t, db, pid, maker, Buy, 50, 100, 1)
	// Taker deposits 100 LETH and SELLs 80 LETH (market) into the bid.
	seedDeposit(t, db, taker, tokLETH, 100)

	allAccts := []AccountID{maker, taker}
	allAssets := []AssetID{tokLETH, tokLUSD}
	before := totalLedger(t, db, allAccts, allAssets)

	req := SwapRequest{
		PoolID: pid, TakerUser: taker, Side: Sell,
		Base: tokLETH, Quote: tokLUSD,
		AmountIn: 80, OrderID: 1_000, TimestampN: 2_000,
		Class: ClassPublicDEX,
	}
	res, err := ExecuteSwap(db, orderBookRouter(), req)
	if err != nil {
		t.Fatalf("ExecuteSwap: %v", err)
	}

	// Taker sold 80 LETH, received 80*50 = 4000 LUSD.
	if res.AmountIn != 80 {
		t.Fatalf("AmountIn = %d, want 80", res.AmountIn)
	}
	if res.AmountOut != 4000 {
		t.Fatalf("AmountOut = %d, want 4000 (80 LETH @ 50)", res.AmountOut)
	}
	// The fill came from the OrderBook.
	if len(res.Fills) != 1 || res.Fills[0].Source != SourceOrderBook {
		t.Fatalf("expected one OrderBook fill, got %+v", res.Fills)
	}

	// Taker ledger: 100-80 = 20 LETH available, 4000 LUSD available.
	if av, _ := GetAvailable(db, taker, tokLETH); av != 20 {
		t.Fatalf("taker LETH available = %d, want 20", av)
	}
	if av, _ := GetAvailable(db, taker, tokLUSD); av != 4000 {
		t.Fatalf("taker LUSD available = %d, want 4000", av)
	}
	// Maker ledger: bought 80 LETH (available), spent 4000 of its 5000 locked LUSD;
	// 1000 LUSD still locked on the resting remainder (20 LETH @ 50).
	if av, _ := GetAvailable(db, maker, tokLETH); av != 80 {
		t.Fatalf("maker LETH available = %d, want 80", av)
	}
	if lk, _ := GetLocked(db, maker, tokLUSD); lk != 1000 {
		t.Fatalf("maker LUSD locked = %d, want 1000 (20 LETH @ 50 remaining)", lk)
	}

	// CONSERVATION: total (available+locked) per asset is unchanged by the trade.
	after := totalLedger(t, db, allAccts, allAssets)
	for asset, amt := range before {
		if after[asset] != amt {
			t.Fatalf("conservation violated for %x: before=%d after=%d", asset[:4], amt, after[asset])
		}
	}

	// The C-surface journal mirrors the taker's net: -80 LETH (locked into vault),
	// +4000 LUSD (proceeds out of vault). Refund of unused LETH lock is 0 here.
	net := res.Journal.NetTakerDelta()
	if net[tokLETH].Int64() != -80 {
		t.Fatalf("journal net LETH = %s, want -80", net[tokLETH])
	}
	if net[tokLUSD].Int64() != 4000 {
		t.Fatalf("journal net LUSD = %s, want 4000", net[tokLUSD])
	}
}

// Test9999Swap_FillsFromV4WhenOrderBookLiquid asserts the router fills entirely from the
// V4 OrderBook when the OrderBook has the depth — the AMM is present but untouched.
func Test9999Swap_FillsFromV4WhenOrderBookLiquid(t *testing.T) {
	db := newStore()
	pid := seedMarket(t, db, 2, tokLETH, tokLUSD)
	maker := account(0xAA)
	taker := account(0xBB)
	// Deep OrderBook bid: 1000 LETH @ 50.
	seedDeposit(t, db, maker, tokLUSD, 1_000_000)
	restOrder(t, db, pid, maker, Buy, 50, 1000, 1)
	seedDeposit(t, db, taker, tokLETH, 100)

	// AMM with shallow reserves (would price worse).
	pool := newMemAMM(0) // 0 bps fee
	pool.set(pid, 10_000, 500_000)
	router := NewRouter(NewOrderBookSource(), NewAMMSource(pool))

	req := SwapRequest{
		PoolID: pid, TakerUser: taker, Side: Sell,
		Base: tokLETH, Quote: tokLUSD, AmountIn: 50, OrderID: 9, TimestampN: 1,
		Class: ClassPublicDEX,
	}
	res, err := ExecuteSwap(db, router, req)
	if err != nil {
		t.Fatalf("ExecuteSwap: %v", err)
	}
	// OrderBook @ 50 gives 2500 LUSD for 50 LETH; the AMM would give less. All from OrderBook.
	if res.AmountOut != 2500 {
		t.Fatalf("AmountOut = %d, want 2500 (all from OrderBook @ 50)", res.AmountOut)
	}
	for _, f := range res.Fills {
		if f.Source == SourceAMM {
			t.Fatalf("AMM was used despite a better OrderBook price: %+v", res.Fills)
		}
	}
	// AMM reserves untouched.
	b, q, _, _ := pool.Reserves(db, pid)
	if b != 10_000 || q != 500_000 {
		t.Fatalf("AMM reserves moved: base=%d quote=%d", b, q)
	}
}

// Test9999Swap_FallsToV2V3WhenNoV4 asserts the router falls through to the AMM when
// the OrderBook has NO crossable depth.
func Test9999Swap_FallsToV2V3WhenNoV4(t *testing.T) {
	db := newStore()
	pid := seedMarket(t, db, 3, tokLETH, tokLUSD)
	taker := account(0xBB)
	seedDeposit(t, db, taker, tokLETH, 1000)

	// No OrderBook liquidity at all; AMM has reserves.
	pool := newMemAMM(0)
	pool.set(pid, 100_000, 5_000_000) // price ~50 LUSD/LETH
	router := NewRouter(NewOrderBookSource(), NewAMMSource(pool))

	req := SwapRequest{
		PoolID: pid, TakerUser: taker, Side: Sell,
		Base: tokLETH, Quote: tokLUSD, AmountIn: 100, OrderID: 9, TimestampN: 1,
		Class: ClassPublicDEX,
	}
	res, err := ExecuteSwap(db, router, req)
	if err != nil {
		t.Fatalf("ExecuteSwap: %v", err)
	}
	// out = 100 * 5_000_000 / (100_000 + 100) = 4995 (xy=k, integer).
	wantOut := ConstantProductOut(100_000, 5_000_000, 100)
	if res.AmountOut != wantOut {
		t.Fatalf("AmountOut = %d, want %d (AMM curve)", res.AmountOut, wantOut)
	}
	if len(res.Fills) != 1 || res.Fills[0].Source != SourceAMM {
		t.Fatalf("expected one AMM fill, got %+v", res.Fills)
	}
	// AMM reserves moved: +100 base, -wantOut quote.
	b, q, _, _ := pool.Reserves(db, pid)
	if b != 100_100 || q != 5_000_000-wantOut {
		t.Fatalf("AMM reserves wrong: base=%d quote=%d", b, q)
	}
}

// Test9999Swap_BestExecAcrossV4AndAMM asserts the router SPLITS across the OrderBook and
// the AMM when neither alone is best for the whole amount: it takes the OrderBook depth
// at the better price, then the AMM for the remainder.
func Test9999Swap_BestExecAcrossV4AndAMM(t *testing.T) {
	db := newStore()
	pid := seedMarket(t, db, 4, tokLETH, tokLUSD)
	maker := account(0xAA)
	taker := account(0xBB)
	// Thin but BETTER OrderBook bid: 10 LETH @ 60 (better than AMM's ~50).
	seedDeposit(t, db, maker, tokLUSD, 1_000_000)
	restOrder(t, db, pid, maker, Buy, 60, 10, 1)
	seedDeposit(t, db, taker, tokLETH, 1000)

	// AMM priced ~50.
	pool := newMemAMM(0)
	pool.set(pid, 100_000, 5_000_000)
	router := NewRouter(NewOrderBookSource(), NewAMMSource(pool))

	// Taker sells 30 LETH. OrderBook absorbs 10 @ 60 = 600; AMM absorbs the other 20.
	req := SwapRequest{
		PoolID: pid, TakerUser: taker, Side: Sell,
		Base: tokLETH, Quote: tokLUSD, AmountIn: 30, OrderID: 9, TimestampN: 1,
		Class: ClassPublicDEX,
	}
	res, err := ExecuteSwap(db, router, req)
	if err != nil {
		t.Fatalf("ExecuteSwap: %v", err)
	}
	// Expect a OrderBook fill of 10 LETH -> 600 LUSD and an AMM fill of 20 LETH.
	var sawOrderBook, sawAMM bool
	var orderBookIn, ammIn uint64
	for _, f := range res.Fills {
		switch f.Source {
		case SourceOrderBook:
			sawOrderBook = true
			for _, tr := range f.Trades {
				orderBookIn += tr.BaseUnits.Uint64()
			}
		case SourceAMM:
			sawAMM = true
			ammIn += f.AMMBaseDelta
		}
	}
	if !sawOrderBook || !sawAMM {
		t.Fatalf("expected a split across OrderBook+AMM, got %+v", res.Fills)
	}
	if orderBookIn != 10 {
		t.Fatalf("OrderBook leg base = %d, want 10", orderBookIn)
	}
	if ammIn != 20 {
		t.Fatalf("AMM leg base = %d, want 20", ammIn)
	}
	if res.AmountIn != 30 {
		t.Fatalf("total AmountIn = %d, want 30", res.AmountIn)
	}
	// Output = 600 (OrderBook) + AMM curve(20).
	ammOut := ConstantProductOut(100_000, 5_000_000, 20)
	if res.AmountOut != 600+ammOut {
		t.Fatalf("AmountOut = %d, want %d (600 OrderBook + %d AMM)", res.AmountOut, 600+ammOut, ammOut)
	}
}

// Test9999Swap_NoLiquidityReverts asserts a swap with NO source able to fill
// reverts (the taker's lock is rolled back by the caller; here ExecuteSwap returns
// the error and the journal shows the input was locked but nothing was committed —
// the precompile's EVM snapshot rolls it back).
func Test9999Swap_NoLiquidityReverts(t *testing.T) {
	db := newStore()
	pid := seedMarket(t, db, 5, tokLETH, tokLUSD)
	taker := account(0xBB)
	seedDeposit(t, db, taker, tokLETH, 100)
	// No OrderBook, no AMM.
	req := SwapRequest{
		PoolID: pid, TakerUser: taker, Side: Sell,
		Base: tokLETH, Quote: tokLUSD, AmountIn: 50, OrderID: 9, TimestampN: 1,
		Class: ClassPublicDEX,
	}
	_, err := ExecuteSwap(db, orderBookRouter(), req)
	if !errors.Is(err, ErrNoLiquidity) {
		t.Fatalf("expected ErrNoLiquidity, got %v", err)
	}
}

// Test9999Swap_BuyRequiresLimit asserts an exact-input BUY with no price ceiling is
// refused (unbounded-slippage market buy).
func Test9999Swap_BuyRequiresLimit(t *testing.T) {
	db := newStore()
	pid := seedMarket(t, db, 6, tokLETH, tokLUSD)
	taker := account(0xBB)
	seedDeposit(t, db, taker, tokLUSD, 10_000)
	req := SwapRequest{
		PoolID: pid, TakerUser: taker, Side: Buy,
		Base: tokLETH, Quote: tokLUSD, AmountIn: 5000, OrderID: 9, TimestampN: 1,
		Class: ClassPublicDEX, // LimitPrice 0 -> refused
	}
	_, err := ExecuteSwap(db, orderBookRouter(), req)
	if !errors.Is(err, ErrBuyRequiresLimit) {
		t.Fatalf("expected ErrBuyRequiresLimit, got %v", err)
	}
}

// TestSwap_BuyWithLimit proves the exact-input BUY path: a maker rests an ASK, a
// taker BUYs base with a quote budget bounded by a ceiling price; the budget buys
// floor(budget/ceiling) base and unspent quote is refunded.
func TestSwap_BuyWithLimit(t *testing.T) {
	db := newStore()
	pid := seedMarket(t, db, 7, tokLETH, tokLUSD)
	maker := account(0xAA)
	taker := account(0xBB)
	// Maker rests a SELL (ask) of 100 LETH @ 50 (locks 100 LETH base).
	seedDeposit(t, db, maker, tokLETH, 1000)
	restOrder(t, db, pid, maker, Sell, 50, 100, 1)
	// Taker deposits 5000 LUSD, buys base with a ceiling of 50 -> budget buys 100 base.
	seedDeposit(t, db, taker, tokLUSD, 5000)

	allAccts := []AccountID{maker, taker}
	allAssets := []AssetID{tokLETH, tokLUSD}
	before := totalLedger(t, db, allAccts, allAssets)

	req := SwapRequest{
		PoolID: pid, TakerUser: taker, Side: Buy,
		Base: tokLETH, Quote: tokLUSD, AmountIn: 5000,
		LimitPrice: price(50), LimitIsUpper: true,
		OrderID: 9, TimestampN: 1, Class: ClassPublicDEX,
	}
	res, err := ExecuteSwap(db, orderBookRouter(), req)
	if err != nil {
		t.Fatalf("ExecuteSwap: %v", err)
	}
	// 5000 LUSD @ ceiling 50 buys 100 LETH for 5000 LUSD.
	if res.AmountOut != 100 {
		t.Fatalf("AmountOut = %d, want 100 LETH", res.AmountOut)
	}
	if res.AmountIn != 5000 {
		t.Fatalf("AmountIn = %d, want 5000 LUSD", res.AmountIn)
	}
	if av, _ := GetAvailable(db, taker, tokLETH); av != 100 {
		t.Fatalf("taker LETH = %d, want 100", av)
	}
	// Conservation.
	after := totalLedger(t, db, allAccts, allAssets)
	for asset, amt := range before {
		if after[asset] != amt {
			t.Fatalf("conservation violated %x: before=%d after=%d", asset[:4], amt, after[asset])
		}
	}
}
