// venue.go is the keeper's direct ZAP client to the live D-Chain venue (cmd/dvenue),
// used to SEED maker liquidity so a Phase-A taker order routed through the seam has a
// resting book to fill against. It reuses the exact clob_* wire + custody-seal polling
// patterns from cmd/fourpath-live (the proven live-venue client): ensure/open a market,
// deposit maker inventory, rest a maker ask, and observe balances/depth until sealed.
//
// This is ONLY the seeding side. The taker order itself does NOT go to the venue directly
// from the keeper — it rides the C->D atomic seam: a Phase-A swap to 0x9999 stages a C->D
// object, the keeper funds the D order via ImportTx + a settling RelayOrderTx submitted to
// the dexvm (keeper.go), and the dexvm proposer relays that order to THIS venue once.
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"time"

	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/rpc"
)

// read-only observation methods served by cmd/dvenue (same as fourpath-live).
const (
	clobDepthMethod   = "clob_depth"
	clobBalanceMethod = "clob_balance"
)

// txDepositTag is dchain.TxDeposit's wire tag, mixed into the content ref so deposits are
// exactly-once on retry. Kept in sync with pkg/dchain tx kinds.
const txDepositTag = 1

// venueClient is a thin live ZAP client over rpc.ZAPDial.
type venueClient struct {
	conn *rpc.ZAPConn
	ctx  context.Context
}

// dialVenue opens a ZAP connection to the venue at addr (host:port). The returned close
// func releases the connection and the dial context.
func dialVenue(ctx context.Context, addr string) (*venueClient, func(), error) {
	dctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	conn, err := rpc.ZAPDial(dctx, addr)
	if err != nil {
		cancel()
		return nil, nil, fmt.Errorf("ZAPDial(%s): %w", addr, err)
	}
	return &venueClient{conn: conn, ctx: dctx}, func() { _ = conn.Close(); cancel() }, nil
}

// poolIDForSymbol mirrors pkg/dchain.poolIDForSymbol: symbol bytes, high byte tagged 0xD0
// so an all-symbol-prefix pool can't collide with a zero id.
func poolIDForSymbol(sym string) [32]byte {
	var p [32]byte
	copy(p[:], sym)
	p[31] = 0xD0
	return p
}

// assetID mirrors pkg/dchain.a32: a label packed big-endian into the low 8 bytes.
func assetID(label uint64) [32]byte {
	var a [32]byte
	binary.BigEndian.PutUint64(a[24:32], label)
	return a
}

// contentRef mirrors pkg/dchain.contentRef so deposits are exactly-once on retry and
// genuinely distinct otherwise.
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

func (c *venueClient) ensureMarket(pool [32]byte) error {
	if _, err := c.conn.Call(c.ctx, zapwire.MethodEnsureMarket, zapwire.EncodeEnsureMarket(pool)); err != nil {
		return fmt.Errorf("clob_ensure_market: %w", err)
	}
	return nil
}

func (c *venueClient) openMarket(pool, base, quote [32]byte) error {
	resp, err := c.conn.Call(c.ctx, zapwire.MethodOpenMarket, zapwire.EncodeOpenMarket(pool, base, quote))
	if err != nil {
		return fmt.Errorf("clob_open_market: %w", err)
	}
	if len(resp) < 9 || resp[8] != zapwire.StatusPlaced {
		return fmt.Errorf("clob_open_market not placed: resp=%x", resp)
	}
	return nil
}

// deposit credits user's available balance for asset and waits for the sealer to credit it.
// The live venue ACKs StatusPlaced (accepted) but credits when the block seals, so it
// confirms via clob_balance rather than a synchronous echo (mirrors fourpath-live.deposit).
func (c *venueClient) deposit(user string, asset [32]byte, amount uint64) error {
	ref := contentRef(byte(txDepositTag), user, asset, amount)
	resp, err := c.conn.Call(c.ctx, zapwire.MethodDeposit, zapwire.EncodeDeposit(user, asset, amount, ref))
	if err != nil {
		return fmt.Errorf("clob_deposit %s %d: %w", user, amount, err)
	}
	if status, _, derr := zapwire.DecodeBalanceResp(resp); derr == nil && status == zapwire.StatusRejected {
		return fmt.Errorf("clob_deposit %s %d: REJECTED", user, amount)
	}
	return c.waitAvailable(user, asset, amount)
}

// place rests a limit order (e.g. a maker ask) and confirms it was accepted.
func (c *venueClient) place(pool [32]byte, side uint8, price, size float64, user string) error {
	resp, err := c.conn.Call(c.ctx, zapwire.MethodPlace, zapwire.EncodePlace(pool, side, price, size, user))
	if err != nil {
		return fmt.Errorf("clob_place %s: %w", user, err)
	}
	if _, status, derr := zapwire.DecodeAck(resp); derr != nil || status != zapwire.StatusPlaced {
		return fmt.Errorf("clob_place %s: status=%d err=%v", user, status, derr)
	}
	return nil
}

// balance reads clob_balance: (available, locked).
func (c *venueClient) balance(user string, asset [32]byte) (available, locked uint64, err error) {
	req := make([]byte, zapwire.UserSize+zapwire.AssetIDSize)
	copy(req[0:zapwire.UserSize], user)
	copy(req[zapwire.UserSize:], asset[:])
	resp, err := c.conn.Call(c.ctx, clobBalanceMethod, req)
	if err != nil {
		return 0, 0, fmt.Errorf("clob_balance %s: %w", user, err)
	}
	if len(resp) < 16 {
		return 0, 0, fmt.Errorf("clob_balance short resp: %x", resp)
	}
	return binary.BigEndian.Uint64(resp[0:8]), binary.BigEndian.Uint64(resp[8:16]), nil
}

// depth reads clob_depth: (orders, remaining, ok).
func (c *venueClient) depth(pool [32]byte) (orders int, remaining float64, ok bool, err error) {
	resp, err := c.conn.Call(c.ctx, clobDepthMethod, pool[:])
	if err != nil {
		return 0, 0, false, fmt.Errorf("clob_depth: %w", err)
	}
	if len(resp) < 29 {
		return 0, 0, false, fmt.Errorf("clob_depth short resp: %x", resp)
	}
	return int(binary.BigEndian.Uint32(resp[0:4])), math.Float64frombits(binary.BigEndian.Uint64(resp[4:12])), resp[28] == 1, nil
}

// waitAvailable polls clob_balance until available >= want (the credit has sealed) or
// times out.
func (c *venueClient) waitAvailable(user string, asset [32]byte, want uint64) error {
	deadline := time.Now().Add(20 * time.Second)
	for {
		a, _, err := c.balance(user, asset)
		if err != nil {
			return err
		}
		if a >= want {
			return nil
		}
		if time.Now().After(deadline) {
			a, lck, _ := c.balance(user, asset)
			return fmt.Errorf("clob_deposit/credit %s: available=%d locked=%d, want >=%d (sealer not crediting)", user, a, lck, want)
		}
		time.Sleep(150 * time.Millisecond)
	}
}

// waitDepth polls until the book's order count and remaining match (the rested ask sealed).
func (c *venueClient) waitDepth(pool [32]byte, wantOrders int, wantRem float64) error {
	deadline := time.Now().Add(20 * time.Second)
	for {
		n, rem, ok, err := c.depth(pool)
		if err != nil {
			return err
		}
		if ok && n == wantOrders && rem == wantRem {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("book orders=%d remaining=%v ok=%v, want %d / %v", n, rem, ok, wantOrders, wantRem)
		}
		time.Sleep(150 * time.Millisecond)
	}
}

// seedSpec describes one maker book to seed: a market (base/quote) with a single resting
// maker ask of askSize at askPrice. The keeper seeds this so a Phase-A taker BUY routed
// through the seam fills exactly one ask.
type seedSpec struct {
	Symbol   string
	Base     [32]byte
	Quote    [32]byte
	Maker    string
	AskSize  float64
	AskPrice float64
}

// seedMakerLiquidity ensures+opens the market, deposits the maker's base inventory, rests
// one maker ask, and waits for both the deposit credit and the resting ask to seal. It is
// idempotent on retry via content-addressed deposit refs (a re-run with the same spec does
// not double-credit). Returns the pool id of the seeded market.
func (c *venueClient) seedMakerLiquidity(s seedSpec) ([32]byte, error) {
	pool := poolIDForSymbol(s.Symbol)
	if err := c.ensureMarket(pool); err != nil {
		return pool, err
	}
	if err := c.openMarket(pool, s.Base, s.Quote); err != nil {
		return pool, err
	}
	// Maker funds AskSize of base inventory so the resting ask is fully collateralised.
	makerBase := uint64(math.Ceil(s.AskSize))
	if err := c.deposit(s.Maker, s.Base, makerBase); err != nil {
		return pool, err
	}
	if err := c.place(pool, zapwire.SideSell, s.AskPrice, s.AskSize, s.Maker); err != nil {
		return pool, err
	}
	if err := c.waitDepth(pool, 1, s.AskSize); err != nil {
		return pool, err
	}
	return pool, nil
}
