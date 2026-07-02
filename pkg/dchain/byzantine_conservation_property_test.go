// Copyright (C) 2019-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// byzantine_conservation_property_test.go is threat #4: value conservation under
// ADVERSARIAL matcher load, PLUS fork-freedom under the same load. Where
// conservation_property_test.go stresses the custody rail (deposit/withdraw), this
// property test hammers the MATCHER's byzantine surface — self-crosses, deep
// multi-price books, oversized (over-fill) takers, and cancel-then-cross phantom
// liquidity — and asserts, after EVERY op:
//
//	(1) CONSERVATION: for each asset, Σ (available + locked) over the whole ledger
//	    equals the amount deposited. The matcher mints and burns NOTHING regardless
//	    of how adversarial the order stream is — INCLUDING when it declines to build
//	    a block (a fail-safe refusal moves no value).
//	(2) NO FORK: an INDEPENDENT validator VM replaying the proposer's exact accepted
//	    block bytes derives the identical execution root — a byzantine order stream
//	    can never split the honest validators.
//
// FAIL-SAFE REFUSALS (FIXED — this test is now the regression guard). A
// self-cross-heavy stream previously drove the matcher into a state where a single
// later submit was REFUSED at build with "resting order missing full settlement
// identity (orderuser)" or "insufficient locked balance" — the matcher declining to
// settle rather than minting (fail-safe, no mint/burn/fork). Root cause was in
// pkg/dex/orderbook.go tryMatchImmediateLocked: the self-trade skip removed the
// self-maker from the IN-MEMORY book (removeOrder + delete Orders/ordersMap) without
// a matching consensus state write, so the in-memory book (which feeds the execRoot
// via BookToRows) and the persisted order:/orderuser:/reserve rows diverged. FIX:
// the matcher now RECORDS the self-cancelled maker (ob.selfCanceled), and applySubmit
// drains it so settleOrderEffects/execute persist the removal through the SAME
// cancel path a TxCancel uses (delete order:/orderuser: rows, release the reserve
// locked->available). This test asserts ZERO fail-safe refusals across the stream
// (see TestByzantineMatcher_SelfCrossCancelsRestingMakerConsistently for the direct
// pin); it still COUNTS refusals as defense-in-depth so a regression would surface.
//
// Deterministic: a fixed seed table, block-derived ids, no wall-clock.

package dchain

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/luxfi/database/memdb"
	"github.com/luxfi/dex/pkg/zapwire"
)

var byzantineMatcherSeeds = []int64{1, 5, 42, 101, 777, 4242, 31337, 65537}

const byzMatcherOpsPerSeed = 160

// tryProposeReplay proposes a one-tx block on the proposer. On success it replays
// the identical accepted bytes to the validator, asserts NO FORK, and returns
// (height, built=true). On a FAIL-SAFE matcher refusal (BuildBlock error) it drains
// the mempool and returns built=false — no block was produced, so no value moved
// and the validator stays in sync. A post-build (parse/verify/accept) fault is a
// hard error (that would be a genuine consensus break, not a matcher refusal).
func tryProposeReplay(t *testing.T, proposer, validator *VM, tx *Tx) (uint64, bool) {
	t.Helper()
	ctx := context.Background()
	proposer.mempool.Add(tx)
	built, err := proposer.BuildBlock(ctx)
	if err != nil {
		proposer.mempool.Drain(1 << 20) // clear the un-buildable tx; fail-safe, no state change
		return 0, false
	}
	rp, err := proposer.ParseBlock(ctx, built.Bytes())
	if err != nil {
		t.Fatalf("ParseBlock(proposer): %v", err)
	}
	if err := rp.Verify(ctx); err != nil {
		t.Fatalf("Verify(proposer): %v", err)
	}
	if err := rp.Accept(ctx); err != nil {
		t.Fatalf("Accept(proposer): %v", err)
	}
	applyBytes(t, validator, built.Bytes())
	if proposer.lastRoot != validator.lastRoot {
		t.Fatalf("FORK under adversarial op: proposer root %x != validator root %x",
			proposer.lastRoot[:8], validator.lastRoot[:8])
	}
	return proposer.lastAcceptedHeight, true
}

// TestByzantineConservation_AdversarialMatcherStream drives a seeded adversarial
// matcher stream and asserts conservation + validator agreement after every op,
// counting (not hiding) the matcher's fail-safe refusals.
func TestByzantineConservation_AdversarialMatcherStream(t *testing.T) {
	var (
		aLUX  = a32(0x4c5558_00000001) // base (shared across both markets)
		aLUSD = a32(0x4c555344_000001) // quote A
		aLETH = a32(0x4c455448_000001) // quote B
	)
	poolA := [32]byte{'L', 'U', 'X', '/', 'L', 'U', 'S', 'D', 0xA0}
	poolB := [32]byte{'L', 'U', 'X', '/', 'L', 'E', 'T', 'H', 0xB0}
	type market struct {
		pool        [32]byte
		base, quote [32]byte
	}
	markets := []market{{poolA, aLUX, aLUSD}, {poolB, aLUX, aLETH}}
	allAssets := [][32]byte{aLUX, aLUSD, aLETH}
	accounts := []string{"alice", "bob", "carol", "dave"}
	const perAcct = uint64(1_000_000)

	totalOps, totalRefusals := 0, 0
	for _, seed := range byzantineMatcherSeeds {
		seed := seed
		t.Run(fmt.Sprintf("seed-%d", seed), func(t *testing.T) {
			rng := rand.New(rand.NewSource(seed))
			proposer, _ := newTestVM(t, memdb.New())
			defer proposer.Shutdown(context.Background())
			validator, _ := newTestVM(t, memdb.New())
			defer validator.Shutdown(context.Background())

			imported := map[[32]byte]uint64{}

			// propose+replay a block that MUST build (funding/setup).
			mustStep := func(where string, txs ...*Tx) {
				t.Helper()
				raw := proposeBytes(t, proposer, txs...)
				applyBytes(t, validator, raw)
				if proposer.lastRoot != validator.lastRoot {
					t.Fatalf("%s: FORK — proposer %x != validator %x", where, proposer.lastRoot[:8], validator.lastRoot[:8])
				}
			}

			// Σ(available+locked) == imported for every asset, on BOTH nodes. This holds
			// after an ACCEPTED block AND after a fail-safe refusal (no value moved).
			assertConserved := func(where string) {
				t.Helper()
				for _, vm := range []*VM{proposer, validator} {
					avail, locked := ledgerTotals(t, vm)
					for _, asset := range allAssets {
						if got := avail[asset] + locked[asset]; got != imported[asset] {
							t.Fatalf("%s: CONSERVATION VIOLATED asset %x: Σ(A+L)=%d (A=%d L=%d) want I=%d (delta %d)",
								where, asset, got, avail[asset], locked[asset], imported[asset], int64(got)-int64(imported[asset]))
						}
					}
					for a := range avail {
						if _, ok := imported[a]; !ok && avail[a] != 0 {
							t.Fatalf("%s: available holds UNKNOWN asset %x=%d (value from nowhere)", where, a, avail[a])
						}
					}
				}
			}

			mustStep("open-markets", openMarketTx(t, poolA, aLUX, aLUSD), openMarketTx(t, poolB, aLUX, aLETH))
			for _, acct := range accounts {
				for _, asset := range allAssets {
					mustStep(fmt.Sprintf("fund %s %x", acct, asset), depositTx(t, acct, asset, perAcct))
					imported[asset] += perAcct
				}
			}
			assertConserved("after funding")

			type placedOrder struct {
				oid   uint64
				owner string
			}
			live := map[[32]byte][]placedOrder{}
			refusals := 0

			propose := func(tx *Tx) (uint64, bool) {
				h, built := tryProposeReplay(t, proposer, validator, tx)
				if !built {
					refusals++
				}
				return h, built
			}

			for s := 0; s < byzMatcherOpsPerSeed; s++ {
				mk := markets[rng.Intn(len(markets))]
				acct := accounts[rng.Intn(len(accounts))]

				switch rng.Intn(5) {
				case 0: // PLACE resting liquidity (builds multi-price books).
					side := uint8(rng.Intn(2))
					price := float64(1 + rng.Intn(10))
					size := float64(1 + rng.Intn(30))
					if h, ok := propose(placePoolTx(t, mk.pool, side, price, size, acct)); ok {
						live[mk.pool] = append(live[mk.pool], placedOrder{blockDeterministicID(h, 0), acct})
					}

				case 1: // SUBMIT a normal crossing order.
					side := uint8(rng.Intn(2))
					price := float64(1 + rng.Intn(10))
					size := float64(1 + rng.Intn(30))
					propose(submitPoolTx(t, mk.pool, side, false, price, size, acct))

				case 2: // OVER-FILL: an oversized taker.
					side := uint8(rng.Intn(2))
					price := float64(1 + rng.Intn(10))
					size := float64(60 + rng.Intn(120))
					propose(submitPoolTx(t, mk.pool, side, false, price, size, acct))

				case 3: // SELF-CROSS: the SAME account rests then crosses itself.
					price := float64(1 + rng.Intn(10))
					size := float64(1 + rng.Intn(30))
					if h, ok := propose(placePoolTx(t, mk.pool, zapwire.SideSell, price, size, acct)); ok {
						live[mk.pool] = append(live[mk.pool], placedOrder{blockDeterministicID(h, 0), acct})
						propose(submitPoolTx(t, mk.pool, zapwire.SideBuy, false, price+float64(rng.Intn(3)), size, acct))
					}

				case 4: // CANCEL (real, filled, or stale id).
					ids := live[mk.pool]
					var oid uint64
					owner := acct
					if len(ids) > 0 {
						po := ids[rng.Intn(len(ids))]
						oid, owner = po.oid, po.owner
					} else {
						oid = uint64(rng.Int63())
					}
					propose(cancelPoolTx(t, mk.pool, oid, owner))
				}

				assertConserved(fmt.Sprintf("seed=%d step=%d", seed, s))
				totalOps++
			}

			// The matcher's fail-safe refusals must be PER-TX, not a permanent wedge:
			// a fresh clean market must still trade after the adversarial stream.
			poolClean := [32]byte{'L', 'U', 'X', '/', 'C', 'L', 'N', 0xCC}
			mustStep("open-clean", openMarketTx(t, poolClean, aLUX, aLUSD))
			mustStep("clean-place", placePoolTx(t, poolClean, zapwire.SideSell, 5.0, 10.0, "carol"))
			if _, ok := propose(submitPoolTx(t, poolClean, zapwire.SideBuy, false, 5.0, 10.0, "dave")); !ok {
				t.Fatalf("WEDGE: a FRESH market cannot trade after the adversarial stream (seed %d)", seed)
			}
			assertConserved(fmt.Sprintf("seed=%d final", seed))
			totalRefusals += refusals
			if refusals > 0 {
				t.Logf("seed %d: %d fail-safe matcher refusals (no mint, no fork); clean market still trades", seed, refusals)
			}
		})
	}
	// REGRESSION GUARD: with the self-cross cancel-persist fix, a well-formed
	// adversarial stream (every op is funded + valid) produces ZERO matcher refusals.
	// A non-zero count means the self-cross desync (or a new one) has returned.
	if totalRefusals != 0 {
		t.Fatalf("REGRESSION: %d fail-safe matcher refusals across the stream — the self-cross desync (or a new one) is back", totalRefusals)
	}
	t.Logf("adversarial matcher property: %d seeds x ~%d ops = %d ops; %d refusals (fixed); conservation + no-fork held throughout",
		len(byzantineMatcherSeeds), byzMatcherOpsPerSeed, totalOps, totalRefusals)
}
