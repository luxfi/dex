// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
)

// batch_match.go is the BATCH matching path: it crosses the incoming orders of
// MANY independent books in one call, so a block's worth of orders pays a single
// dispatch instead of one dispatch per order. It is the flat-ABI counterpart to
// the per-cross MatchOrderGPU/MatchOrderCPU primitive — same DEXOrder/DEXTrade
// surface, same price-time semantics — lifted to a whole batch.
//
// THE DECOMPLECT: "which books can run together" is orthogonal to "how one book
// crosses". Orders for DIFFERENT books share no state, so they are matched
// CONCURRENTLY. Orders for the SAME book are matched STRICTLY SEQUENTIALLY in the
// caller's given order, because price-time priority / FIFO within a book is the
// determinism invariant that must never be reordered. The per-book cross reuses
// the exact MatchOrderCPU oracle the single-cross path and the GPU shadow gate
// already use — there is one matching primitive, batched here, not a second copy.
//
// THE DETERMINISM CONTRACT (proved by batch_match_parity_test.go): the returned
// results, and the post-cross book state, are BYTE-IDENTICAL to matching every
// incoming order one-at-a-time, per book, in the caller's order — regardless of
// how books are scheduled across workers or split into tiles. Concurrency and
// tiling are pure throughput; they change no committed byte.
//
// GPU-READY: a GPU batch backend implements the SAME BatchMatcher interface. It
// flattens a tile of books into one device buffer (one GPU block per book — books
// independent → parallel blocks; within a block the taker stream is serial over
// the shared remaining) and dispatches ONCE per tile. The CPU matcher here is the
// authority the GPU backend is byte-compared against, exactly as MatchOrderCPU is
// the authority for MatchOrderGPU in match_verify.go.

// BatchTileSize bounds how many books a single dispatch tile processes. A batch
// larger than this is split into sequential tiles so no one dispatch is
// unbounded — an unbounded multi-million-book dispatch once hung a GPU. Books
// WITHIN a tile are matched concurrently; tiles run in sequence. This is the CPU
// matcher's default; a GPU backend uses the same tiling to bound device memory
// and kernel-launch footprint per dispatch.
const BatchTileSize = 65536

// BatchIncoming is one taker to cross against its book. Seq is the taker's global
// position in the batch (the block/tx order); it is echoed on the result so the
// caller can reassemble results into a deterministic, worker-independent order.
// TradeBase and Timestamp are the deterministic values the VM derived for this
// cross (block-context trade-id base and block timestamp) — the batch path mints
// nothing, exactly like the consensus single-cross path.
type BatchIncoming struct {
	Seq       int
	Order     DEXOrder
	TradeBase uint64
	Timestamp uint64
}

// BatchBook is all the sequential match work for ONE book: the resting makers
// (both sides; the cross selects the opposite side of each taker) and the takers
// to cross against them, IN THE ORDER they must be applied (the caller's tx
// order). Distinct BatchBooks share no state and are matched concurrently. Book
// is mutated in place to its post-cross state (Remaining/Status), matching what
// the per-order path leaves behind.
type BatchBook struct {
	Book     []DEXOrder
	Incoming []BatchIncoming
}

// BatchResult is one taker's outcome: the fills it produced and its leftover
// quantity. Seq echoes the incoming Seq for reassembly.
type BatchResult struct {
	Seq       int
	Trades    []DEXTrade
	Remaining uint64
}

// BatchMatcher matches many independent books in one call. Implementations must
// honor the determinism contract above: output byte-identical to the per-order
// sequential path for the same input ordering, whatever the scheduling.
type BatchMatcher interface {
	// MatchBatch matches every book's incoming orders and returns one BatchResult
	// per incoming order across all books, in ascending Seq order. Book slices in
	// books are mutated in place to their post-cross state.
	MatchBatch(books []BatchBook) []BatchResult
	// Name labels the active backend (e.g. "cpu-batch") for logs and benches.
	Name() string
}

// cpuBatchMatcher is the pure-Go, GPU-free BatchMatcher. It crosses books
// concurrently on the CPU using MatchOrderCPU as the per-cross primitive, bounded
// two ways: books are processed in tiles of tileSize (no unbounded dispatch), and
// within a tile a fixed worker pool crosses books in parallel. Within one book
// the takers are matched strictly in the caller's order.
type cpuBatchMatcher struct {
	workers  int
	tileSize int
}

// BatchMatchCPU is the default CPU BatchMatcher (GOMAXPROCS workers, BatchTileSize
// tiles). It never touches a GPU and is the authority a GPU batch backend is
// byte-compared against.
var BatchMatchCPU BatchMatcher = cpuBatchMatcher{workers: runtime.GOMAXPROCS(0), tileSize: BatchTileSize}

// NewCPUBatchMatcher returns a CPU BatchMatcher with an explicit worker count and
// tile size. workers <= 0 defaults to GOMAXPROCS; tileSize <= 0 defaults to
// BatchTileSize. Used for tuning and to exercise tile boundaries in tests.
func NewCPUBatchMatcher(workers, tileSize int) BatchMatcher {
	if workers <= 0 {
		workers = runtime.GOMAXPROCS(0)
	}
	if tileSize <= 0 {
		tileSize = BatchTileSize
	}
	return cpuBatchMatcher{workers: workers, tileSize: tileSize}
}

func (cpuBatchMatcher) Name() string { return "cpu-batch" }

// MatchBatch crosses every book, parallel across books within a tile and
// sequential across tiles, then reassembles results into ascending Seq order.
func (m cpuBatchMatcher) MatchBatch(books []BatchBook) []BatchResult {
	if len(books) == 0 {
		return nil
	}
	workers := m.workers
	if workers < 1 {
		workers = 1
	}
	tileSize := m.tileSize
	if tileSize < 1 {
		tileSize = BatchTileSize
	}

	// Each book writes ONLY its own output slot, so workers never coordinate and
	// the merged output is independent of which worker finished first.
	perBook := make([][]BatchResult, len(books))
	total := 0
	for _, b := range books {
		total += len(b.Incoming)
	}

	// Bounded tiles: no scheduling wave exceeds tileSize books.
	for start := 0; start < len(books); start += tileSize {
		end := start + tileSize
		if end > len(books) {
			end = len(books)
		}
		matchTile(books[start:end], perBook[start:end], workers)
	}

	// Flatten in book order, then stable-sort by Seq so the output is the exact
	// tx-interleaved order the per-order path produces — the determinism contract.
	out := make([]BatchResult, 0, total)
	for _, pb := range perBook {
		out = append(out, pb...)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Seq < out[j].Seq })
	return out
}

// matchTile crosses one tile of books using a fixed worker pool. Each worker
// claims the next book via an atomic cursor (load-balancing without a per-book
// goroutine, so a million-book tile costs `workers` goroutines, not a million).
func matchTile(books []BatchBook, perBook [][]BatchResult, workers int) {
	if len(books) == 0 {
		return
	}
	if workers > len(books) {
		workers = len(books)
	}
	if workers <= 1 {
		for i := range books {
			perBook[i] = matchOneBook(books[i])
		}
		return
	}

	var cursor int64 = -1
	var wg sync.WaitGroup
	wg.Add(workers)
	for w := 0; w < workers; w++ {
		go func() {
			defer wg.Done()
			for {
				i := int(atomic.AddInt64(&cursor, 1))
				if i >= len(books) {
					return
				}
				perBook[i] = matchOneBook(books[i])
			}
		}()
	}
	wg.Wait()
}

// matchOneBook crosses one book's takers strictly in order. Each taker sees the
// book state left by the prior takers (makers filled by an earlier cross are
// already decremented / FILLED), so price-time priority and FIFO within the book
// are preserved exactly as the per-order path would leave them.
func matchOneBook(bb BatchBook) []BatchResult {
	if len(bb.Incoming) == 0 {
		return nil
	}
	res := make([]BatchResult, 0, len(bb.Incoming))
	// One scratch index buffer reused across this book's takers: the index list is
	// dead once a cross returns, so recomputing it into the same backing array
	// (never shared across goroutines — matchOneBook owns one book) turns a
	// per-cross allocation into a per-book one.
	scratch := make([]uint32, 0, len(bb.Book))
	for _, in := range bb.Incoming {
		// The opposite side of the book in price-time priority for THIS taker,
		// recomputed each cross because a prior taker may have filled makers and a
		// buy taker vs a sell taker consume opposite sides.
		idx := oppositeSideByPriceTime(bb.Book, in.Order.Side, scratch)
		inc := in.Order
		trades, rem := MatchOrderCPU(&inc, bb.Book, idx, in.TradeBase, in.Timestamp)
		res = append(res, BatchResult{Seq: in.Seq, Trades: trades, Remaining: rem})
	}
	return res
}

// oppositeSideByPriceTime returns the indices of the makers a taker of takerSide
// can cross — the opposite side, still open, with remaining size — sorted into
// price-time priority: a buy taker consumes asks ascending in price; a sell taker
// consumes bids descending; ties broken by ascending OrderID (FIFO). This is the
// same ordering the single-cross shadow capture (sortShadowRows) imposes, applied
// to indices so the book slice positions stay stable across a book's takers. The
// result is appended into dst[:0] so a caller can reuse a scratch buffer.
func oppositeSideByPriceTime(book []DEXOrder, takerSide uint8, dst []uint32) []uint32 {
	wantSide := DEXSideAsk // buy taker crosses asks
	if takerSide == DEXSideAsk {
		wantSide = DEXSideBid // sell taker crosses bids
	}
	idx := dst[:0]
	for i := range book {
		bo := &book[i]
		if bo.Side != wantSide {
			continue
		}
		if bo.Status != DEXStatusOpen && bo.Status != DEXStatusPartial {
			continue
		}
		if bo.Remaining == 0 {
			continue
		}
		idx = append(idx, uint32(i))
	}
	sort.SliceStable(idx, func(a, b int) bool {
		pa, pb := book[idx[a]].Price, book[idx[b]].Price
		if pa.Integer != pb.Integer {
			if takerSide == DEXSideBid {
				return pa.Integer < pb.Integer // best ask = lowest price first
			}
			return pa.Integer > pb.Integer // best bid = highest price first
		}
		if pa.Fraction != pb.Fraction {
			if takerSide == DEXSideBid {
				return pa.Fraction < pb.Fraction
			}
			return pa.Fraction > pb.Fraction
		}
		return book[idx[a]].OrderID < book[idx[b]].OrderID // FIFO within a level
	})
	return idx
}
