// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

// MatchOrderCPU is the canonical pure-Go oracle for the CUDA
// match_order_kernel in luxcpp/cuda/kernels/gpu/dex_swap.cu. Same
// semantics, byte-equal trade output (after both paths clear padding):
// every CUDA dispatch is required to produce the same trade list and
// remaining quantity for the same (incoming, book, indices) inputs.
//
// Semantics match the kernel:
//   - one incoming order (taker) matched against book orders identified
//     by `bookIndices`
//   - a book order matches if its side is opposite incoming's AND the
//     prices cross (bid >= ask)
//   - book orders with status FILLED or CANCELLED are skipped
//   - quantities filled at the maker's price; remaining is decremented;
//     status flips to PARTIAL / FILLED accordingly
//   - one DEXTrade emitted per fill, with maker = book order,
//     taker = incoming, trade_id = tradeIDBase + i
//
// Returns: trades emitted, remaining quantity on the incoming order.
// Book orders in `book` may be mutated in place to reflect new
// Remaining / Status values, matching the kernel's atomic updates.
func MatchOrderCPU(
	incoming *DEXOrder,
	book []DEXOrder,
	bookIndices []uint32,
	tradeIDBase uint64,
	timestamp uint64,
) (trades []DEXTrade, remaining uint64) {
	remaining = incoming.Quantity
	if incoming.Quantity == 0 || len(bookIndices) == 0 {
		return nil, remaining
	}

	trades = make([]DEXTrade, 0, len(bookIndices))

	for _, idx := range bookIndices {
		if remaining == 0 {
			break
		}
		if int(idx) >= len(book) {
			continue
		}
		bo := &book[idx]

		if !sideMatches(incoming.Side, bo.Side) {
			continue
		}
		if !pricesCross(incoming.Side, incoming.Price, bo.Price) {
			continue
		}
		if bo.Status != DEXStatusOpen && bo.Status != DEXStatusPartial {
			continue
		}
		if bo.Remaining == 0 {
			continue
		}

		fill := remaining
		if bo.Remaining < fill {
			fill = bo.Remaining
		}
		if fill == 0 {
			continue
		}

		bo.Remaining -= fill
		if bo.Remaining == 0 {
			bo.Status = DEXStatusFilled
		} else {
			bo.Status = DEXStatusPartial
		}
		bo.ClearPadding()

		trade := DEXTrade{
			TradeID:      tradeIDBase + uint64(len(trades)),
			MakerOrderID: bo.OrderID,
			TakerOrderID: incoming.OrderID,
			MakerUserID:  bo.UserID,
			TakerUserID:  incoming.UserID,
			Price:        bo.Price,
			Quantity:     fill,
			Timestamp:    timestamp,
			TakerSide:    incoming.Side,
		}
		trade.ClearPadding()
		trades = append(trades, trade)

		remaining -= fill
	}

	return trades, remaining
}

// sideMatches: taker's side must be opposite the maker's.
func sideMatches(takerSide, makerSide uint8) bool {
	return (takerSide == DEXSideBid && makerSide == DEXSideAsk) ||
		(takerSide == DEXSideAsk && makerSide == DEXSideBid)
}

// pricesCross: bid >= ask in fixed-point Q64.64.
func pricesCross(takerSide uint8, takerPx, makerPx DEXPrice) bool {
	var bid, ask DEXPrice
	if takerSide == DEXSideBid {
		bid, ask = takerPx, makerPx
	} else {
		bid, ask = makerPx, takerPx
	}
	if bid.Integer != ask.Integer {
		return bid.Integer > ask.Integer
	}
	return bid.Fraction >= ask.Fraction
}
