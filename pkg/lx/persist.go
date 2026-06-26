// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package lx

import (
	"encoding/binary"
	"math"
	"sort"
	"unsafe"
)

// persist.go is the DEXOrder <-> OrderBook bridge: the deterministic, flat-row
// serialization the d-chain VM uses to persist a book to durable storage and to
// rebuild it on restart. pkg/lx has no other persistence; this is the single,
// canonical encoding.
//
// THE ROW IS THE GPU ABI. A persisted resting order is a 64-byte DEXOrder
// (orderbook_cuda_types.go) — the exact, byte-for-byte layout the CUDA/HIP/Metal
// matching kernels consume. Using one representation for "the order as stored"
// and "the order as fed to the matcher" means there is exactly one row format in
// the system: zero divergence between what consensus commits and what the
// accelerator matches.
//
// PRICE/SIZE ARE Q64.64 FIXED-POINT, NEVER float64. The book's in-RAM Order
// carries float64 Price/Size for the legacy API, but float64 is NOT
// byte-reproducible across architectures (x86_64 vs aarch64 rounding differs at
// the ULP), so persisting or merkleizing the float would fork consensus across a
// heterogeneous validator set. The row therefore stores the integer Q64.64 form
// (DEXPrice{Integer, Fraction}) and an integer base quantity in 1e8 fixed units.
// The conversion here is the ONE place float64 crosses into the deterministic
// domain, and it is a pure, total, deterministic function of the float bits.
//
// USER IDENTITY IN A ROW IS 8 BYTES. DEXOrder.UserID is a uint64 (the GPU ABI's
// user field, used for on-device self-trade prevention). The wire/protocol user
// is a 16-byte field and the book's Order.User is a string; the row folds that to
// the leading 8 bytes (big-endian). This is sufficient for the matcher's STP and
// for rebuilding price-time priority. The CANONICAL 16-byte settlement identity is
// NOT a property of the order-book row — it lives in d-chain state: the custody
// LEDGER keys balance:/locked: by the full user[16] (so two addresses sharing a
// leading 8-byte prefix are DISTINCT accounts, never able to drain each other),
// the idempotency index keys on user[16]+seq, and orderuser:<orderID> carries each
// resting order's full user[16] so a maker fill credits the right account even
// after a row-rebuilt restart truncated the in-RAM user to 8 bytes. Keeping the
// row at the GPU ABI's 8-byte user is the decomplected choice: the book row models
// matching, d-chain state models settlement identity.

// QuantityScale is the fixed-point scale for the integer base quantity stored in
// a DEXOrder row (Quantity / Remaining). A float64 size s is stored as
// round(s * 1e8); 1e8 matches PriceMultiplier so the integer domain is uniform
// across price and size. The matcher's resting quantities are always
// non-negative, so the unsigned domain is exact for any size representable in
// 1e8 units up to ~1.8e11 base units.
const QuantityScale = float64(PriceMultiplier) // 1e8

// floatToFixedPrice converts a float64 price to the Q64.64 DEXPrice form
// deterministically. The integer part is the floor; the fractional part is the
// remaining mantissa scaled by 2^64 and rounded to nearest. This is a pure
// function of the IEEE-754 value: every architecture that agrees on the float
// bits (which they do for a value carried verbatim on the wire) computes the
// identical {Integer, Fraction}. Negative or non-finite inputs clamp to zero;
// the matcher rejects those upstream so they never reach a persisted row.
func floatToFixedPrice(price float64) DEXPrice {
	if !(price > 0) { // also catches NaN
		return DEXPrice{}
	}
	intPart := math.Floor(price)
	if intPart > math.MaxUint64 {
		// Saturate rather than wrap; safePriceToInt already bounds real prices
		// far below this, so this is a defensive clamp, not a live path.
		return DEXPrice{Integer: math.MaxUint64, Fraction: math.MaxUint64}
	}
	frac := price - intPart
	// frac in [0,1); scale to the 64-bit fraction domain. math.Round gives the
	// nearest integer; 2^64 is represented exactly as a float64.
	fracUnits := math.Round(frac * 18446744073709551616.0) // 2^64
	var fraction uint64
	if fracUnits >= 18446744073709551616.0 {
		// Rounding pushed it to a full unit: carry into the integer part.
		return DEXPrice{Integer: uint64(intPart) + 1, Fraction: 0}
	}
	fraction = uint64(fracUnits)
	return DEXPrice{Integer: uint64(intPart), Fraction: fraction}
}

// Float converts a Q64.64 DEXPrice to float64 — the single inverse of the
// fixed-point price form, used to repopulate the in-RAM Order.Price on rebuild
// and to render a committed price for display/read APIs. The deterministic state
// (rows, merkle leaves) always uses the integer form, so this float projection
// never affects consensus.
func (p DEXPrice) Float() float64 {
	return float64(p.Integer) + float64(p.Fraction)/18446744073709551616.0 // /2^64
}

// fixedPriceToFloat is the package-internal alias for DEXPrice.Float (one
// implementation; the existing call sites keep their name).
func fixedPriceToFloat(p DEXPrice) float64 { return p.Float() }

// FixedQtyToFloat converts an integer 1e8 fixed-unit base quantity back to
// float64 — the single inverse of floatToFixedQty, exported so read APIs render
// a committed quantity through the SAME conversion the rebuild path uses.
func FixedQtyToFloat(q uint64) float64 { return float64(q) / QuantityScale }

// floatToFixedQty converts a float64 base quantity to the integer 1e8 fixed-unit
// form stored in a row. Rounds to nearest; clamps a negative/NaN to zero.
func floatToFixedQty(qty float64) uint64 {
	if !(qty > 0) {
		return 0
	}
	scaled := math.Round(qty * QuantityScale)
	if scaled >= math.MaxUint64 {
		return math.MaxUint64
	}
	return uint64(scaled)
}

// fixedQtyToFloat is the package-internal alias for FixedQtyToFloat (one
// implementation; the existing call sites keep their name).
func fixedQtyToFloat(q uint64) float64 { return FixedQtyToFloat(q) }

// userToUint64 folds a user identity string to the row's 8-byte UserID
// (big-endian over the leading 8 bytes, zero-padded). Deterministic and total.
func userToUint64(user string) uint64 {
	var b [8]byte
	copy(b[:], user)
	return binary.BigEndian.Uint64(b[:])
}

// uint64ToUser renders an 8-byte UserID back to the trimmed string form used by
// the in-RAM book. Trailing NULs are removed so a short user round-trips to its
// original string (the common case where the user fits in 8 bytes).
func uint64ToUser(id uint64) string {
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], id)
	n := 8
	for n > 0 && b[n-1] == 0 {
		n--
	}
	return string(b[:n])
}

// sideToDEX maps the book Side to the DEXOrder side byte.
func sideToDEX(s Side) uint8 {
	if s == Sell {
		return DEXSideAsk
	}
	return DEXSideBid
}

// dexToSide maps a DEXOrder side byte back to the book Side.
func dexToSide(s uint8) Side {
	if s == DEXSideAsk {
		return Sell
	}
	return Buy
}

// typeToDEX maps the book OrderType to the DEXOrder type byte. Only the resting
// forms are persistable (a resting order is always a Limit by construction in
// the consensus OrderBook path); Market/IOC/FOK takers never rest and so never reach
// a row. Anything non-Limit maps to DEXOrderLimit defensively.
func typeToDEX(t OrderType) uint8 {
	switch t {
	case Market:
		return DEXOrderMarket
	default:
		return DEXOrderLimit
	}
}

// dexToType maps a DEXOrder type byte back to the book OrderType.
func dexToType(t uint8) OrderType {
	switch t {
	case DEXOrderMarket:
		return Market
	default:
		return Limit
	}
}

// statusToDEX maps a book status string to the DEXOrder status byte.
func statusToDEX(status string, remaining, size float64) uint8 {
	switch status {
	case Canceled, StatusCancelled:
		return DEXStatusCancelled
	case Filled:
		return DEXStatusFilled
	}
	if remaining < size {
		return DEXStatusPartial
	}
	return DEXStatusOpen
}

// dexToStatus maps a DEXOrder status byte back to a book status string.
func dexToStatus(s uint8) string {
	switch s {
	case DEXStatusFilled:
		return Filled
	case DEXStatusCancelled:
		return Canceled
	case DEXStatusPartial:
		return PartiallyFilled
	default:
		return Open
	}
}

// OrderToRow converts a resting Order to its canonical 64-byte DEXOrder row. The
// price/size are committed to the deterministic Q64.64 / 1e8-integer domain; the
// padding is zeroed so two equal orders always produce byte-identical rows.
func OrderToRow(o *Order) DEXOrder {
	remaining := o.RemainingSize
	if remaining == 0 && o.Size > 0 && o.Filled == 0 {
		remaining = o.Size
	}
	user := o.User
	if user == "" {
		user = o.UserID
	}
	row := DEXOrder{
		OrderID:   o.ID,
		UserID:    userToUint64(user),
		Price:     floatToFixedPrice(o.Price),
		Quantity:  floatToFixedQty(o.Size),
		Remaining: floatToFixedQty(remaining),
		Timestamp: uint64(o.Timestamp.UnixNano()),
		Side:      sideToDEX(o.Side),
		Type:      typeToDEX(o.Type),
		Status:    statusToDEX(o.Status, remaining, o.Size),
	}
	row.ClearPadding()
	return row
}

// RowToOrder reconstructs an in-RAM Order from a canonical DEXOrder row. The
// deterministic integer fields are the source of truth; the float Price/Size are
// derived for the legacy book API only. Timestamp is rebuilt from the stored
// unix-nanos so the resting order keeps its consensus-assigned, block-derived
// stamp (price-time priority is preserved exactly across a restart).
func RowToOrder(row DEXOrder) *Order {
	user := uint64ToUser(row.UserID)
	return &Order{
		ID:            row.OrderID,
		Type:          dexToType(row.Type),
		Side:          dexToSide(row.Side),
		Price:         fixedPriceToFloat(row.Price),
		Size:          fixedQtyToFloat(row.Quantity),
		RemainingSize: fixedQtyToFloat(row.Remaining),
		Timestamp:     nanosToTime(int64(row.Timestamp)),
		User:          user,
		UserID:        user,
		Status:        dexToStatus(row.Status),
	}
}

// EncodeRow writes a DEXOrder as its 64 byte on-disk value. The bytes are the
// raw little-endian struct image (matching the GPU ABI's in-memory layout on the
// little-endian targets the venue runs); a fixed 64-byte record means a row is a
// single LSM value with zero framing overhead.
func EncodeRow(row DEXOrder) []byte {
	row.ClearPadding()
	out := make([]byte, unsafe.Sizeof(DEXOrder{}))
	*(*DEXOrder)(unsafe.Pointer(&out[0])) = row
	return out
}

// DecodeRow reads a 64-byte on-disk value back into a DEXOrder. Returns ok=false
// for any value that is not exactly the row width, so a truncated/garbage record
// can never be silently reinterpreted into a different order.
func DecodeRow(b []byte) (DEXOrder, bool) {
	if uintptr(len(b)) != unsafe.Sizeof(DEXOrder{}) {
		return DEXOrder{}, false
	}
	row := *(*DEXOrder)(unsafe.Pointer(&b[0]))
	row.ClearPadding()
	return row, true
}

// EncodeTrade writes a DEXTrade as its 80 byte on-disk value (the trade-log
// record). Same flat-image discipline as EncodeRow.
func EncodeTrade(t DEXTrade) []byte {
	t.ClearPadding()
	out := make([]byte, unsafe.Sizeof(DEXTrade{}))
	*(*DEXTrade)(unsafe.Pointer(&out[0])) = t
	return out
}

// DecodeTrade reads an 80-byte on-disk value back into a DEXTrade.
func DecodeTrade(b []byte) (DEXTrade, bool) {
	if uintptr(len(b)) != unsafe.Sizeof(DEXTrade{}) {
		return DEXTrade{}, false
	}
	t := *(*DEXTrade)(unsafe.Pointer(&b[0]))
	t.ClearPadding()
	return t, true
}

// TradeToRow converts a matched Trade to its canonical 80-byte DEXTrade row,
// committing price/size to the deterministic domain. takerSide is the side the
// taker submitted with (the resting maker is the opposite side); it is supplied
// by the caller because the in-RAM Trade does not always carry it.
func TradeToRow(t Trade, takerSide Side) DEXTrade {
	row := DEXTrade{
		TradeID:     t.ID,
		MakerUserID: userToUint64(makerUser(t, takerSide)),
		TakerUserID: userToUint64(takerUser(t, takerSide)),
		Price:       floatToFixedPrice(t.Price),
		Quantity:    floatToFixedQty(t.Size),
		Timestamp:   uint64(t.Timestamp.UnixNano()),
		TakerSide:   sideToDEX(takerSide),
	}
	// Maker/taker order IDs: the buy order is BuyOrder, the sell is SellOrder;
	// the taker is whichever matches takerSide.
	if takerSide == Buy {
		row.TakerOrderID = t.BuyOrder
		row.MakerOrderID = t.SellOrder
	} else {
		row.TakerOrderID = t.SellOrder
		row.MakerOrderID = t.BuyOrder
	}
	row.ClearPadding()
	return row
}

// makerUser returns the resting (maker) user string for a trade given the taker
// side. If the taker bought, the maker sold (Seller); else the maker bought.
func makerUser(t Trade, takerSide Side) string {
	if takerSide == Buy {
		return pickUser(t.Seller, t.SellUserID)
	}
	return pickUser(t.Buyer, t.BuyUserID)
}

// takerUser returns the taker user string for a trade given the taker side.
func takerUser(t Trade, takerSide Side) string {
	if takerSide == Buy {
		return pickUser(t.Buyer, t.BuyUserID)
	}
	return pickUser(t.Seller, t.SellUserID)
}

func pickUser(primary, fallback string) string {
	if primary != "" {
		return primary
	}
	return fallback
}

// BookToRows serializes every resting order in the book to canonical DEXOrder
// rows, ordered deterministically: bids by descending price then ascending
// orderID, then asks by ascending price then ascending orderID. This ordering is
// the canonical book ordering used for the book merkle root, so the row stream is
// itself deterministic and a round-trip is byte-stable.
//
// Only the resting side of the book is serialized — the trade log and meta are
// persisted separately by the d-chain state layer. The book is, per the
// cryptographer's design, a rebuildable accelerator: these rows are sufficient to
// reconstruct it exactly.
func BookToRows(ob *OrderBook) []DEXOrder {
	if ob == nil {
		return nil
	}
	ob.mu.RLock()
	defer ob.mu.RUnlock()

	rows := make([]DEXOrder, 0, len(ob.Orders))
	for _, o := range ob.Orders {
		if o == nil {
			continue
		}
		rows = append(rows, OrderToRow(o))
	}
	sortRows(rows)
	return rows
}

// sortRows imposes the canonical book ordering on a row slice: bids (side=bid)
// first by descending price, asks by ascending price, ties broken by ascending
// orderID. Deterministic and total.
func sortRows(rows []DEXOrder) {
	sort.Slice(rows, func(i, j int) bool {
		a, b := rows[i], rows[j]
		// bids before asks
		if a.Side != b.Side {
			return a.Side == DEXSideBid
		}
		if a.Side == DEXSideBid {
			// descending price for bids
			if a.Price.Integer != b.Price.Integer {
				return a.Price.Integer > b.Price.Integer
			}
			if a.Price.Fraction != b.Price.Fraction {
				return a.Price.Fraction > b.Price.Fraction
			}
		} else {
			// ascending price for asks
			if a.Price.Integer != b.Price.Integer {
				return a.Price.Integer < b.Price.Integer
			}
			if a.Price.Fraction != b.Price.Fraction {
				return a.Price.Fraction < b.Price.Fraction
			}
		}
		return a.OrderID < b.OrderID
	})
}

// RowsToBook reconstructs a fresh OrderBook for symbol from a row stream. Each
// row becomes a resting Order inserted via ConsensusAddOrder (deterministic
// path: the row's own ID and timestamp are authoritative, never re-minted). The
// LastOrderID watermark is advanced past the highest seen ID so any future
// consensus order keeps a monotonic ID space. Fully-filled / cancelled rows are
// skipped — only live resting liquidity is rebuilt.
func RowsToBook(symbol string, rows []DEXOrder) *OrderBook {
	ob := NewOrderBook(symbol)
	var maxID uint64
	for _, row := range rows {
		if row.OrderID > maxID {
			maxID = row.OrderID
		}
		if row.Status == DEXStatusFilled || row.Status == DEXStatusCancelled {
			continue
		}
		if row.Remaining == 0 {
			continue
		}
		o := RowToOrder(row)
		o.Symbol = symbol
		ob.ConsensusAddOrder(o)
	}
	ob.LastOrderID = maxID
	return ob
}
