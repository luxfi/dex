// Copyright (C) 2019-2025, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

// Package zapwire is the single, frozen source of truth for the DEX ZAP wire
// frame the D-Chain gateway (dex/pkg/api ZAPServer) speaks and that every
// client — the chains/dexvm atomic proxy and the LP-9010 EVM precompile —
// encodes against.
//
// It is a PURE-GO LEAF: it imports only the standard library, so a public,
// CGO-disabled build can import these constants and codecs WITHOUT dragging in
// the cgo/GPU matching engine that lives in dex/pkg/dex. That separation is what
// lets the public precompile stay a pure-Go module while still framing requests
// byte-identically to the venue's d-chain.
//
// The frame is FROZEN. The same bytes must round-trip across the precompile
// (engine_zap.go), the proxy relay client, and the server (zap_server.go).
// Changing any size or method name here is a wire-breaking change and must be
// versioned at the transport layer, never silently.
package zapwire

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DEX market-routed ZAP method names. These are the poolId-keyed framing of
// the order-book operations the V4 PoolManager facade (a central-limit-order
// book, NOT an AMM) exposes:
//
//   - EnsureMarket ("initialize"): idempotently create the market/book.
//   - Place       ("modifyLiquidity" +delta): rest a limit order.
//   - Cancel      ("modifyLiquidity" -delta): cancel a resting order.
//   - Submit      ("swap"): submit a MARKETABLE order, get its fills back.
//
// Wire method names use the dex_ namespace: on the D-Chain the DEX *is* the
// DEX (the AMM lives in C-Chain contracts, not this VM), so the public wire
// names follow the luxd service-prefix convention (eth_/avm./platform.) and
// expose the chain's purpose — `dex_*` — not the internal engine name. The Go
// identifiers (MethodPlace, dexMethods, "DEX") stay as the engine's internal
// name; only the on-wire string values are dex_*.
const (
	MethodEnsureMarket = "dex_ensure_market"
	MethodPlace        = "dex_place"
	MethodCancel       = "dex_cancel"
	MethodSubmit       = "dex_submit" // marketable order, returns fills
)

// DEX ack status bytes.
const (
	StatusPlaced   uint8 = 0
	StatusCanceled uint8 = 1
	StatusRejected uint8 = 2
)

// Wire frame sizes. All multi-byte integers are big-endian; all floats are
// IEEE-754 float64 in big-endian bit form.
const (
	// PoolIDSize is the V4 poolId width: keccak256(PoolKey).
	PoolIDSize = 32
	// UserSize is the on-wire user identity width (null-padded).
	UserSize = 16

	// EnsureMarketReqSize: poolId[32].
	EnsureMarketReqSize = PoolIDSize

	// PlaceReqSize: poolId[32] + side[1] + price[8] + size[8] + user[16].
	PlaceReqSize = PoolIDSize + 1 + 8 + 8 + UserSize // 65

	// CancelReqSize: poolId[32] + orderId[8].
	CancelReqSize = PoolIDSize + 8 // 40

	// SubmitReqSize: poolId[32] + side[1] + isMarket[1] + limitPrice[8] +
	// size[8] + user[16].
	SubmitReqSize = PoolIDSize + 1 + 1 + 8 + 8 + UserSize // 66

	// AckSize: orderId[8] + status[1] + seq[8] + pad[7].
	AckSize = 24

	// FillWireSize is one fill in a Submit response: price[8] + size[8] +
	// takerSide[1]. Base traded = sum(size); quote traded = sum(price*size).
	// The adapter derives the V4 BalanceDelta from these server-produced
	// numbers ONLY — it never trusts a client-supplied amount.
	FillWireSize = 17
)

// Side encodes the order side on the wire: 0=buy/bid, 1=sell/ask.
const (
	SideBuy  uint8 = 0
	SideSell uint8 = 1
)

// Fill is one decoded fill from a Submit response.
type Fill struct {
	Price     float64
	Size      float64
	TakerSide uint8
}

// PutFloat64 writes f as a big-endian IEEE-754 float64 into b[0:8].
func PutFloat64(b []byte, f float64) {
	binary.BigEndian.PutUint64(b, math.Float64bits(f))
}

// Float64 reads a big-endian IEEE-754 float64 from b[0:8].
func Float64(b []byte) float64 {
	return math.Float64frombits(binary.BigEndian.Uint64(b))
}

// padNull copies s into a fresh size-byte buffer, null-padded.
func padNull(s string, size int) []byte {
	out := make([]byte, size)
	copy(out, s)
	return out
}

// trimNull returns b with trailing NUL bytes removed.
func trimNull(b []byte) []byte {
	for i := len(b) - 1; i >= 0; i-- {
		if b[i] != 0 {
			return b[:i+1]
		}
	}
	return b[:0]
}

// EncodeEnsureMarket builds the EnsureMarket request payload.
func EncodeEnsureMarket(poolID [32]byte) []byte {
	out := make([]byte, EnsureMarketReqSize)
	copy(out[0:PoolIDSize], poolID[:])
	return out
}

// EncodePlace builds a Place (rest limit order) request payload.
func EncodePlace(poolID [32]byte, side uint8, price, size float64, user string) []byte {
	out := make([]byte, PlaceReqSize)
	copy(out[0:PoolIDSize], poolID[:])
	out[PoolIDSize] = side
	PutFloat64(out[PoolIDSize+1:PoolIDSize+9], price)
	PutFloat64(out[PoolIDSize+9:PoolIDSize+17], size)
	copy(out[PoolIDSize+17:], padNull(user, UserSize))
	return out
}

// EncodeCancel builds a Cancel request payload.
func EncodeCancel(poolID [32]byte, orderID uint64) []byte {
	out := make([]byte, CancelReqSize)
	copy(out[0:PoolIDSize], poolID[:])
	binary.BigEndian.PutUint64(out[PoolIDSize:PoolIDSize+8], orderID)
	return out
}

// EncodeSubmit builds a Submit (marketable order) request payload. When
// isMarket is true the order sweeps unconditionally and limitPrice is ignored;
// otherwise it is an IOC limit bounded by limitPrice.
func EncodeSubmit(poolID [32]byte, side uint8, isMarket bool, limitPrice, size float64, user string) []byte {
	out := make([]byte, SubmitReqSize)
	copy(out[0:PoolIDSize], poolID[:])
	out[PoolIDSize] = side
	if isMarket {
		out[PoolIDSize+1] = 1
	}
	PutFloat64(out[PoolIDSize+2:PoolIDSize+10], limitPrice)
	PutFloat64(out[PoolIDSize+10:PoolIDSize+18], size)
	copy(out[PoolIDSize+18:], padNull(user, UserSize))
	return out
}

// DecodeFills parses a Submit response: fillCount[4] then fillCount ×
// (price[8] + size[8] + takerSide[1]). Returns an error on a malformed frame.
func DecodeFills(resp []byte) ([]Fill, error) {
	if len(resp) < 4 {
		return nil, fmt.Errorf("zapwire: submit response too short: %d", len(resp))
	}
	n := int(binary.BigEndian.Uint32(resp[0:4]))
	if len(resp) < 4+n*FillWireSize {
		return nil, fmt.Errorf("zapwire: submit response truncated: have %d, need %d for %d fills",
			len(resp), 4+n*FillWireSize, n)
	}
	fills := make([]Fill, n)
	off := 4
	for i := 0; i < n; i++ {
		fills[i] = Fill{
			Price:     Float64(resp[off : off+8]),
			Size:      Float64(resp[off+8 : off+16]),
			TakerSide: resp[off+16],
		}
		off += FillWireSize
	}
	return fills, nil
}

// EncodeFills builds a Submit response from fills (server side / test fakes).
func EncodeFills(fills []Fill) []byte {
	out := make([]byte, 4+len(fills)*FillWireSize)
	binary.BigEndian.PutUint32(out[0:4], uint32(len(fills)))
	off := 4
	for _, f := range fills {
		PutFloat64(out[off:off+8], f.Price)
		PutFloat64(out[off+8:off+16], f.Size)
		out[off+16] = f.TakerSide
		off += FillWireSize
	}
	return out
}

// DecodeSubmit parses a Submit request payload into its fields. Clients build
// these; the server and test harnesses decode them.
func DecodeSubmit(payload []byte) (poolID [32]byte, side uint8, isMarket bool, limitPrice, size float64, user string, err error) {
	if len(payload) < SubmitReqSize {
		err = fmt.Errorf("zapwire: submit request too short: %d", len(payload))
		return
	}
	copy(poolID[:], payload[0:PoolIDSize])
	side = payload[PoolIDSize]
	isMarket = payload[PoolIDSize+1] == 1
	limitPrice = Float64(payload[PoolIDSize+2 : PoolIDSize+10])
	size = Float64(payload[PoolIDSize+10 : PoolIDSize+18])
	user = string(trimNull(payload[PoolIDSize+18 : PoolIDSize+18+UserSize]))
	return
}

// DecodePlace parses a Place request payload into its fields (poolId[32] +
// side[1] + price[8] + size[8] + user[16]).
func DecodePlace(payload []byte) (poolID [32]byte, side uint8, price, size float64, user string, err error) {
	if len(payload) < PlaceReqSize {
		err = fmt.Errorf("zapwire: place request too short: %d", len(payload))
		return
	}
	copy(poolID[:], payload[0:PoolIDSize])
	side = payload[PoolIDSize]
	price = Float64(payload[PoolIDSize+1 : PoolIDSize+9])
	size = Float64(payload[PoolIDSize+9 : PoolIDSize+17])
	user = string(trimNull(payload[PoolIDSize+17 : PoolIDSize+17+UserSize]))
	return
}

// DecodeCancel parses a Cancel request payload (poolId[32] + orderId[8]).
func DecodeCancel(payload []byte) (poolID [32]byte, orderID uint64, err error) {
	if len(payload) < CancelReqSize {
		err = fmt.Errorf("zapwire: cancel request too short: %d", len(payload))
		return
	}
	copy(poolID[:], payload[0:PoolIDSize])
	orderID = binary.BigEndian.Uint64(payload[PoolIDSize : PoolIDSize+8])
	return
}

// EncodeAck builds an ack response (orderId, status, seq).
func EncodeAck(orderID uint64, status uint8, seq uint64) []byte {
	out := make([]byte, AckSize)
	binary.BigEndian.PutUint64(out[0:8], orderID)
	out[8] = status
	binary.BigEndian.PutUint64(out[9:17], seq)
	return out
}

// DecodeAck reads orderId and status from an ack response. A reject frame
// (orderId[8] + status[1] + reasonLen[2] + reason) is also accepted: status is
// read from byte 8 and the rest ignored.
func DecodeAck(resp []byte) (orderID uint64, status uint8, err error) {
	if len(resp) < 9 {
		err = fmt.Errorf("zapwire: ack response too short: %d", len(resp))
		return
	}
	orderID = binary.BigEndian.Uint64(resp[0:8])
	status = resp[8]
	return
}

// =========================================================================
// CUSTODY frames — deposit / withdraw / open-market.
//
// These are the funds-in / funds-out / market-binding operations of the DEX
// "money lives in the order book" model: a DEPOSIT credits an account's
// available D-Chain balance (what the book draws from), a WITHDRAW debits a
// realized balance back out, and OPEN-MARKET binds a market's (base, quote)
// asset identities so an order knows which asset it spends.
//
// They are ADDITIVE and ORTHOGONAL to the four FROZEN order frames above
// (EnsureMarket/Place/Cancel/Submit), which are byte-identical across the
// precompile, the proxy relay, and the d-chain gateway and MUST stay so. The
// custody frames are new methods with their own fixed sizes; adding them does
// not touch a single byte of the frozen order frames. They are still FROZEN in
// the same sense — the precompile/proxy/d-chain re-define identical constants —
// just newer.
//
// Asset identity is the FULL 32-byte asset id (AssetIDSize), NOT a truncated
// handle. Keying a value-bearing custody ledger by a truncated id is unsound: a
// truncation maps distinct assets to the same key, so a worthless token whose id
// folds to the native-LUX key could mint a native claim and drain the native
// vault (and two tokens sharing the truncated prefix collide). The id is
// therefore carried and keyed at FULL width, derived by an INJECTIVE map on each
// rail so distinct assets NEVER collide:
//   - native LUX  = a reserved all-zero id (ids.Empty on the proxy rail);
//   - ERC-20      = the 20-byte C-Chain address embedded in 32 bytes (left-pad),
//     so two distinct token addresses — or a token vs native — map
//     to two distinct ids.
//
// The d-chain keys balance:/locked: (and the market (base,quote) binding and the
// per-order reserve) by this full 32-byte id. The DEXOrder/DEXTrade matcher rows
// carry NO asset field (they key by user+order id), so widening the asset leaves
// the FROZEN matcher row ABI byte-identical — the asset lives only in the ledger
// keyspace, never on the hot order/trade frames.
//
// IDEMPOTENCY REFERENCE (ref[32]) — the funds-safety field. A deposit (mint)
// and a withdraw (burn + vault release) are IRREVERSIBLE, so "is this the same
// operation?" MUST mean the same thing on BOTH sides of the rail: the EVM
// precompile, which keys replay-idempotency on the executing tx, and the
// d-chain, which content-addresses its dedup index (seen:<txID>) on the FULL
// frame. The custody frames therefore carry a 32-byte ref — the originating
// transaction's identity (EVM txHash on the precompile rail; the consumed
// import / settlement reference on the chains/dexvm atomic rail). Because the
// d-chain txID = Checksum256(type‖body) hashes the WHOLE body, the ref is folded
// into the seen: key automatically: two GENUINELY distinct custody ops (distinct
// originating tx, hence distinct ref) are distinct on the d-chain too — so a
// second genuine withdraw re-clamps to CURRENT available (0 after the first) and
// releases nothing, instead of the seen: index replaying the first realized
// amount against a vault that already paid (the drain). A byte-identical retry of
// ONE op (same ref) still dedups to exactly-once. The ref is ASSET-GENERIC: it is
// opaque 32 bytes that serves the native (address(0)) and ERC-20 rails identically
// — the asset[8] handle distinguishes the currency, the ref distinguishes the op.
//
// The ref is APPENDED at the tail of the custody body, so user[16]/asset[8]/
// amount[8] keep their original offsets (0/16/24) — every existing decode site is
// unchanged; only the body grows by 32 and a new ref read is added. The four
// FROZEN order frames are NOT touched: their sizes, offsets, and codecs are
// byte-identical to before (the matchers byte-match on them), and this block adds
// nothing to them.
const (
	// MethodDeposit credits an account's available balance from value the proxy
	// atomically imported (shared-memory ImportTx). Funds-IN leg.
	MethodDeposit = "dex_deposit"
	// MethodWithdraw debits an account's realized available balance for the
	// proxy to atomically export (shared-memory ExportTx). Funds-OUT leg.
	MethodWithdraw = "dex_withdraw"
	// MethodOpenMarket binds a market's (base, quote) asset handles. Idempotent;
	// the asset binding is what lets place/submit know which asset to lock.
	MethodOpenMarket = "dex_open_market"
)

const (
	// AssetIDSize is the on-wire asset id width: the FULL 32-byte injective asset
	// id (NOT a truncated handle). It keys the d-chain balance:/locked: ledger and
	// the market (base,quote) binding. See the custody block comment for why a
	// truncated key is unsound (native-vault drain via id collision).
	AssetIDSize = 32

	// RefSize is the on-wire idempotency-reference width: the 32-byte identity of
	// the originating transaction (EVM txHash / atomic import reference) that the
	// d-chain folds into its content-addressed seen: dedup key. See the custody
	// block comment above.
	RefSize = 32

	// DepositReqSize: user[16] + asset[32] + amount[8] + ref[32].
	DepositReqSize = UserSize + AssetIDSize + 8 + RefSize // 88

	// WithdrawReqSize: user[16] + asset[32] + amount[8] + ref[32].
	WithdrawReqSize = UserSize + AssetIDSize + 8 + RefSize // 88

	// OpenMarketReqSize: poolId[32] + baseAsset[32] + quoteAsset[32].
	OpenMarketReqSize = PoolIDSize + AssetIDSize + AssetIDSize // 96

	// BalanceReqSize: user[16] + asset[32]. The read-only balance observation.
	BalanceReqSize = UserSize + AssetIDSize // 48

	// BalanceRespSize: status[1] + amount[8]. A deposit/withdraw response reports
	// the amount actually credited/debited (a withdraw clamps to available, so the
	// caller — the proxy export leg — learns the exact realized amount to export).
	BalanceRespSize = 1 + 8

	// custodyRefOff is the byte offset of ref[32] within a deposit/withdraw body
	// (after user[16]+asset[32]+amount[8]).
	custodyRefOff = UserSize + AssetIDSize + 8 // 56

	// custodyAmountOff is the byte offset of amount[8] within a deposit/withdraw
	// body (after user[16]+asset[32]).
	custodyAmountOff = UserSize + AssetIDSize // 48
)

// EncodeDeposit builds a deposit request: user[16] + asset[32] + amount[8] +
// ref[32]. asset is the FULL 32-byte injective asset id (see the custody block
// comment). ref is the originating-tx idempotency reference; it folds into the
// d-chain seen: key via the content-addressed txID.
func EncodeDeposit(user string, asset [32]byte, amount uint64, ref [32]byte) []byte {
	out := make([]byte, DepositReqSize)
	copy(out[0:UserSize], padNull(user, UserSize))
	copy(out[UserSize:custodyAmountOff], asset[:])
	binary.BigEndian.PutUint64(out[custodyAmountOff:custodyRefOff], amount)
	copy(out[custodyRefOff:custodyRefOff+RefSize], ref[:])
	return out
}

// DecodeDeposit parses a deposit request, returning the full 32-byte asset id and
// the idempotency ref.
func DecodeDeposit(payload []byte) (user string, asset [32]byte, amount uint64, ref [32]byte, err error) {
	if len(payload) < DepositReqSize {
		err = fmt.Errorf("zapwire: deposit payload too short: %d", len(payload))
		return
	}
	user = string(trimNull(payload[0:UserSize]))
	copy(asset[:], payload[UserSize:custodyAmountOff])
	amount = binary.BigEndian.Uint64(payload[custodyAmountOff:custodyRefOff])
	copy(ref[:], payload[custodyRefOff:custodyRefOff+RefSize])
	return
}

// EncodeWithdraw builds a withdraw request: user[16] + asset[32] + amount[8] +
// ref[32]. asset is the FULL 32-byte injective asset id; amount is the requested
// withdrawal (the d-chain clamps it to available and reports the realized amount
// in the response). ref is the originating-tx idempotency reference.
func EncodeWithdraw(user string, asset [32]byte, amount uint64, ref [32]byte) []byte {
	out := make([]byte, WithdrawReqSize)
	copy(out[0:UserSize], padNull(user, UserSize))
	copy(out[UserSize:custodyAmountOff], asset[:])
	binary.BigEndian.PutUint64(out[custodyAmountOff:custodyRefOff], amount)
	copy(out[custodyRefOff:custodyRefOff+RefSize], ref[:])
	return out
}

// DecodeWithdraw parses a withdraw request, returning the full 32-byte asset id
// and the idempotency ref.
func DecodeWithdraw(payload []byte) (user string, asset [32]byte, amount uint64, ref [32]byte, err error) {
	if len(payload) < WithdrawReqSize {
		err = fmt.Errorf("zapwire: withdraw payload too short: %d", len(payload))
		return
	}
	user = string(trimNull(payload[0:UserSize]))
	copy(asset[:], payload[UserSize:custodyAmountOff])
	amount = binary.BigEndian.Uint64(payload[custodyAmountOff:custodyRefOff])
	copy(ref[:], payload[custodyRefOff:custodyRefOff+RefSize])
	return
}

// EncodeOpenMarket builds an open-market request: poolId[32] + baseAsset[32] +
// quoteAsset[32]. The assets are the FULL 32-byte injective ids the d-chain keys
// the ledger by, so an order's locked spend names the SAME asset key a deposit
// credited.
func EncodeOpenMarket(poolID [32]byte, baseAsset, quoteAsset [32]byte) []byte {
	out := make([]byte, OpenMarketReqSize)
	copy(out[0:PoolIDSize], poolID[:])
	copy(out[PoolIDSize:PoolIDSize+AssetIDSize], baseAsset[:])
	copy(out[PoolIDSize+AssetIDSize:], quoteAsset[:])
	return out
}

// DecodeOpenMarket parses an open-market request.
func DecodeOpenMarket(payload []byte) (poolID [32]byte, baseAsset, quoteAsset [32]byte, err error) {
	if len(payload) < OpenMarketReqSize {
		err = fmt.Errorf("zapwire: open-market payload too short: %d", len(payload))
		return
	}
	copy(poolID[:], payload[0:PoolIDSize])
	copy(baseAsset[:], payload[PoolIDSize:PoolIDSize+AssetIDSize])
	copy(quoteAsset[:], payload[PoolIDSize+AssetIDSize:PoolIDSize+2*AssetIDSize])
	return
}

// EncodeBalanceResp builds a deposit/withdraw response: status[1] + amount[8],
// where amount is the realized credited/debited value.
func EncodeBalanceResp(status uint8, amount uint64) []byte {
	out := make([]byte, BalanceRespSize)
	out[0] = status
	binary.BigEndian.PutUint64(out[1:9], amount)
	return out
}

// DecodeBalanceResp reads (status, realized amount) from a balance response.
func DecodeBalanceResp(resp []byte) (status uint8, amount uint64, err error) {
	if len(resp) < BalanceRespSize {
		err = fmt.Errorf("zapwire: balance response too short: %d", len(resp))
		return
	}
	status = resp[0]
	amount = binary.BigEndian.Uint64(resp[1:9])
	return
}
