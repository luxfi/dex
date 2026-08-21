// Copyright (C) 2025-2026, Lux Industries Inc. All rights reserved.
// See the file LICENSE for licensing terms.

package dex

import (
	"encoding/binary"
	"errors"
	"math"

	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
)

// SignedOrder is an Order, the auth SCOPE the signature is valid within, and an
// attached secp256k1 signature over the caller-claimed sender. The signature
// covers SigningHash(); the expected verification is
// "ecrecover(SigningHash, Sig) == Sender".
//
// Sig is the standard Ethereum (r||s||v) layout where v ∈ {0, 1}.
//
// ChainID, NetworkID and Expiry are the authorization scope, and they live on the
// envelope rather than on the matcher's Order because they say WHERE and UNTIL
// WHEN a signature is good — not what to match. SigningHash binds them, so a
// signature made for one chain, one network or before one deadline does not
// recover its signer on another chain, another network, or after it expires. The
// D-Chain tx digest (pkg/dchain txAuthDigest) binds the same scope for the same
// reason: a signature has to name the chain it spends on to spend on only one.
type SignedOrder struct {
	Order
	Sig       [65]byte
	Sender    common.Address
	ChainID   [32]byte
	NetworkID uint32
	Expiry    uint64
}

// orderAuthDomain tags the order-authorization digest. A versioned domain
// separator keeps an order signature from ever colliding with any other
// keccak256 preimage in the stack, and makes the next encoding a distinct
// namespace instead of a silent reinterpretation of these bytes. It mirrors the
// D-Chain's txAuthDomain ("lux.dchain.tx.auth.v2").
const orderAuthDomain = "lux.dex.order.auth.v1"

// sizeTickEpsilon is the slack allowed when deciding whether a size lands on a
// tick. It exists only to absorb the representation error of a decimal quantity
// in binary floating point — it is far below one tick, so it can never let two
// distinct ticks collide.
const sizeTickEpsilon = 1e-6

// SigningHash is a deterministic 32-byte digest of everything a signature over an
// order authorizes: the auth scope, and EVERY field the matcher reads to reach a
// decision. A field the digest omits can be rewritten by any relay between the
// signer and the matcher while the signature still verifies, so the signed
// subset must BE the whole order the matcher acts on — the price guard, the
// post-only / reduce-only / STP flags, the stop and bracket legs, the iceberg's
// visible size, the lifetime, and the account the fill is attributed to.
//
// The encoding is a hand-written fixed-width layout so a field cannot silently
// drop out of the preimage. Integers are big-endian; every string is a uint16
// length followed by its bytes; every advanced-order float is bound by its exact
// IEEE-754 bits — identical on every architecture and injective, so two distinct
// values never share a digest and no float→int conversion (whose out-of-range
// result differs by CPU arch) enters the preimage. Price and Size keep the
// settled-grid encoding the off-tick refusal below depends on.
//
//	"lux.dex.order.auth.v1"           domain separator
//	uint32  NetworkID                 } scope: one signature is good on exactly
//	32      ChainID                   } one network and one chain, and only
//	uint64  Expiry                    } until one deadline
//	uint64  ID
//	uint16  len(Symbol) || Symbol
//	uint8   Side
//	uint8   Type
//	int64   PriceInt(Price)            // overflow → returns error
//	uint64  SizeTicks(Size)            // off-tick → returns error
//	uint64  StopPrice   (IEEE-754 bits)
//	uint64  LimitPrice  (IEEE-754 bits)
//	uint64  DisplaySize (IEEE-754 bits)
//	uint64  PegOffset   (IEEE-754 bits)
//	uint64  TakeProfit  (IEEE-754 bits)
//	uint64  StopLoss    (IEEE-754 bits)
//	uint16  len(TimeInForce) || TimeInForce
//	uint8   PostOnly
//	uint8   ReduceOnly
//	uint8   Hidden
//	uint32  Flags
//	uint16  len(UserID) || UserID
//	uint16  len(User) || User
//	uint16  len(ClientID) || ClientID
//	20      Sender
func (o *SignedOrder) SigningHash() ([32]byte, error) {
	priceInt, err := safePriceToInt(o.Price)
	if err != nil {
		return [32]byte{}, err
	}
	if o.Size < 0 || o.Size > MaxSafePrice {
		return [32]byte{}, errors.New("size out of range for SigningHash")
	}
	// The digest must cover the size the matcher ACTS on, not a rounded stand-in
	// for it. Rounding here made the hash non-injective: two sizes that round to
	// one tick shared a digest, so a signature taken over one could be presented
	// with the other and still verify. On a permissionless exchange that is a
	// signed order whose quantity the holder can change.
	//
	// A size off-tick is refused rather than snapped. Snapping decides, silently,
	// that the signer meant something they did not write — and the whole point of
	// the signature is that only the signer decides that.
	scaled := o.Size * PriceMultiplier
	sizeTicks := uint64(math.Round(scaled))
	if math.Abs(scaled-float64(sizeTicks)) > sizeTickEpsilon {
		return [32]byte{}, errors.New("size is not an exact number of ticks; refusing to round inside a signature")
	}

	appendStr := func(buf []byte, s string) []byte {
		buf = binary.BigEndian.AppendUint16(buf, uint16(len(s)))
		return append(buf, s...)
	}
	boolByte := func(b bool) byte {
		if b {
			return 1
		}
		return 0
	}

	buf := make([]byte, 0, 176+len(o.Symbol)+len(o.TimeInForce)+len(o.UserID)+len(o.User)+len(o.ClientID))

	buf = append(buf, orderAuthDomain...)

	buf = binary.BigEndian.AppendUint32(buf, o.NetworkID)
	buf = append(buf, o.ChainID[:]...)
	buf = binary.BigEndian.AppendUint64(buf, o.Expiry)

	buf = binary.BigEndian.AppendUint64(buf, o.ID)
	buf = appendStr(buf, o.Symbol)
	buf = append(buf, byte(o.Side), byte(o.Type))
	buf = binary.BigEndian.AppendUint64(buf, uint64(priceInt))
	buf = binary.BigEndian.AppendUint64(buf, sizeTicks)

	buf = binary.BigEndian.AppendUint64(buf, math.Float64bits(o.StopPrice))
	buf = binary.BigEndian.AppendUint64(buf, math.Float64bits(o.LimitPrice))
	buf = binary.BigEndian.AppendUint64(buf, math.Float64bits(o.DisplaySize))
	buf = binary.BigEndian.AppendUint64(buf, math.Float64bits(o.PegOffset))
	buf = binary.BigEndian.AppendUint64(buf, math.Float64bits(o.TakeProfit))
	buf = binary.BigEndian.AppendUint64(buf, math.Float64bits(o.StopLoss))

	buf = appendStr(buf, o.TimeInForce)
	buf = append(buf, boolByte(o.PostOnly), boolByte(o.ReduceOnly), boolByte(o.Hidden))
	buf = binary.BigEndian.AppendUint32(buf, uint32(o.Flags))

	buf = appendStr(buf, o.UserID)
	buf = appendStr(buf, o.User)
	buf = appendStr(buf, o.ClientID)
	buf = append(buf, o.Sender[:]...)

	var out [32]byte
	copy(out[:], crypto.Keccak256(buf))
	return out, nil
}
