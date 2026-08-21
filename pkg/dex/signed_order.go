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

// SignedOrder is an Order plus an attached secp256k1 signature and the
// caller-claimed sender address. The signature covers SigningHash(); the
// expected verification is "ecrecover(SigningHash, Sig) == Sender".
//
// Sig is the standard Ethereum (r||s||v) layout where v ∈ {0, 1}.
type SignedOrder struct {
	Order
	Sig    [65]byte
	Sender common.Address
}

// SigningHash is a deterministic 32-byte digest of the order fields that the
// signature covers. The fields are packed in a fixed order with fixed widths
// so the digest is identical on every host. We deliberately keep the encoding
// minimal — IDs, symbol, side/type, price (PriceInt), size (uint64 ticks),
// sender, and the user-provided ClientID — anything beyond that is not part
// of the auth surface for the GPU-batched ingestion path.
//
// The encoding is:
//
//	uint64 ID
//	uint16 len(Symbol) || Symbol
//	uint8  Side
//	uint8  Type
//	int64  PriceInt(Price)            // overflow → returns error
//	uint64 SizeTicks(Size)            // size * PriceMultiplier (overflow → err)
//	uint16 len(ClientID) || ClientID
//	20     Sender
// sizeTickEpsilon is the slack allowed when deciding whether a size lands on a
// tick. It exists only to absorb the representation error of a decimal quantity
// in binary floating point — it is far below one tick, so it can never let two
// distinct ticks collide.
const sizeTickEpsilon = 1e-6

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

	buf := make([]byte, 0, 8+2+len(o.Symbol)+1+1+8+8+2+len(o.ClientID)+20)
	var u8 [8]byte

	binary.BigEndian.PutUint64(u8[:], o.ID)
	buf = append(buf, u8[:]...)

	binary.BigEndian.PutUint16(u8[:2], uint16(len(o.Symbol)))
	buf = append(buf, u8[:2]...)
	buf = append(buf, o.Symbol...)

	buf = append(buf, byte(o.Side))
	buf = append(buf, byte(o.Type))

	binary.BigEndian.PutUint64(u8[:], uint64(priceInt))
	buf = append(buf, u8[:]...)

	binary.BigEndian.PutUint64(u8[:], sizeTicks)
	buf = append(buf, u8[:]...)

	binary.BigEndian.PutUint16(u8[:2], uint16(len(o.ClientID)))
	buf = append(buf, u8[:2]...)
	buf = append(buf, o.ClientID...)

	buf = append(buf, o.Sender[:]...)

	var out [32]byte
	copy(out[:], crypto.Keccak256(buf))
	return out, nil
}
