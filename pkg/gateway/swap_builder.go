package gateway

import (
	"encoding/hex"
	"math/big"
	"strings"
)

// V4 precompile swap function selector: swap((address,address,uint24,int24,address),(bool,int256,uint160),bytes)
// keccak256 first 4 bytes = 0xF3CD914C
var swapSelector = [4]byte{0xF3, 0xCD, 0x91, 0x4C}

// LXPool precompile address (LP-9010)
const LXPoolPrecompile = "0x0000000000000000000000000000000000009010"

// V4 sqrt price limits (used when caller wants "swap as much as possible")
var (
	// MIN_SQRT_RATIO + 1 — floor price for zeroForOne swaps
	MinSqrtRatio = func() *big.Int {
		v, _ := new(big.Int).SetString("4295128740", 10)
		return v
	}()
	// MAX_SQRT_RATIO - 1 — ceiling price for !zeroForOne swaps
	MaxSqrtRatio = func() *big.Int {
		v, _ := new(big.Int).SetString("1461446703485210103287273052203988822378723970341", 10)
		return v
	}()
)

// swapRequest is the JSON body for POST /v1/swap
type swapRequest struct {
	TokenIn   string  `json:"tokenIn"`
	TokenOut  string  `json:"tokenOut"`
	ChainID   uint64  `json:"chainId"`
	Amount    string  `json:"amount"`
	IsExactIn bool    `json:"isExactIn"`
	Slippage  float64 `json:"slippage,omitempty"`
	Recipient string  `json:"recipient"`
	Deadline  int64   `json:"deadline,omitempty"`
	Fee       uint32  `json:"fee,omitempty"`
	Hooks     string  `json:"hooks,omitempty"`
}

// swapResponse is the API response for POST /v1/swap
type swapResponse struct {
	Swap  swapTx    `json:"swap"`
	Quote SwapQuote `json:"quote"`
}

// swapTx is the unsigned transaction for the client to sign
type swapTx struct {
	To       string `json:"to"`
	Data     string `json:"data"`
	Value    string `json:"value"`
	ChainID  uint64 `json:"chainId"`
	GasLimit uint64 `json:"gasLimit"`
}

// buildSwapCalldata encodes V4 precompile swap calldata.
//
// ABI layout (after 4-byte selector):
//
//	[0:160]   PoolKey: currency0(32) + currency1(32) + fee(32) + tickSpacing(32) + hooks(32)
//	[160:192] zeroForOne (bool, 32 bytes)
//	[192:224] amountSpecified (int256, negative=exactIn, positive=exactOut)
//	[224:256] sqrtPriceLimitX96 (uint160)
//	[256:288] hookData offset
//	[288:320] hookData length (0 = empty)
func buildSwapCalldata(
	currency0, currency1 string,
	fee uint32,
	tickSpacing int32,
	hooks string,
	zeroForOne bool,
	amountSpecified *big.Int,
	sqrtPriceLimitX96 *big.Int,
) []byte {
	data := make([]byte, 320)

	// Selector
	copy(data[0:4], swapSelector[:])

	// PoolKey (5 x 32-byte slots)
	off := 4
	copy(data[off+12:off+32], addrBytes(currency0))
	off += 32
	copy(data[off+12:off+32], addrBytes(currency1))
	off += 32
	// fee (uint24)
	data[off+29] = byte(fee >> 16)
	data[off+30] = byte(fee >> 8)
	data[off+31] = byte(fee)
	off += 32
	// tickSpacing (int24 sign-extended)
	if tickSpacing < 0 {
		for i := off; i < off+29; i++ {
			data[i] = 0xff
		}
	}
	data[off+29] = byte(tickSpacing >> 16)
	data[off+30] = byte(tickSpacing >> 8)
	data[off+31] = byte(tickSpacing)
	off += 32
	// hooks (address)
	if hooks != "" {
		copy(data[off+12:off+32], addrBytes(hooks))
	}
	off += 32 // off = 164

	// SwapParams (3 x 32-byte slots)
	if zeroForOne {
		data[off+31] = 1
	}
	off += 32
	// amountSpecified (int256 two's complement)
	copy(data[off:off+32], intTo256(amountSpecified))
	off += 32
	// sqrtPriceLimitX96
	lb := sqrtPriceLimitX96.Bytes()
	copy(data[off+32-len(lb):off+32], lb)
	off += 32 // off = 260

	// hookData: offset word + length word (both 0-padded = empty bytes)
	// offset = 256 bytes from start of args (points to length at position 288)
	big.NewInt(256).FillBytes(data[off : off+32])
	// length = 0 already zero

	return data
}

// addrBytes converts hex address to 20 bytes
func addrBytes(addr string) []byte {
	addr = strings.TrimPrefix(addr, "0x")
	if len(addr) < 40 {
		addr = strings.Repeat("0", 40-len(addr)) + addr
	}
	b, _ := hex.DecodeString(addr[:40])
	return b
}

// intTo256 encodes a big.Int as 32-byte two's complement
func intTo256(v *big.Int) []byte {
	r := make([]byte, 32)
	if v.Sign() >= 0 {
		b := v.Bytes()
		copy(r[32-len(b):], b)
	} else {
		p := new(big.Int).Add(new(big.Int).Lsh(big.NewInt(1), 256), v)
		b := p.Bytes()
		copy(r[32-len(b):], b)
	}
	return r
}

// feeToTS maps V4 fee tier to tick spacing
var feeToTS = map[uint32]int32{100: 1, 500: 10, 3000: 60, 10000: 200}

func getTS(fee uint32) int32 {
	if ts, ok := feeToTS[fee]; ok {
		return ts
	}
	return 60
}

func pickFee(req swapRequest, q *SwapQuote) uint32 {
	if req.Fee > 0 {
		return req.Fee
	}
	if len(q.Route) > 0 && q.Route[0].Fee > 0 {
		return uint32(q.Route[0].Fee)
	}
	return 3000
}

func zeroForOne(tokenIn, tokenOut string) bool {
	return strings.ToLower(tokenIn) < strings.ToLower(tokenOut)
}

func sortPair(a, b string) (string, string) {
	if strings.ToLower(a) < strings.ToLower(b) {
		return a, b
	}
	return b, a
}

// buildSwapFromQuote constructs an unsigned swap tx from a quote
func buildSwapFromQuote(req swapRequest, quote *SwapQuote) *swapResponse {
	fee := pickFee(req, quote)
	ts := getTS(fee)
	zfo := zeroForOne(req.TokenIn, req.TokenOut)
	c0, c1 := sortPair(req.TokenIn, req.TokenOut)

	amount := parseBigIntStr(req.Amount)
	amtSpec := new(big.Int).Set(amount)
	if req.IsExactIn {
		amtSpec.Neg(amtSpec) // V4: negative = exact input
	}

	var limit *big.Int
	if zfo {
		limit = new(big.Int).Set(MinSqrtRatio)
	} else {
		limit = new(big.Int).Set(MaxSqrtRatio)
	}

	calldata := buildSwapCalldata(c0, c1, fee, ts, req.Hooks, zfo, amtSpec, limit)

	gas := uint64(150000)
	if quote.GasEstimate != nil && quote.GasEstimate.Sign() > 0 {
		gas = quote.GasEstimate.Uint64()
	}

	return &swapResponse{
		Swap: swapTx{
			To:       LXPoolPrecompile,
			Data:     "0x" + hex.EncodeToString(calldata),
			Value:    "0x0",
			ChainID:  req.ChainID,
			GasLimit: gas,
		},
		Quote: *quote,
	}
}
