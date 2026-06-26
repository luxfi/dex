package gateway

import (
	"context"
	"encoding/hex"
	"fmt"
	"math/big"
)

// Lux DEX precompile addresses.
const (
	// NativePoolManager is the V4 AMM singleton (pool state, swap math).
	NativePoolManager = "0x0000000000000000000000000000000000009010"
	// NativeSwapRouter is the V4 swap router (ExactInput/Output).
	NativeSwapRouter = "0x0000000000000000000000000000000000009012"
	// NativeOrderBook is the central limit order book precompile.
	NativeOrderBook = "0x0000000000000000000000000000000000009020"
)

// quoteExactInputSingle(address,address,uint256,uint24)
// Selector: first 4 bytes of keccak256("quoteExactInputSingle(address,address,uint256,uint24)")
var selectorNativeQuote = lxrMustDecodeHex("30d07f21")

// NativeDEXVenue quotes from the Lux V4 DEX precompiles via eth_call.
// It calls the PoolManager (0x9010) for AMM quotes and falls back to
// the DEX (0x9020) for limit-order liquidity.
type NativeDEXVenue struct {
	evm    *EVMClient
	name   string
	feeBPS uint32 // default fee tier for pool lookup
	useDEX bool   // also query DEX for better prices
}

// NativeDEXConfig configures a NativeDEXVenue.
type NativeDEXConfig struct {
	// RPCURL is the Lux EVM RPC endpoint (e.g. https://rpc.dev.lux.network/rpc).
	RPCURL string
	// FeeBPS is the default fee tier in basis points for pool discovery.
	// Common values: 1 (0.01%), 5 (0.05%), 30 (0.30%), 100 (1%).
	FeeBPS uint32
	// UseDEX enables querying the DEX precompile (0x9020) alongside AMM.
	UseDEX bool
}

// NewNativeDEXVenue creates a venue that queries Lux DEX precompiles.
func NewNativeDEXVenue(cfg NativeDEXConfig) *NativeDEXVenue {
	feeBPS := cfg.FeeBPS
	if feeBPS == 0 {
		feeBPS = 30 // 0.30% default
	}
	return &NativeDEXVenue{
		evm:    NewEVMClient(cfg.RPCURL),
		name:   "v4_native",
		feeBPS: feeBPS,
		useDEX: cfg.UseDEX,
	}
}

func (v *NativeDEXVenue) Name() string       { return v.name }
func (v *NativeDEXVenue) IsExecutable() bool { return true }

func (v *NativeDEXVenue) Quote(ctx context.Context, req VenueQuoteRequest) (*VenueQuote, error) {
	// Build quoteExactInputSingle calldata:
	// selector(4) + tokenIn(32) + tokenOut(32) + amountIn(32) + fee(32) = 132 bytes
	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || amountIn.Sign() <= 0 {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}

	inAddr, err := lxrDecodeAddress(req.TokenIn)
	if err != nil {
		return nil, fmt.Errorf("tokenIn: %w", err)
	}
	outAddr, err := lxrDecodeAddress(req.TokenOut)
	if err != nil {
		return nil, fmt.Errorf("tokenOut: %w", err)
	}

	calldata := make([]byte, 0, 132)
	calldata = append(calldata, selectorNativeQuote...)
	calldata = append(calldata, lxrPadAddress(inAddr)...)
	calldata = append(calldata, lxrPadAddress(outAddr)...)
	calldata = append(calldata, lxrPadUint256(amountIn)...)
	calldata = append(calldata, lxrPadUint256(big.NewInt(int64(v.feeBPS)))...)

	result, err := v.evm.CallContract(ctx, NativePoolManager, calldata)
	if err != nil {
		// Pool may not exist for this pair — not an error, just no liquidity.
		return nil, nil
	}

	if len(result) < 32 {
		return nil, nil
	}

	amountOut := new(big.Int).SetBytes(result[:32])
	if amountOut.Sign() <= 0 {
		return nil, nil
	}

	quote := &VenueQuote{
		Venue:       v.name,
		AmountOut:   amountOut.String(),
		Fee:         fmt.Sprintf("%d", v.feeBPS),
		GasEstimate: "150000",
		Executable:  true,
	}

	// If DEX enabled, also check order book for better price.
	if v.useDEX {
		dexQuote, err := v.queryDEX(ctx, req, amountIn)
		if err == nil && dexQuote != nil {
			dexOut, _ := new(big.Int).SetString(dexQuote.AmountOut, 10)
			if dexOut != nil && dexOut.Cmp(amountOut) > 0 {
				return dexQuote, nil // DEX has better price
			}
		}
	}

	return quote, nil
}

// queryDEX checks the DEX precompile for limit-order liquidity.
// getBestQuote(address,address,uint256,bool)
var selectorDEXQuote = lxrMustDecodeHex("a2e62045")

func (v *NativeDEXVenue) queryDEX(ctx context.Context, req VenueQuoteRequest, amountIn *big.Int) (*VenueQuote, error) {
	inAddr, _ := lxrDecodeAddress(req.TokenIn)
	outAddr, _ := lxrDecodeAddress(req.TokenOut)

	isBuy := byte(1) // buying tokenOut
	var buyWord [32]byte
	buyWord[31] = isBuy

	calldata := make([]byte, 0, 132)
	calldata = append(calldata, selectorDEXQuote...)
	calldata = append(calldata, lxrPadAddress(inAddr)...)
	calldata = append(calldata, lxrPadAddress(outAddr)...)
	calldata = append(calldata, lxrPadUint256(amountIn)...)
	calldata = append(calldata, buyWord[:]...)

	result, err := v.evm.CallContract(ctx, NativeOrderBook, calldata)
	if err != nil {
		return nil, err
	}

	if len(result) < 32 {
		return nil, nil
	}

	amountOut := new(big.Int).SetBytes(result[:32])
	if amountOut.Sign() <= 0 {
		return nil, nil
	}

	return &VenueQuote{
		Venue:       v.name + "_dex",
		AmountOut:   amountOut.String(),
		Fee:         "0", // DEX maker fees may differ
		GasEstimate: "200000",
		Executable:  true,
	}, nil
}

// NativeSwapHop describes a single step in a native swap route.
type NativeSwapHop struct {
	TokenIn  string
	TokenOut string
	PoolID   string
}

// BuildNativeSwapCalldata builds calldata for the V4 SwapRouter (0x9012).
// This delegates to the existing V4 calldata builders.
func BuildNativeSwapCalldata(route []NativeSwapHop, amountIn, amountOut *big.Int) (string, string, error) {
	if len(route) == 1 {
		hop := route[0]
		calldata, err := BuildExactInputSingleCalldata(
			hop.TokenIn, hop.TokenOut, amountIn, amountOut,
			big.NewInt(0), hop.PoolID,
		)
		return NativeSwapRouter, calldata, err
	}

	path := make([]string, 0, len(route)+1)
	path = append(path, route[0].TokenIn)
	for _, hop := range route {
		path = append(path, hop.TokenOut)
	}
	calldata, err := BuildExactInputCalldata(path, amountIn, amountOut, big.NewInt(0))
	return NativeSwapRouter, calldata, err
}

// init-time validation.
func init() {
	if len(selectorNativeQuote) != 4 {
		panic("selectorNativeQuote must be 4 bytes")
	}
	if len(selectorDEXQuote) != 4 {
		panic("selectorDEXQuote must be 4 bytes")
	}
	// Validate precompile addresses are valid hex.
	for _, addr := range []string{NativePoolManager, NativeSwapRouter, NativeOrderBook} {
		if _, err := hex.DecodeString(addr[2:]); err != nil {
			panic("invalid precompile address: " + addr)
		}
	}
}
