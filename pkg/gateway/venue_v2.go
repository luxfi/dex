package gateway

import (
	"context"
	"encoding/hex"
	"fmt"
	"math/big"
)

// Standard Uniswap V2 Router02 function selectors.
var (
	// getAmountsOut(uint256,address[])
	selectorV2GetAmountsOut = lxrMustDecodeHex("d06ca61f")

	// swapExactTokensForTokens(uint256,uint256,address[],address,uint256)
	selectorV2SwapExact = lxrMustDecodeHex("38ed1739")
)

// UniswapV2Venue quotes from any Uniswap V2-compatible Router02 contract.
// Works with Uniswap V2, SushiSwap, PancakeSwap, TraderJoe, and any V2 fork.
type UniswapV2Venue struct {
	evm           *EVMClient
	routerAddress string
	venueName     string
}

// UniswapV2Config configures a V2 venue.
type UniswapV2Config struct {
	// RPCURL is the EVM RPC endpoint.
	RPCURL string
	// RouterAddress is the V2 Router02 contract (0x-prefixed).
	RouterAddress string
	// Name identifies this venue (e.g. "uniswap_v2", "sushiswap", "traderjoe").
	Name string
}

// NewUniswapV2Venue creates a venue that quotes via a V2 Router02.
func NewUniswapV2Venue(cfg UniswapV2Config) *UniswapV2Venue {
	name := cfg.Name
	if name == "" {
		name = "uniswap_v2"
	}
	return &UniswapV2Venue{
		evm:           NewEVMClient(cfg.RPCURL),
		routerAddress: cfg.RouterAddress,
		venueName:     name,
	}
}

func (v *UniswapV2Venue) Name() string      { return v.venueName }
func (v *UniswapV2Venue) IsExecutable() bool { return true }

func (v *UniswapV2Venue) Quote(ctx context.Context, req VenueQuoteRequest) (*VenueQuote, error) {
	if req.Type != VenueQuoteTypeExactInput {
		return nil, nil // V2 getAmountsOut only supports exact input
	}

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

	// ABI encode: getAmountsOut(uint256 amountIn, address[] path)
	// Layout: selector(4) + amountIn(32) + offset_to_path(32) + path_length(32) + path[0](32) + path[1](32)
	calldata := make([]byte, 0, 4+32*5)
	calldata = append(calldata, selectorV2GetAmountsOut...)
	calldata = append(calldata, lxrPadUint256(amountIn)...)
	calldata = append(calldata, lxrPadUint256(big.NewInt(64))...) // offset to dynamic array
	calldata = append(calldata, lxrPadUint256(big.NewInt(2))...)  // path length = 2
	calldata = append(calldata, lxrPadAddress(inAddr)...)
	calldata = append(calldata, lxrPadAddress(outAddr)...)

	result, err := v.evm.CallContract(ctx, v.routerAddress, calldata)
	if err != nil {
		return nil, nil // no pool for this pair
	}

	// Response: offset(32) + length(32) + amounts[0](32) + amounts[1](32)
	if len(result) < 128 {
		return nil, nil
	}

	amountOut := new(big.Int).SetBytes(result[96:128]) // amounts[1]
	if amountOut.Sign() <= 0 {
		return nil, nil
	}

	return &VenueQuote{
		Venue:       v.venueName,
		AmountOut:   amountOut.String(),
		Fee:         "30", // standard V2 fee: 0.30%
		GasEstimate: "120000",
		Executable:  true,
	}, nil
}

// BuildV2SwapCalldata builds calldata for Router02.swapExactTokensForTokens.
//
// Parameters:
//   - routerAddress: the V2 Router02 contract address
//   - path: token addresses in swap order (at least 2)
//   - amountIn: exact input amount
//   - amountOutMin: minimum output (slippage protection)
//   - recipient: address to receive output tokens
//   - deadline: unix timestamp deadline (0 for block.timestamp)
//
// Returns 0x-prefixed calldata.
func BuildV2SwapCalldata(routerAddress string, path []string, amountIn, amountOutMin *big.Int, recipient string, deadline *big.Int) (string, error) {
	if len(path) < 2 {
		return "", fmt.Errorf("path must have at least 2 tokens")
	}

	recipientAddr, err := lxrDecodeAddress(recipient)
	if err != nil {
		return "", fmt.Errorf("recipient: %w", err)
	}

	if deadline == nil || deadline.Sign() == 0 {
		deadline = big.NewInt(0xFFFFFFFF) // far future
	}

	// ABI: selector(4) + amountIn(32) + amountOutMin(32) + offset(32) + recipient(32) + deadline(32)
	//      + path_length(32) + path[i](32)...
	numWords := 5 + 1 + len(path) // 5 fixed params + length + path entries
	calldata := make([]byte, 0, 4+numWords*32)
	calldata = append(calldata, selectorV2SwapExact...)
	calldata = append(calldata, lxrPadUint256(amountIn)...)
	calldata = append(calldata, lxrPadUint256(amountOutMin)...)
	calldata = append(calldata, lxrPadUint256(big.NewInt(160))...)              // offset to path (5 * 32 = 160)
	calldata = append(calldata, lxrPadAddress(recipientAddr)...)
	calldata = append(calldata, lxrPadUint256(deadline)...)
	calldata = append(calldata, lxrPadUint256(big.NewInt(int64(len(path))))...) // path length

	for i, addr := range path {
		a, err := lxrDecodeAddress(addr)
		if err != nil {
			return "", fmt.Errorf("path[%d]: %w", i, err)
		}
		calldata = append(calldata, lxrPadAddress(a)...)
	}

	return "0x" + hex.EncodeToString(calldata), nil
}

func init() {
	if len(selectorV2GetAmountsOut) != 4 {
		panic("selectorV2GetAmountsOut must be 4 bytes")
	}
	if len(selectorV2SwapExact) != 4 {
		panic("selectorV2SwapExact must be 4 bytes")
	}
}
