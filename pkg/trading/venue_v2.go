package trading

import (
	"context"
	"encoding/hex"
	"fmt"
	"math/big"
)

// Uniswap V2 Router02 selectors.
var (
	// getAmountsOut(uint256,address[])
	selectorV2GetAmountsOut = mustDecodeHex("d06ca61f")
	// swapExactTokensForTokens(uint256,uint256,address[],address,uint256)
	selectorV2SwapExact = mustDecodeHex("38ed1739")
)

// UniswapV2Venue quotes from any V2-compatible Router02.
// Works with Uniswap V2, SushiSwap, PancakeSwap V2, TraderJoe, etc.
type UniswapV2Venue struct {
	evm           *EVMClient
	routerAddress string
	venueName     string
}

type UniswapV2Config struct {
	RPCURL        string
	RouterAddress string
	Name          string
}

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

func (v *UniswapV2Venue) Quote(ctx context.Context, req QuoteRequest) (*VenueQuote, error) {
	if req.Type != QuoteTypeExactInput {
		return nil, nil
	}

	amountIn, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || amountIn.Sign() <= 0 {
		return nil, fmt.Errorf("invalid amount: %s", req.Amount)
	}

	inAddr, err := decodeAddress(req.TokenIn)
	if err != nil {
		return nil, fmt.Errorf("tokenIn: %w", err)
	}
	outAddr, err := decodeAddress(req.TokenOut)
	if err != nil {
		return nil, fmt.Errorf("tokenOut: %w", err)
	}

	// ABI: getAmountsOut(uint256 amountIn, address[] path)
	calldata := make([]byte, 0, 4+32*5)
	calldata = append(calldata, selectorV2GetAmountsOut...)
	calldata = append(calldata, padUint256(amountIn)...)
	calldata = append(calldata, padUint256(big.NewInt(64))...) // offset to array
	calldata = append(calldata, padUint256(big.NewInt(2))...)  // path length
	calldata = append(calldata, padAddress(inAddr)...)
	calldata = append(calldata, padAddress(outAddr)...)

	result, err := v.evm.CallContract(ctx, v.routerAddress, calldata)
	if err != nil {
		return nil, nil
	}

	// Response: offset(32) + length(32) + amounts[0](32) + amounts[1](32)
	if len(result) < 128 {
		return nil, nil
	}

	amountOut := new(big.Int).SetBytes(result[96:128])
	if amountOut.Sign() <= 0 {
		return nil, nil
	}

	return &VenueQuote{
		Venue:       v.venueName,
		AmountOut:   amountOut.String(),
		Fee:         "30",
		GasEstimate: "120000",
		Executable:  true,
	}, nil
}

// BuildV2SwapCalldata builds Router02.swapExactTokensForTokens calldata.
func BuildV2SwapCalldata(routerAddress string, path []string, amountIn, amountOutMin *big.Int, recipient string, deadline *big.Int) (string, error) {
	if len(path) < 2 {
		return "", fmt.Errorf("path must have at least 2 tokens")
	}

	recipientAddr, err := decodeAddress(recipient)
	if err != nil {
		return "", fmt.Errorf("recipient: %w", err)
	}

	if deadline == nil || deadline.Sign() == 0 {
		deadline = big.NewInt(0xFFFFFFFF)
	}

	numWords := 5 + 1 + len(path)
	calldata := make([]byte, 0, 4+numWords*32)
	calldata = append(calldata, selectorV2SwapExact...)
	calldata = append(calldata, padUint256(amountIn)...)
	calldata = append(calldata, padUint256(amountOutMin)...)
	calldata = append(calldata, padUint256(big.NewInt(160))...)
	calldata = append(calldata, padAddress(recipientAddr)...)
	calldata = append(calldata, padUint256(deadline)...)
	calldata = append(calldata, padUint256(big.NewInt(int64(len(path))))...)

	for i, addr := range path {
		a, err := decodeAddress(addr)
		if err != nil {
			return "", fmt.Errorf("path[%d]: %w", i, err)
		}
		calldata = append(calldata, padAddress(a)...)
	}

	return "0x" + hex.EncodeToString(calldata), nil
}
