package trading

import (
	"context"
	"encoding/hex"
	"fmt"
	"math/big"
)

// Uniswap V3 selectors.
var (
	// quoteExactInputSingle(address,address,uint24,uint256,uint160)
	selectorV3QuoteExactInputSingle = mustDecodeHex("f7729d43")
	// exactInputSingle((address,address,uint24,address,uint256,uint256,uint256,uint160))
	selectorV3ExactInputSingle = mustDecodeHex("414bf389")
)

// UniswapV3Venue quotes from any V3-compatible Quoter contract.
// Works with Uniswap V3, PancakeSwap V3, etc.
type UniswapV3Venue struct {
	evm           *EVMClient
	quoterAddress string
	routerAddress string
	venueName     string
	feeTier       uint32 // V3 fee: 100, 500, 3000, or 10000
}

type UniswapV3Config struct {
	RPCURL        string
	QuoterAddress string
	RouterAddress string
	Name          string
	FeeTier       uint32
}

func NewUniswapV3Venue(cfg UniswapV3Config) *UniswapV3Venue {
	name := cfg.Name
	if name == "" {
		name = "uniswap_v3"
	}
	feeTier := cfg.FeeTier
	if feeTier == 0 {
		feeTier = 3000
	}
	return &UniswapV3Venue{
		evm:           NewEVMClient(cfg.RPCURL),
		quoterAddress: cfg.QuoterAddress,
		routerAddress: cfg.RouterAddress,
		venueName:     name,
		feeTier:       feeTier,
	}
}

func (v *UniswapV3Venue) Name() string       { return v.venueName }
func (v *UniswapV3Venue) IsExecutable() bool { return true }

func (v *UniswapV3Venue) Quote(ctx context.Context, req QuoteRequest) (*VenueQuote, error) {
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

	calldata := make([]byte, 0, 4+32*5)
	calldata = append(calldata, selectorV3QuoteExactInputSingle...)
	calldata = append(calldata, padAddress(inAddr)...)
	calldata = append(calldata, padAddress(outAddr)...)
	calldata = append(calldata, padUint256(big.NewInt(int64(v.feeTier)))...)
	calldata = append(calldata, padUint256(amountIn)...)
	calldata = append(calldata, padUint256(big.NewInt(0))...)

	result, err := v.evm.CallContract(ctx, v.quoterAddress, calldata)
	if err != nil {
		return nil, nil
	}
	if len(result) < 32 {
		return nil, nil
	}

	amountOut := new(big.Int).SetBytes(result[:32])
	if amountOut.Sign() <= 0 {
		return nil, nil
	}

	feeBPS := v.feeTier / 100

	return &VenueQuote{
		Venue:       v.venueName,
		AmountOut:   amountOut.String(),
		Fee:         fmt.Sprintf("%d", feeBPS),
		GasEstimate: "180000",
		Executable:  true,
	}, nil
}

// BuildV3SwapCalldata builds V3 SwapRouter.exactInputSingle calldata.
func BuildV3SwapCalldata(routerAddress string, tokenIn, tokenOut string, fee uint32, recipient string, amountIn, amountOutMin *big.Int, deadline *big.Int) (string, error) {
	inAddr, err := decodeAddress(tokenIn)
	if err != nil {
		return "", fmt.Errorf("tokenIn: %w", err)
	}
	outAddr, err := decodeAddress(tokenOut)
	if err != nil {
		return "", fmt.Errorf("tokenOut: %w", err)
	}
	recipientAddr, err := decodeAddress(recipient)
	if err != nil {
		return "", fmt.Errorf("recipient: %w", err)
	}

	if deadline == nil || deadline.Sign() == 0 {
		deadline = big.NewInt(0xFFFFFFFF)
	}

	calldata := make([]byte, 0, 4+32*8)
	calldata = append(calldata, selectorV3ExactInputSingle...)
	calldata = append(calldata, padAddress(inAddr)...)
	calldata = append(calldata, padAddress(outAddr)...)
	calldata = append(calldata, padUint256(big.NewInt(int64(fee)))...)
	calldata = append(calldata, padAddress(recipientAddr)...)
	calldata = append(calldata, padUint256(deadline)...)
	calldata = append(calldata, padUint256(amountIn)...)
	calldata = append(calldata, padUint256(amountOutMin)...)
	calldata = append(calldata, padUint256(big.NewInt(0))...)

	return "0x" + hex.EncodeToString(calldata), nil
}
