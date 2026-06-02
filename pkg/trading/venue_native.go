package trading

import (
	"context"
	"fmt"
	"math/big"
)

// NativeCLOB is the V4 CLOB precompile address.
const NativeCLOB = "0x0000000000000000000000000000000000009020"

// quoteExactInputSingle(address,address,uint256,uint24)
var selectorNativeQuote = mustDecodeHex("30d07f21")

// getBestQuote(address,address,uint256,bool)
var selectorCLOBQuote = mustDecodeHex("a2e62045")

// NativeDEXVenue quotes from the Lux V4 DEX precompiles via eth_call.
// Calls PoolManager (0x9010) for AMM quotes and optionally CLOB (0x9020).
type NativeDEXVenue struct {
	evm     *EVMClient
	name    string
	feeBPS  uint32
	useCLOB bool
}

// NativeDEXConfig configures a NativeDEXVenue.
type NativeDEXConfig struct {
	RPCURL  string
	FeeBPS  uint32
	UseCLOB bool
}

func NewNativeDEXVenue(cfg NativeDEXConfig) *NativeDEXVenue {
	feeBPS := cfg.FeeBPS
	if feeBPS == 0 {
		feeBPS = 30
	}
	return &NativeDEXVenue{
		evm:     NewEVMClient(cfg.RPCURL),
		name:    "v4_native",
		feeBPS:  feeBPS,
		useCLOB: cfg.UseCLOB,
	}
}

func (v *NativeDEXVenue) Name() string       { return v.name }
func (v *NativeDEXVenue) IsExecutable() bool { return true }

func (v *NativeDEXVenue) Quote(ctx context.Context, req QuoteRequest) (*VenueQuote, error) {
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

	calldata := make([]byte, 0, 132)
	calldata = append(calldata, selectorNativeQuote...)
	calldata = append(calldata, padAddress(inAddr)...)
	calldata = append(calldata, padAddress(outAddr)...)
	calldata = append(calldata, padUint256(amountIn)...)
	calldata = append(calldata, padUint256(big.NewInt(int64(v.feeBPS)))...)

	result, err := v.evm.CallContract(ctx, PoolManagerAddress, calldata)
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

	quote := &VenueQuote{
		Venue:       v.name,
		AmountOut:   amountOut.String(),
		Fee:         fmt.Sprintf("%d", v.feeBPS),
		GasEstimate: "150000",
		Executable:  true,
	}

	if v.useCLOB {
		clobQuote, err := v.queryCLOB(ctx, req, amountIn)
		if err == nil && clobQuote != nil {
			clobOut, _ := new(big.Int).SetString(clobQuote.AmountOut, 10)
			if clobOut != nil && clobOut.Cmp(amountOut) > 0 {
				return clobQuote, nil
			}
		}
	}

	return quote, nil
}

func (v *NativeDEXVenue) queryCLOB(ctx context.Context, req QuoteRequest, amountIn *big.Int) (*VenueQuote, error) {
	inAddr, _ := decodeAddress(req.TokenIn)
	outAddr, _ := decodeAddress(req.TokenOut)

	var buyWord [32]byte
	buyWord[31] = 1

	calldata := make([]byte, 0, 132)
	calldata = append(calldata, selectorCLOBQuote...)
	calldata = append(calldata, padAddress(inAddr)...)
	calldata = append(calldata, padAddress(outAddr)...)
	calldata = append(calldata, padUint256(amountIn)...)
	calldata = append(calldata, buyWord[:]...)

	result, err := v.evm.CallContract(ctx, NativeCLOB, calldata)
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
		Venue:       v.name + "_clob",
		AmountOut:   amountOut.String(),
		Fee:         "0",
		GasEstimate: "200000",
		Executable:  true,
	}, nil
}
