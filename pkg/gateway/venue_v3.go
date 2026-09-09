package gateway

import (
	"context"
	"fmt"
	"math/big"
)

// QuoterV2.quoteExactInputSingle((address,address,uint256,uint24,uint160))
//
// The struct is a single static tuple, so it encodes inline: five words after
// the selector, no offset. The older QuoterV1 took the same five values as
// flat arguments under a different selector; a chain still running V1 answers
// nothing to this one, which reads as a pair with no pool rather than a wrong
// answer.
var selectorV3QuoteExactInputSingle = lxrMustDecodeHex("c6a5026a")

// The fee tiers a V3 deployment publishes, in hundredths of a basis point.
// A quote asks each and keeps the best, because which tier holds the liquidity
// for a pair is a fact about that pair and not about the chain.
var v3FeeTiers = []uint32{100, 500, 3000, 10000}

// UniswapV3Venue quotes from any Uniswap V3-compatible QuoterV2.
//
// It exists because the gateway could read a V2 router and the native
// precompiles and nothing in between, so every V3 market — which on most
// chains is where the liquidity is — was invisible. Reading the quoter over
// JSON-RPC needs no API key and no upstream account: the pool is the source.
type UniswapV3Venue struct {
	evm       *EVMClient
	quoter    string
	router    string
	venueName string
	tiers     []uint32
}

// UniswapV3Config configures a V3 venue.
type UniswapV3Config struct {
	// RPCURL is the EVM RPC endpoint.
	RPCURL string
	// QuoterAddress is the QuoterV2 contract (0x-prefixed).
	QuoterAddress string
	// RouterAddress is the SwapRouter02 that executes what the quoter priced.
	// Empty means this venue quotes and does not build.
	RouterAddress string
	// Name identifies this venue (e.g. "uniswap_v3", "pancake_v3").
	Name string
	// FeeTiers overrides the tiers asked. Empty means the four V3 publishes.
	FeeTiers []uint32
}

// NewUniswapV3Venue creates a venue that quotes via a QuoterV2.
func NewUniswapV3Venue(cfg UniswapV3Config) *UniswapV3Venue {
	name := cfg.Name
	if name == "" {
		name = "uniswap_v3"
	}
	tiers := cfg.FeeTiers
	if len(tiers) == 0 {
		tiers = v3FeeTiers
	}
	return &UniswapV3Venue{
		evm:       NewEVMClient(cfg.RPCURL),
		quoter:    cfg.QuoterAddress,
		router:    cfg.RouterAddress,
		venueName: name,
		tiers:     tiers,
	}
}

func (v *UniswapV3Venue) Name() string       { return v.venueName }
func (v *UniswapV3Venue) IsExecutable() bool { return true }

func (v *UniswapV3Venue) Quote(ctx context.Context, req VenueQuoteRequest) (*VenueQuote, error) {
	if req.Type != VenueQuoteTypeExactInput {
		return nil, nil
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

	best := new(big.Int)
	var bestFee uint32
	for _, fee := range v.tiers {
		// (tokenIn, tokenOut, amountIn, fee, sqrtPriceLimitX96) — a static
		// tuple, so five words inline. The price limit is zero: no bound.
		calldata := make([]byte, 0, 4+32*5)
		calldata = append(calldata, selectorV3QuoteExactInputSingle...)
		calldata = append(calldata, lxrPadAddress(inAddr)...)
		calldata = append(calldata, lxrPadAddress(outAddr)...)
		calldata = append(calldata, lxrPadUint256(amountIn)...)
		calldata = append(calldata, lxrPadUint256(new(big.Int).SetUint64(uint64(fee)))...)
		calldata = append(calldata, lxrPadUint256(big.NewInt(0))...)

		result, err := v.evm.CallContract(ctx, v.quoter, calldata)
		if err != nil || len(result) < 32 {
			// A tier with no pool reverts. That is the ordinary answer to
			// three of the four tiers on most pairs, not a failure to report.
			continue
		}

		out := new(big.Int).SetBytes(result[0:32])
		if out.Cmp(best) > 0 {
			best = out
			bestFee = fee
		}
	}

	if best.Sign() <= 0 {
		return nil, nil
	}

	return &VenueQuote{
		Venue:     v.venueName,
		AmountOut: best.String(),
		// The tier the quote came from, in hundredths of a basis point, which
		// is the unit V3 states a fee in.
		Fee:         fmt.Sprintf("%d", bestFee),
		GasEstimate: "180000",
		Executable:  true,
	}, nil
}
