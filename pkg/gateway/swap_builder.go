package gateway

// The shapes POST /v1/trade/swap speaks.
//
// What used to live here — a calldata builder aimed at the PoolManager
// precompile at 0x9010, whichever venue had actually priced the trade — is
// gone. Each venue builds its own transaction now (venue_swap.go), because the
// venue that read the price is the only thing that knows what executes it.

type swapRequest struct {
	TokenIn   string `json:"tokenIn"`
	TokenOut  string `json:"tokenOut"`
	ChainID   uint64 `json:"chainId"`
	Amount    string `json:"amount"`
	IsExactIn bool   `json:"isExactIn"`
	// Slippage is a percentage. Zero means half a percent.
	Slippage  float64 `json:"slippage,omitempty"`
	Recipient string  `json:"recipient"`
	Deadline  int64   `json:"deadline,omitempty"`
}

type swapResponse struct {
	Swap  UnsignedTxResponse `json:"swap"`
	Quote SwapQuote          `json:"quote"`
}
