package gateway

// The shapes POST /v1/trade/swap speaks.
//
// What used to live here — a calldata builder aimed at the PoolManager
// precompile at 0x9010, whichever venue had actually priced the trade — is
// gone. Each venue builds its own transaction now (venue_swap.go), because the
// venue that read the price is the only thing that knows what executes it.

// A swap is a quote plus where the proceeds go, so it is that request plus two
// fields rather than a second copy of the same six. Embedding flattens in JSON,
// so the shape on the wire is unchanged — and the two cannot drift about what
// slippage or isExactIn mean.
type swapRequest struct {
	quoteRequest
	Recipient string `json:"recipient"`
	Deadline  int64  `json:"deadline,omitempty"`
}

type swapResponse struct {
	Swap  UnsignedTxResponse `json:"swap"`
	Quote SwapQuote          `json:"quote"`
}
