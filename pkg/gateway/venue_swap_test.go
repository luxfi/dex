package gateway

import (
	"encoding/hex"
	"math/big"
	"strings"
	"testing"
)

// The calldata a wallet signs is the one thing here that spends money, so its
// layout is asserted field by field rather than compared to a blob.
//
// SwapRouter02.exactInputSingle takes a SEVEN-field static tuple, encoded
// inline. The repository carried this same selector, 04e45aaf, against a
// six-field signature it called ExactInputSingle(address,address,uint256,
// uint256,uint160,bytes32) — which hashes to f48f227e. A router handed the
// short form reads amountIn where the fee tier belongs and amountOutMinimum
// where the recipient belongs.
func TestV3SwapEncodesSwapRouter02(t *testing.T) {
	v := NewUniswapV3Venue(UniswapV3Config{
		RPCURL:        "http://chain.invalid",
		QuoterAddress: testOther,
		RouterAddress: "0xbbD8d9A1E6bf5627A2F6D2aF38664bbe1cEF63Bb",
	})

	tx, err := v.Swap(SwapOrder{
		TokenIn:   testWETH,
		TokenOut:  testLUSD,
		AmountIn:  big.NewInt(1_000_000),
		MinOut:    big.NewInt(995_000),
		Recipient: testOther,
		Fee:       3000,
	})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	if !strings.EqualFold(tx.To, "0xbbD8d9A1E6bf5627A2F6D2aF38664bbe1cEF63Bb") {
		t.Errorf("to = %s, want the router this venue was given", tx.To)
	}

	raw, err := hex.DecodeString(strings.TrimPrefix(tx.Data, "0x"))
	if err != nil {
		t.Fatalf("calldata: %v", err)
	}
	if got, want := hex.EncodeToString(raw[:4]), "04e45aaf"; got != want {
		t.Errorf("selector = %s, want %s", got, want)
	}
	if len(raw) != 4+32*7 {
		t.Fatalf("calldata is %d bytes, want %d — a seven-field tuple encodes inline", len(raw), 4+32*7)
	}

	word := func(i int) []byte { return raw[4+32*i : 4+32*(i+1)] }
	addr := func(i int) string { return "0x" + hex.EncodeToString(word(i)[12:]) }
	num := func(i int) *big.Int { return new(big.Int).SetBytes(word(i)) }

	for _, c := range []struct {
		field string
		got   string
		want  string
	}{
		{"tokenIn", addr(0), strings.ToLower(testWETH)},
		{"tokenOut", addr(1), strings.ToLower(testLUSD)},
		{"recipient", addr(3), strings.ToLower(testOther)},
	} {
		if c.got != c.want {
			t.Errorf("%s = %s, want %s", c.field, c.got, c.want)
		}
	}
	if num(2).Int64() != 3000 {
		t.Errorf("fee = %s, want 3000 — the tier the quote was read from", num(2))
	}
	if num(4).Int64() != 1_000_000 {
		t.Errorf("amountIn = %s, want 1000000", num(4))
	}
	if num(5).Int64() != 995_000 {
		t.Errorf("amountOutMinimum = %s, want 995000", num(5))
	}
	if num(6).Sign() != 0 {
		t.Errorf("sqrtPriceLimitX96 = %s, want 0 — the floor is the bound", num(6))
	}
}

// A V3 swap that does not name the tier its quote came from executes against a
// different pool at a different price, so it is refused rather than guessed.
func TestV3SwapRefusesToGuessTheTier(t *testing.T) {
	v := NewUniswapV3Venue(UniswapV3Config{RPCURL: "http://chain.invalid", QuoterAddress: testOther, RouterAddress: testOther})
	if _, err := v.Swap(SwapOrder{TokenIn: testWETH, TokenOut: testLUSD, AmountIn: big.NewInt(1), MinOut: big.NewInt(1), Recipient: testOther}); err == nil {
		t.Error("a swap with no tier was built anyway")
	}
}

// A venue with no router quotes and does not build, and says so by returning
// nothing rather than calldata for an address it does not have.
func TestAVenueWithNoRouterBuildsNothing(t *testing.T) {
	v := NewUniswapV3Venue(UniswapV3Config{RPCURL: "http://chain.invalid", QuoterAddress: testOther})
	tx, err := v.Swap(SwapOrder{TokenIn: testWETH, TokenOut: testLUSD, AmountIn: big.NewInt(1), MinOut: big.NewInt(1), Recipient: testOther, Fee: 3000})
	if err != nil || tx != nil {
		t.Errorf("got (%v, %v), want (nil, nil)", tx, err)
	}
}

// swapExactTokensForTokens's path is a dynamic array, so word three is an
// offset to it and the array follows the five head words.
func TestV2SwapEncodesTheDynamicPath(t *testing.T) {
	v := NewUniswapV2Venue(UniswapV2Config{RPCURL: "http://chain.invalid", RouterAddress: testOther})
	tx, err := v.Swap(SwapOrder{
		TokenIn: testWETH, TokenOut: testLUSD,
		AmountIn: big.NewInt(7), MinOut: big.NewInt(6),
		Recipient: testOther, Deadline: 1893456000,
	})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	raw, _ := hex.DecodeString(strings.TrimPrefix(tx.Data, "0x"))
	if got := hex.EncodeToString(raw[:4]); got != "38ed1739" {
		t.Errorf("selector = %s, want 38ed1739", got)
	}
	word := func(i int) *big.Int { return new(big.Int).SetBytes(raw[4+32*i : 4+32*(i+1)]) }
	if word(2).Int64() != 160 {
		t.Errorf("path offset = %s, want 160 — five head words", word(2))
	}
	if word(4).Int64() != 1893456000 {
		t.Errorf("deadline = %s, want the one asked for", word(4))
	}
	if word(5).Int64() != 2 {
		t.Errorf("path length = %s, want 2", word(5))
	}
	if got := "0x" + hex.EncodeToString(raw[4+32*6+12:4+32*7]); got != strings.ToLower(testWETH) {
		t.Errorf("path[0] = %s, want tokenIn", got)
	}
	if got := "0x" + hex.EncodeToString(raw[4+32*7+12:4+32*8]); got != strings.ToLower(testLUSD) {
		t.Errorf("path[1] = %s, want tokenOut", got)
	}
}

// A router handed the quoted amount as its floor reverts on any movement at
// all, including the movement the caller's own trade causes.
func TestTheFloorSitsBelowTheQuote(t *testing.T) {
	quoted := big.NewInt(1_000_000)
	for _, c := range []struct {
		tolerance float64
		want      int64
	}{
		{0.5, 995_000},
		{1, 990_000},
		{5, 950_000},
		{0, 995_000},   // unset falls to half a percent
		{-3, 995_000},  // and so does nonsense
		{100, 995_000}, // including a tolerance that would accept nothing back
		// Finer than a basis point. In whole basis points this truncated to
		// zero and the floor came back EQUAL to the quote, so a request for
		// very tight protection was answered with none at all and the swap
		// reverted on the first wei of movement.
		{0.005, 999_950},
		{0.01, 999_900},
	} {
		if got := leastAccepted(quoted, c.tolerance); got.Int64() != c.want {
			t.Errorf("tolerance %v: floor = %s, want %d", c.tolerance, got, c.want)
		}
	}
	if got := leastAccepted(nil, 1); got.Sign() != 0 {
		t.Errorf("no quote: floor = %s, want 0", got)
	}
	if leastAccepted(quoted, 0.5).Cmp(quoted) >= 0 {
		t.Error("the floor is not below the quote")
	}
}
