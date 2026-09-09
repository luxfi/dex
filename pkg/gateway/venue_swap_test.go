package gateway

import (
	"encoding/hex"
	"math/big"
	"strings"
	"testing"
	"time"
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

	raw := unwrap(t, tx.Data)
	if got, want := hex.EncodeToString(raw[:4]), "04e45aaf"; got != want {
		t.Errorf("selector = %s, want %s", got, want)
	}
	if len(raw) != 4+32*7 {
		t.Fatalf("the call is %d bytes, want %d — a seven-field tuple encodes inline", len(raw), 4+32*7)
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

// SwapRouter02 dropped the deadline from exactInputSingle's tuple and takes it
// through multicall(uint256,bytes[]) instead, which is how Uniswap's own
// interface sends one. Without the wrap a deadline handed to /v1/trade/swap was
// accepted and silently dropped on every V3 route while the V2 route honoured
// it — one field meaning two things depending on which pool won the quote.
//
// amountOutMinimum bounds the price; the deadline bounds the time. A signed
// swap that sits in a mempool and lands an hour later still passes its floor,
// at a price nobody would choose now.
func TestV3SwapCarriesTheDeadlineThroughMulticall(t *testing.T) {
	v := NewUniswapV3Venue(UniswapV3Config{
		RPCURL: "http://chain.invalid", QuoterAddress: testOther, RouterAddress: testOther,
	})
	tx, err := v.Swap(SwapOrder{
		TokenIn: testWETH, TokenOut: testLUSD,
		AmountIn: big.NewInt(1_000_000), MinOut: big.NewInt(995_000),
		Recipient: testOther, Fee: 3000, Deadline: 1893456000,
	})
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	raw, _ := hex.DecodeString(strings.TrimPrefix(tx.Data, "0x"))

	if got := hex.EncodeToString(raw[:4]); got != "5ae401dc" {
		t.Fatalf("selector = %s, want 5ae401dc — multicall(uint256,bytes[])", got)
	}
	word := func(i int) *big.Int { return new(big.Int).SetBytes(raw[4+32*i : 4+32*(i+1)]) }
	if word(0).Int64() != 1893456000 {
		t.Errorf("deadline = %s, want the one asked for", word(0))
	}
	if word(1).Int64() != 64 {
		t.Errorf("array offset = %s, want 64 — two head words", word(1))
	}
	if word(2).Int64() != 1 {
		t.Errorf("array length = %s, want 1", word(2))
	}
	// Offsets inside a dynamic array count from the start of the ARRAY, not the
	// start of the calldata. Counting from the wrong place is what makes a
	// hand-written encoding revert.
	if word(3).Int64() != 32 {
		t.Errorf("element offset = %s, want 32", word(3))
	}

	inner := word(4).Int64()
	if inner != 4+32*7 {
		t.Fatalf("inner length = %d, want %d", inner, 4+32*7)
	}
	body := raw[4+32*5:]
	if got := hex.EncodeToString(body[:4]); got != "04e45aaf" {
		t.Errorf("wrapped call = %s, want exactInputSingle", got)
	}
	// The blob is padded to a whole word, and the padding is not part of it.
	if len(body)%32 != 0 {
		t.Errorf("calldata is %d bytes past the head — dynamic data pads to a word", len(body)%32)
	}
	if int64(len(body)) < inner {
		t.Errorf("the wrapped call is %d bytes but the length says %d", len(body), inner)
	}

	// A deadline nobody asked for is twenty minutes out, not zero — zero is a
	// deadline in 1970 and every swap would revert as too old.
	tx, _ = v.Swap(SwapOrder{
		TokenIn: testWETH, TokenOut: testLUSD,
		AmountIn: big.NewInt(1), MinOut: big.NewInt(1), Recipient: testOther, Fee: 3000,
	})
	raw, _ = hex.DecodeString(strings.TrimPrefix(tx.Data, "0x"))
	if d := new(big.Int).SetBytes(raw[4 : 4+32]).Int64(); d < time.Now().Unix() {
		t.Errorf("unset deadline = %d, which is already past", d)
	}
}

// unwrap returns the one call inside SwapRouter02's multicall(deadline, calls).
func unwrap(t *testing.T, data string) []byte {
	t.Helper()
	raw, err := hex.DecodeString(strings.TrimPrefix(data, "0x"))
	if err != nil {
		t.Fatalf("calldata: %v", err)
	}
	if got := hex.EncodeToString(raw[:4]); got != "5ae401dc" {
		t.Fatalf("selector = %s, want the multicall the deadline rides in", got)
	}
	n := new(big.Int).SetBytes(raw[4+32*4 : 4+32*5]).Int64()
	return raw[4+32*5 : 4+32*5+int(n)]
}

// An amount is a whole number of a token's smallest unit that fits in one EVM
// word. Anything else is refused where it enters, because the encoder cannot
// refuse: it wrote a too-wide number as its HIGH 32 bytes, so a caller asking
// to swap one number got calldata for a completely unrelated one.
func TestAnAmountIsAWholeNumberThatFitsInAWord(t *testing.T) {
	tooWide := new(big.Int).Lsh(big.NewInt(1), 256).String()
	for _, c := range []struct{ amount, why string }{
		{"", "empty"},
		{"0", "zero"},
		{"-1", "negative"},
		{"1.5", "not whole"},
		{"1e18", "not decimal digits"},
		{"twelve", "not a number"},
		{tooWide, "one bit wider than a word"},
	} {
		if _, err := wholeUnits(c.amount); err == nil {
			t.Errorf("%s (%q) was accepted", c.why, c.amount)
		}
	}
	widest := new(big.Int).Sub(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(1))
	if got, err := wholeUnits(widest.String()); err != nil || got.Cmp(widest) != 0 {
		t.Errorf("the widest word was refused: %v %v", got, err)
	}

	// And the encoder keeps the LOW bytes, which is what the EVM does, rather
	// than the leading ones.
	over := new(big.Int).Add(new(big.Int).Lsh(big.NewInt(1), 256), big.NewInt(7))
	if got := new(big.Int).SetBytes(lxrPadUint256(over)); got.Int64() != 7 {
		t.Errorf("a value one word over encoded as %s, want its low bytes (7)", got)
	}
}
