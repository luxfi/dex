package gateway

import (
	"context"
	"math/big"
	"testing"
)

// Test addresses.
const (
	testLUSD  = "0xc65ea8882020Af7CDa7854d590C6Fcd34BF364ec"
	testWBTC  = "0x1234567890abcdef1234567890abcdef12345678"
	testWETH  = "0xabcdef1234567890abcdef1234567890abcdef12"
	testOther = "0x0000000000000000000000000000000000000001"
)

func e18() *big.Int {
	return new(big.Int).Exp(big.NewInt(10), big.NewInt(18), nil)
}

// newTestVenues creates preconfigured mock venues:
//   - V4 pool: LUSD/WBTC at ~60000 LUSD per WBTC (30 bps fee)
//   - V4 pool: LUSD/WETH at ~3000 LUSD per WETH (30 bps fee)
//   - V4 pool: WBTC/WETH at ~20 WETH per WBTC (5 bps fee)
//   - Alpaca broker: LUSD/WBTC at ~59800 (slightly worse than V4)
//   - Perps broker: LUSD/WBTC at ~59500 (worst of three)
func newTestVenues() (*V4Venue, *BrokerVenue, *BrokerVenue) {
	unit := e18()

	v4 := NewV4Venue("")
	// LUSD/WBTC pool: 10M LUSD / 166 WBTC ≈ 60240 LUSD per WBTC
	usdlReserve := new(big.Int).Mul(big.NewInt(10_000_000), unit)
	wbtcReserve := new(big.Int).Mul(big.NewInt(166), unit)
	v4.AddPool(testLUSD, testWBTC,
		"0x0000000000000000000000000000000000000000000000000000000000000001",
		30, usdlReserve, wbtcReserve)

	// LUSD/WETH pool: 5M LUSD / 1666 WETH ≈ 3000 LUSD per WETH
	wethReserve := new(big.Int).Mul(big.NewInt(1666), unit)
	usdlReserve2 := new(big.Int).Mul(big.NewInt(5_000_000), unit)
	v4.AddPool(testLUSD, testWETH,
		"0x0000000000000000000000000000000000000000000000000000000000000002",
		30, usdlReserve2, wethReserve)

	// WBTC/WETH pool: 100 WBTC / 2000 WETH ≈ 20 WETH per WBTC
	wbtcReserve2 := new(big.Int).Mul(big.NewInt(100), unit)
	wethReserve2 := new(big.Int).Mul(big.NewInt(2000), unit)
	v4.AddPool(testWBTC, testWETH,
		"0x0000000000000000000000000000000000000000000000000000000000000003",
		5, wbtcReserve2, wethReserve2)

	// Alpaca broker: mock price of ~59800 LUSD per WBTC
	alpaca := NewBrokerVenue("http://broker.lux.svc.cluster.local:8090", "alpaca")
	alpacaPrice := new(big.Int).Mul(big.NewInt(59800), unit)
	alpaca.SetMockPrice(alpacaPrice)
	alpaca.SetFees(10, 25)

	// Perps broker: mock price of ~59500 LUSD per WBTC
	perps := NewBrokerVenue("http://broker.lux.svc.cluster.local:8090", "perps")
	perpsPrice := new(big.Int).Mul(big.NewInt(59500), unit)
	perps.SetMockPrice(perpsPrice)
	perps.SetFees(15, 20)

	return v4, alpaca, perps
}

func TestV4VenueExactInput(t *testing.T) {
	v4, _, _ := newTestVenues()
	ctx := context.Background()
	unit := e18()

	amount := new(big.Int).Mul(big.NewInt(1000), unit)
	q, err := v4.Quote(ctx, VenueQuoteRequest{
		TokenIn:  testLUSD,
		TokenOut: testWBTC,
		Amount:   amount.String(),
		Type:     VenueQuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("quote error: %v", err)
	}
	if q == nil {
		t.Fatal("expected non-nil quote")
	}

	out, ok := new(big.Int).SetString(q.AmountOut, 10)
	if !ok || out.Sign() <= 0 {
		t.Fatalf("expected positive amountOut, got %s", q.AmountOut)
	}
	if q.Venue != "v4_native" {
		t.Errorf("venue = %s, want v4_native", q.Venue)
	}
	if !q.Executable {
		t.Error("expected executable = true")
	}
}

func TestV4VenueExactOutput(t *testing.T) {
	v4, _, _ := newTestVenues()
	ctx := context.Background()
	unit := e18()

	wantOut := new(big.Int).Div(unit, big.NewInt(100)) // 0.01 WBTC
	q, err := v4.Quote(ctx, VenueQuoteRequest{
		TokenIn:  testLUSD,
		TokenOut: testWBTC,
		Amount:   wantOut.String(),
		Type:     VenueQuoteTypeExactOutput,
	})
	if err != nil {
		t.Fatalf("quote error: %v", err)
	}
	if q == nil {
		t.Fatal("expected non-nil quote")
	}
	out, _ := new(big.Int).SetString(q.AmountOut, 10)
	if out == nil || out.Sign() <= 0 {
		t.Fatalf("expected positive amountOut, got %s", q.AmountOut)
	}
}

func TestV4VenueNoLiquidity(t *testing.T) {
	v4 := NewV4Venue("")
	ctx := context.Background()

	q, err := v4.Quote(ctx, VenueQuoteRequest{
		TokenIn:  testOther,
		TokenOut: testWBTC,
		Amount:   "1000000000000000000",
		Type:     VenueQuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if q != nil {
		t.Fatalf("expected nil quote for no liquidity, got %+v", q)
	}
}

func TestBrokerVenueQuote(t *testing.T) {
	_, alpaca, _ := newTestVenues()
	ctx := context.Background()
	unit := e18()

	amount := new(big.Int).Mul(big.NewInt(1000), unit)
	q, err := alpaca.Quote(ctx, VenueQuoteRequest{
		TokenIn:  testLUSD,
		TokenOut: testWBTC,
		Amount:   amount.String(),
		Type:     VenueQuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("quote error: %v", err)
	}
	if q == nil {
		t.Fatal("expected non-nil quote")
	}
	if q.Executable {
		t.Error("broker venue should not be executable")
	}
	if q.Venue != "alpaca" {
		t.Errorf("venue = %s, want alpaca", q.Venue)
	}
}

func TestBrokerVenueNoPrice(t *testing.T) {
	b := NewBrokerVenue("http://localhost", "test")
	ctx := context.Background()

	q, err := b.Quote(ctx, VenueQuoteRequest{
		TokenIn:  testLUSD,
		TokenOut: testWBTC,
		Amount:   "1000000000000000000",
		Type:     VenueQuoteTypeExactInput,
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if q != nil {
		t.Fatalf("expected nil quote for no price, got %+v", q)
	}
}

// --- VenueRouter tests ---

func TestVenueRouterQueryAllVenues(t *testing.T) {
	v4, alpaca, perps := newTestVenues()
	vr := NewVenueRouter(v4, alpaca, perps)
	ctx := context.Background()
	unit := e18()

	amount := new(big.Int).Mul(big.NewInt(1000), unit)
	quotes, routing := vr.QueryAllVenues(ctx, VenueQuoteRequest{
		TokenIn:  testLUSD,
		TokenOut: testWBTC,
		Amount:   amount.String(),
		Type:     VenueQuoteTypeExactInput,
	})

	if len(quotes) != 3 {
		t.Fatalf("expected 3 venue quotes, got %d", len(quotes))
	}

	if routing != VenueRoutingV4Native {
		t.Errorf("expected routing %s, got %s", VenueRoutingV4Native, routing)
	}

	// Verify sorted by amountOut descending.
	for i := 1; i < len(quotes); i++ {
		prev, _ := new(big.Int).SetString(quotes[i-1].AmountOut, 10)
		curr, _ := new(big.Int).SetString(quotes[i].AmountOut, 10)
		if prev == nil || curr == nil {
			continue
		}
		if prev.Cmp(curr) < 0 {
			t.Errorf("venues not sorted: [%d]=%s > [%d]=%s",
				i-1, quotes[i-1].AmountOut, i, quotes[i].AmountOut)
		}
	}

	for _, v := range quotes {
		switch v.Venue {
		case "v4_native":
			if !v.Executable {
				t.Error("v4_native should be executable")
			}
		case "alpaca", "perps":
			if v.Executable {
				t.Errorf("%s should not be executable", v.Venue)
			}
		}
	}
}

func TestVenueRouterPreferredVenue(t *testing.T) {
	v4, alpaca, perps := newTestVenues()
	vr := NewVenueRouter(v4, alpaca, perps)
	ctx := context.Background()
	unit := e18()

	amount := new(big.Int).Mul(big.NewInt(1000), unit)
	quotes, _ := vr.QueryAllVenues(ctx, VenueQuoteRequest{
		TokenIn:        testLUSD,
		TokenOut:       testWBTC,
		Amount:         amount.String(),
		Type:           VenueQuoteTypeExactInput,
		PreferredVenue: "alpaca",
	})

	if len(quotes) != 1 {
		t.Fatalf("expected 1 venue (preferred), got %d", len(quotes))
	}
	if quotes[0].Venue != "alpaca" {
		t.Errorf("expected alpaca venue, got %s", quotes[0].Venue)
	}
}

func TestVenueRouterNoLiquidity(t *testing.T) {
	v4 := NewV4Venue("") // no pools added
	vr := NewVenueRouter(v4)
	ctx := context.Background()

	quotes, _ := vr.QueryAllVenues(ctx, VenueQuoteRequest{
		TokenIn:  testOther,
		TokenOut: testWBTC,
		Amount:   "1000000000000000000",
		Type:     VenueQuoteTypeExactInput,
	})

	if len(quotes) != 0 {
		t.Fatalf("expected 0 quotes, got %d", len(quotes))
	}
}

func TestListVenueInfo(t *testing.T) {
	v4, alpaca, perps := newTestVenues()
	vr := NewVenueRouter(v4, alpaca, perps)

	infos := vr.ListVenueInfo()
	if len(infos) != 3 {
		t.Fatalf("expected 3 venues, got %d", len(infos))
	}

	names := map[string]bool{}
	for _, info := range infos {
		names[info.Name] = true
		if info.Status != "active" {
			t.Errorf("venue %s status = %s, want active", info.Name, info.Status)
		}
	}

	for _, name := range []string{"v4_native", "alpaca", "perps"} {
		if !names[name] {
			t.Errorf("missing venue: %s", name)
		}
	}
}

// --- Constant product math tests ---

func TestConstantProductQuote(t *testing.T) {
	unit := e18()

	reserveIn := new(big.Int).Mul(big.NewInt(1_000_000), unit)
	reserveOut := new(big.Int).Mul(big.NewInt(1_000_000), unit)
	amountIn := new(big.Int).Mul(big.NewInt(1000), unit)

	out := constantProductQuote(amountIn, reserveIn, reserveOut, 30)

	expected := new(big.Int).Mul(big.NewInt(996), unit)
	if out.Cmp(expected) < 0 {
		t.Errorf("output too low: got %s, expected at least %s", out, expected)
	}
	if out.Cmp(new(big.Int).Mul(big.NewInt(1000), unit)) >= 0 {
		t.Errorf("output should be less than input for same-value pool: got %s", out)
	}
}

func TestConstantProductInverse(t *testing.T) {
	unit := e18()

	reserveIn := new(big.Int).Mul(big.NewInt(1_000_000), unit)
	reserveOut := new(big.Int).Mul(big.NewInt(1_000_000), unit)

	amountOut := new(big.Int).Mul(big.NewInt(996), unit)
	amountIn := constantProductInverse(amountOut, reserveIn, reserveOut, 30)

	maxIn := new(big.Int).Mul(big.NewInt(1000), unit)
	if amountIn.Cmp(maxIn) > 0 {
		t.Errorf("amountIn too high: %s > %s", amountIn, maxIn)
	}
	if amountIn.Sign() <= 0 {
		t.Error("amountIn should be positive")
	}
}

func TestConstantProductInverseInsufficientLiquidity(t *testing.T) {
	unit := e18()
	reserveIn := new(big.Int).Mul(big.NewInt(1_000_000), unit)
	reserveOut := new(big.Int).Mul(big.NewInt(1_000_000), unit)

	amountOut := new(big.Int).Mul(big.NewInt(2_000_000), unit)
	result := constantProductInverse(amountOut, reserveIn, reserveOut, 30)

	if result.Sign() > 0 {
		t.Errorf("expected zero for insufficient liquidity, got %s", result)
	}
}

// --- Helper function tests ---

func TestComputePriceImpact(t *testing.T) {
	quotes := []VenueQuote{
		{AmountOut: "10000"},
		{AmountOut: "9500"},
	}
	impact := ComputePriceImpact(quotes)
	if impact == "0.00" {
		t.Errorf("expected non-zero price impact between 10000 and 9500, got %s", impact)
	}

	// Single quote should return 0.00
	single := ComputePriceImpact([]VenueQuote{{AmountOut: "10000"}})
	if single != "0.00" {
		t.Errorf("expected 0.00 for single quote, got %s", single)
	}
}

func TestComputeExecutionPrice(t *testing.T) {
	// 1e18 in, 2e18 out -> price = 2e18
	unit := e18()
	price := ComputeExecutionPrice(unit.String(), new(big.Int).Mul(big.NewInt(2), unit).String())
	expected := new(big.Int).Mul(big.NewInt(2), unit)
	if price != expected.String() {
		t.Errorf("execution price = %s, want %s", price, expected)
	}

	// Zero amount in
	if ComputeExecutionPrice("0", "100") != "0" {
		t.Error("expected 0 for zero amountIn")
	}
}

func TestParseBPS(t *testing.T) {
	tests := []struct {
		input string
		want  int
	}{
		{"30", 30},
		{"3000", 3000},
		{"0", 0},
		{"abc", 0},
		{"", 0},
	}
	for _, tt := range tests {
		if got := ParseBPS(tt.input); got != tt.want {
			t.Errorf("ParseBPS(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestValidateVenueQuoteRequest(t *testing.T) {
	valid := VenueQuoteRequest{
		TokenIn:  testLUSD,
		TokenOut: testWBTC,
		Amount:   "1000",
		Type:     VenueQuoteTypeExactInput,
	}
	if err := ValidateVenueQuoteRequest(valid); err != nil {
		t.Fatalf("valid request rejected: %v", err)
	}

	tests := []struct {
		name string
		req  VenueQuoteRequest
	}{
		{"invalid tokenIn", VenueQuoteRequest{
			TokenIn: "bad", TokenOut: testWBTC, Amount: "1000", Type: VenueQuoteTypeExactInput,
		}},
		{"same tokens", VenueQuoteRequest{
			TokenIn: testLUSD, TokenOut: testLUSD, Amount: "1000", Type: VenueQuoteTypeExactInput,
		}},
		{"zero amount", VenueQuoteRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "0", Type: VenueQuoteTypeExactInput,
		}},
		{"bad type", VenueQuoteRequest{
			TokenIn: testLUSD, TokenOut: testWBTC, Amount: "1000", Type: "INVALID",
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := ValidateVenueQuoteRequest(tt.req); err == nil {
				t.Error("expected error for invalid request")
			}
		})
	}
}

func TestIsValidHexAddr(t *testing.T) {
	tests := []struct {
		addr string
		want bool
	}{
		{testLUSD, true},
		{"0x0000000000000000000000000000000000000001", true},
		{"0xc65ea8882020af7cda7854d590c6fcd34bf364ec", true},
		{"not-an-address", false},
		{"0x123", false},
		{"", false},
		{"0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG", false},
	}
	for _, tt := range tests {
		if got := isValidHexAddr(tt.addr); got != tt.want {
			t.Errorf("isValidHexAddr(%q) = %v, want %v", tt.addr, got, tt.want)
		}
	}
}

func TestIsPositiveDecStr(t *testing.T) {
	tests := []struct {
		s    string
		want bool
	}{
		{"1000", true},
		{"1", true},
		{"0", false},
		{"", false},
		{"abc", false},
		{"-1", false},
		{"1.5", false},
	}
	for _, tt := range tests {
		if got := isPositiveDecStr(tt.s); got != tt.want {
			t.Errorf("isPositiveDecStr(%q) = %v, want %v", tt.s, got, tt.want)
		}
	}
}
