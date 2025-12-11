package price

import "strings"

// Normalize standardizes a trading symbol across different sources.
// Handles variations like:
//   - LUX/USDC, LUX-USDC, LUXUSDC -> LUX-USDC
//   - ETH/USD, ETH-USD, ETHUSD -> ETH-USD
//   - BTC/USDT, BTC-USDT, BTCUSDT -> BTC-USDT
func Normalize(symbol string) string {
	s := strings.ToUpper(symbol)
	s = strings.ReplaceAll(s, "/", "-")
	s = strings.ReplaceAll(s, "_", "-")
	s = strings.TrimSpace(s)

	// If no separator, try to detect pair
	if !strings.Contains(s, "-") {
		s = detectPair(s)
	}

	return s
}

func detectPair(s string) string {
	// Common quote currencies to detect
	quotes := []string{"USDC", "USDT", "USD", "BTC", "ETH", "LUX"}

	for _, q := range quotes {
		if strings.HasSuffix(s, q) && len(s) > len(q) {
			base := s[:len(s)-len(q)]
			return base + "-" + q
		}
	}
	return s
}

// SymbolMap maps source-specific symbols to normalized form.
type SymbolMap struct {
	aliases map[string]string
}

// NewSymbolMap creates a symbol mapper with default aliases.
func NewSymbolMap() *SymbolMap {
	return &SymbolMap{
		aliases: defaultAliases(),
	}
}

func defaultAliases() map[string]string {
	return map[string]string{
		// USD variations -> keep as USD (oracle standard)
		"LUX/USD":    "LUX-USD",
		"LUXUSD":     "LUX-USD",
		"ETH/USD":    "ETH-USD",
		"ETHUSD":     "ETH-USD",
		"BTC/USD":    "BTC-USD",
		"BTCUSD":     "BTC-USD",
		"AVAX/USD":   "AVAX-USD",
		"AVAXUSD":    "AVAX-USD",
		"SOL/USD":    "SOL-USD",
		"SOLUSD":     "SOL-USD",

		// USDC variations
		"LUX/USDC":   "LUX-USDC",
		"LUXUSDC":    "LUX-USDC",
		"ETH/USDC":   "ETH-USDC",
		"ETHUSDC":    "ETH-USDC",
		"BTC/USDC":   "BTC-USDC",
		"BTCUSDC":    "BTC-USDC",
		"AVAX/USDC":  "AVAX-USDC",
		"AVAXUSDC":   "AVAX-USDC",

		// USDT variations
		"LUX-USDT":   "LUX-USDT",
		"LUX/USDT":   "LUX-USDT",
		"LUXUSDT":    "LUX-USDT",
		"ETH-USDT":   "ETH-USDT",
		"ETH/USDT":   "ETH-USDT",
		"ETHUSDT":    "ETH-USDT",
		"BTC-USDT":   "BTC-USDT",
		"BTC/USDT":   "BTC-USDT",
		"BTCUSDT":    "BTC-USDT",
		"AVAX-USDT":  "AVAX-USDT",
		"AVAX/USDT":  "AVAX-USDT",
		"AVAXUSDT":   "AVAX-USDT",
	}
}

// Map converts a symbol to its normalized form.
func (m *SymbolMap) Map(symbol string) string {
	s := Normalize(symbol)
	if alias, ok := m.aliases[s]; ok {
		return alias
	}
	return s
}

// AddAlias adds a symbol alias.
func (m *SymbolMap) AddAlias(from, to string) {
	m.aliases[Normalize(from)] = Normalize(to)
}

// Reverse finds source symbols that map to a normalized symbol.
func (m *SymbolMap) Reverse(normalized string) []string {
	result := []string{normalized}
	target := Normalize(normalized)

	for from, to := range m.aliases {
		if Normalize(to) == target {
			result = append(result, from)
		}
	}
	return result
}

// BaseQuote splits a symbol into base and quote currencies.
func BaseQuote(symbol string) (base, quote string) {
	s := Normalize(symbol)
	parts := strings.Split(s, "-")
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return s, ""
}

// Quote extracts the quote currency from a symbol.
func Quote(symbol string) string {
	_, q := BaseQuote(symbol)
	return q
}

// Base extracts the base currency from a symbol.
func Base(symbol string) string {
	b, _ := BaseQuote(symbol)
	return b
}

// SameBase returns true if two symbols have the same base currency.
func SameBase(a, b string) bool {
	return Base(a) == Base(b)
}

// SameQuote returns true if two symbols have the same quote currency.
func SameQuote(a, b string) bool {
	return Quote(a) == Quote(b)
}

// IsUSD returns true if the symbol is quoted in a USD stablecoin.
func IsUSD(symbol string) bool {
	q := Quote(symbol)
	return q == "USD" || q == "USDC" || q == "USDT" || q == "BUSD" || q == "DAI"
}
