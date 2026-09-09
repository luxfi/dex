package gateway

import (
	"encoding/json"
	"regexp"
	"strings"
	"testing"
)

// The file is the boundary, so it is checked here rather than at every request.
//
// It ships inside the binary: nothing between writing it and serving it can
// change a decimal or a symbol, which is exactly why nothing at runtime
// re-validates one. A wrong `decimals` is how a transfer ends up a million
// times the size it was meant to be, and it has to be caught before the build.
func TestTheEmbeddedListIsWellFormed(t *testing.T) {
	var doc struct {
		Tokens []map[string]any `json:"tokens"`
	}
	if err := json.Unmarshal(vendored, &doc); err != nil {
		t.Fatalf("tokens.json: %v", err)
	}
	if len(doc.Tokens) < 900 {
		t.Fatalf("tokens.json carries %d tokens; it held 935", len(doc.Tokens))
	}

	address := regexp.MustCompile(`^0x[0-9a-fA-F]{40}$`)
	seen := map[string]bool{}
	for i, raw := range doc.Tokens {
		// No logo, and nowhere to put one. Every public token list gives logos
		// as URLs on assets.coingecko.com and three other hosts belonging to
		// somebody else; a field that does not exist cannot be filled in by
		// accident later.
		for key := range raw {
			switch key {
			case "chainId", "address", "symbol", "name", "decimals":
			default:
				t.Errorf("token %d carries %q, which is not identity", i, key)
			}
		}

		var tok listed
		b, _ := json.Marshal(raw)
		if err := json.Unmarshal(b, &tok); err != nil {
			t.Fatalf("token %d: %v", i, err)
		}

		where := tok.Symbol + " on " + itoa(tok.ChainID)
		if !address.MatchString(tok.Address) {
			t.Errorf("%s: %q is not an address", where, tok.Address)
		}
		if tok.Decimals < 0 || tok.Decimals > 36 {
			t.Errorf("%s: %d decimals", where, tok.Decimals)
		}
		if strings.TrimSpace(tok.Symbol) == "" || strings.TrimSpace(tok.Name) == "" {
			t.Errorf("%s: a token with no symbol or no name draws as a blank row", where)
		}
		// A symbol carrying a control character is drawn into a market screen
		// by whatever renders it.
		for _, r := range tok.Symbol + tok.Name {
			if r < 0x20 || r == 0x7f {
				t.Errorf("%s: control character in its name", where)
				break
			}
		}
		key := itoa(tok.ChainID) + ":" + strings.ToLower(tok.Address)
		if seen[key] {
			t.Errorf("%s: listed twice", where)
		}
		seen[key] = true
	}
}

// Every chain this gateway quotes has both numéraires listed, because a price
// is read against them and their decimals come from here. A chain missing one
// prices nothing, silently.
func TestEveryChainCanBePriced(t *testing.T) {
	chains := DefaultChainVenues("http://chain.invalid")
	ids := make([]ChainID, 0, len(chains))
	for id := range chains {
		ids = append(ids, id)
	}
	c := NewCatalog(ids)

	for id, v := range chains {
		for what, address := range map[string]string{"stable": v.Stable, "hub": v.Hub} {
			if address == "" {
				t.Errorf("chain %d names no %s", id, what)
				continue
			}
			if _, ok := c.Token(id, address); !ok {
				t.Errorf("chain %d's %s %s is not in the catalog", id, what, address)
			}
		}
	}
}

// The catalog lists what the deployment can quote and nothing else. A screen
// that can list a market this gateway cannot price is a screen with a dead row
// on it.
func TestOnlyTheChainsAskedFor(t *testing.T) {
	c := NewCatalog([]ChainID{ChainIDLux})
	if got := c.Chains(); len(got) != 1 || got[0] != ChainIDLux {
		t.Fatalf("chains: %v", got)
	}
	if n := len(c.List(ChainIDEthereum, "", 100)); n != 0 {
		t.Errorf("Ethereum was not asked for and answered %d tokens", n)
	}
	if n := len(c.List(0, "", 1000)); n != len(c.List(ChainIDLux, "", 1000)) {
		t.Errorf("every chain and the only chain differ")
	}
}

// The chain's own token first, then the dollar it is priced in. Whatever the
// order is, it is the same on every read: a listing that permutes itself
// between two reads of one deployment reads as a deployment that changed.
func TestListingOrderIsTheChains(t *testing.T) {
	c := NewCatalog([]ChainID{ChainIDEthereum})
	rows := c.List(ChainIDEthereum, "", 5)
	if len(rows) != 5 {
		t.Fatalf("got %d rows", len(rows))
	}
	if rows[0].Symbol != "WETH" || rows[1].Symbol != "USDC" {
		t.Errorf("Ethereum leads with %s, %s", rows[0].Symbol, rows[1].Symbol)
	}
	again := c.List(ChainIDEthereum, "", 5)
	for i := range rows {
		if rows[i] != again[i] {
			t.Fatalf("row %d moved between two reads", i)
		}
	}
}

// A search that answers in storage order is a search nobody can use: eleven
// Ethereum tokens contain "us" before USDC does, and a person who typed the
// symbol of the thing they wanted should not scroll past all of them.
func TestSearchAnswersWithWhatWasTyped(t *testing.T) {
	c := NewCatalog([]ChainID{ChainIDEthereum, ChainIDLux})

	for _, typed := range []string{"usdc", "USDC", "UsDc"} {
		got := c.List(ChainIDEthereum, typed, 10)
		if len(got) == 0 || got[0].Symbol != "USDC" {
			t.Errorf("%q found %v", typed, symbols(got))
		}
	}
	if got := c.List(ChainIDLux, "wlux", 5); len(got) == 0 || got[0].Symbol != "WLUX" {
		t.Errorf("wlux found %v", symbols(got))
	}
	// A whole address, in either case, is the token at that address.
	whole := c.List(ChainIDLux, strings.ToUpper(usdLux.Hub), 5)
	if len(whole) != 1 || !strings.EqualFold(whole[0].Address, usdLux.Hub) {
		t.Errorf("a whole address found %v", symbols(whole))
	}
	// And half of one is a typo. Offering the token that happens to share a
	// prefix with a mistyped address is how funds reach the wrong contract.
	if got := c.List(ChainIDLux, usdLux.Hub[:20], 5); len(got) != 0 {
		t.Errorf("half an address found %v", symbols(got))
	}
	if got := c.List(ChainIDEthereum, "no such token anywhere", 5); len(got) != 0 {
		t.Errorf("nothing matched and %d came back", len(got))
	}
}

func TestListingIsCapped(t *testing.T) {
	c := NewCatalog([]ChainID{ChainIDEthereum})
	if n := len(c.List(ChainIDEthereum, "", 3)); n != 3 {
		t.Errorf("limit 3 gave %d", n)
	}
	if n := len(c.List(ChainIDEthereum, "a", 2)); n != 2 {
		t.Errorf("a capped search gave %d", n)
	}
}

// Lower case, upper case, checksummed: one token.
func TestAddressCaseIsNotAFactAboutTheChain(t *testing.T) {
	c := NewCatalog([]ChainID{ChainIDEthereum})
	weth := usdEthereum.Hub
	for _, spelling := range []string{weth, strings.ToLower(weth), strings.ToUpper("0x" + weth[2:])} {
		got, ok := c.Token(ChainIDEthereum, spelling)
		if !ok || got.Symbol != "WETH" {
			t.Errorf("%s answered %v %v", spelling, got.Symbol, ok)
		}
	}
	if _, ok := c.Token(ChainIDEthereum, "0x0000000000000000000000000000000000000000"); ok {
		t.Error("the zero address is a token")
	}
}

func symbols(tokens []Token) []string {
	out := make([]string, len(tokens))
	for i, t := range tokens {
		out[i] = t.Symbol
	}
	return out
}

func itoa(n uint64) string {
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}
