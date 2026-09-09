package gateway

import (
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"
)

// The document and the binary describe one surface, so they are held to each
// other rather than to a reader's memory.
//
// api/openapi.yaml used to describe /v1/trade/* paths that no binary served,
// while the binary served /v1/* and again /trading/*, and the three drifted
// for as long as nothing compared them. A route added in code and not written
// down is a route no client can be told about; a path written down and not
// served is a promise the binary breaks the first time somebody believes it.
func TestTheDocumentIsTheSurface(t *testing.T) {
	doc, err := os.ReadFile("../../api/openapi.yaml")
	if err != nil {
		t.Fatalf("the published contract: %v", err)
	}

	// A path key at the top level of `paths:`, which is the only thing indented
	// exactly two spaces and starting with a slash.
	key := regexp.MustCompile(`(?m)^  (/\S*):$`)

	written := map[string]bool{}
	for _, m := range key.FindAllStringSubmatch(string(doc), -1) {
		path := m[1]
		if !strings.HasPrefix(path, tradePrefix) {
			t.Errorf("%s is documented outside %s", path, tradePrefix)
			continue
		}
		// /v1/trade/pool/{chainId}/{address} is served as the subtree
		// /v1/trade/pool/ — everything after the first placeholder is read
		// out of the path by the handler.
		route := strings.TrimPrefix(path, tradePrefix)
		if i := strings.Index(route, "{"); i >= 0 {
			route = route[:i]
		}
		written[route] = true
	}

	served := map[string]bool{}
	for route := range (&Server{}).routes() {
		served[route] = true
	}

	for route := range served {
		if !written[route] {
			t.Errorf("%s%s is served and not in api/openapi.yaml", tradePrefix, route)
		}
	}
	for route := range written {
		if !served[route] {
			t.Errorf("%s%s is in api/openapi.yaml and not served", tradePrefix, route)
		}
	}

	if len(served) == 0 {
		t.Fatal("no routes")
	}
	if t.Failed() {
		t.Logf("served: %s", sorted(served))
		t.Logf("written: %s", sorted(written))
	}
}

func sorted(set map[string]bool) []string {
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
