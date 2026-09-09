package gateway

import (
	"fmt"
	"net/http"
	"strconv"
)

// ChainVenueInfo is what one chain can be quoted from here.
type ChainVenueInfo struct {
	ChainID ChainID     `json:"chainId"`
	Native  bool        `json:"native"`
	Venues  []VenueInfo `json:"venues"`
}

// handleVenues handles GET /v1/trade/venues.
//
// A venue does not exist on its own — it is a contract on a chain, and the same
// name means a different pool on each one. So the answer is grouped by chain,
// and a caller asking about one chain passes ?chainId=.
//
// This is the question a screen asks before it asks anything else: what can you
// price. Without it a client discovers a chain is unquotable by quoting on it
// and reading an error, which is a worse way to find out.
func (s *Server) handleVenues(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	asked := ChainID(0)
	if q := r.URL.Query().Get("chainId"); q != "" {
		id, err := strconv.ParseUint(q, 10, 64)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, fmt.Errorf("chainId: %w", err))
			return
		}
		asked = ChainID(id)
	}

	out := []ChainVenueInfo{}
	for _, id := range s.chains.Chains() {
		if asked != 0 && id != asked {
			continue
		}
		info := ChainVenueInfo{ChainID: id, Venues: s.chains.For(id).ListVenueInfo()}
		for _, v := range info.Venues {
			// The native arm is our own precompile, and naming it is how a
			// caller tells a market we settle from a market we read.
			if v.Name == VenueNameNative {
				info.Native = true
			}
		}
		out = append(out, info)
	}

	// Venues attached without a chain — a deployment handed one set for
	// everything, which is what a single-chain run does. They read the chain
	// this server calls its own.
	if asked == 0 && len(out) == 0 && s.venues != nil {
		out = append(out, ChainVenueInfo{
			ChainID: s.orders.defaultChainID,
			Venues:  s.venues.ListVenueInfo(),
		})
	}

	if asked != 0 && len(out) == 0 {
		s.writeError(w, http.StatusNotFound, fmt.Errorf("no venue here reads chain %d", asked))
		return
	}

	s.writeJSON(w, http.StatusOK, out)
}
