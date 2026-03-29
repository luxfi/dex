package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// registerMultihopRoutes registers multihop routing HTTP routes.
func (s *Server) registerMultihopRoutes() {
	s.mux.HandleFunc("/v1/route", s.handleRoute)
}

// handleRoute handles POST /v1/route.
// Finds optimal route(s) for a swap through multiple pools.
func (s *Server) handleRoute(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var req RoutingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	if err := req.Validate(); err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}

	// Default chain ID
	if req.ChainID == 0 {
		req.ChainID = uint64(ChainIDLux)
	}

	// Fetch known pools from providers
	ctx := s.requestContext(r)
	pools, err := s.router.GetPools(ctx, PoolsRequest{ChainID: ChainID(req.ChainID)})
	if err != nil {
		// No pools available; return empty routes rather than error
		s.writeJSON(w, http.StatusOK, RoutingResponse{
			Routes:  []MultihopRoute{},
			ChainID: req.ChainID,
		})
		return
	}

	resp, err := BuildRoutesFromPools(pools, req)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}

	s.writeJSON(w, http.StatusOK, resp)
}
