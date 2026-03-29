package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
)

func (s *Server) registerLPRoutes() {
	s.mux.HandleFunc("/v1/position/increase", s.handlePositionIncrease)
	s.mux.HandleFunc("/v1/position/decrease", s.handlePositionDecrease)
	s.mux.HandleFunc("/v1/position/claim", s.handlePositionClaim)
	s.mux.HandleFunc("/v1/position", s.handlePosition)
}

func (s *Server) handlePosition(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodPost:
		s.handleCreatePosition(w, r)
	case http.MethodGet:
		s.handleGetPosition(w, r)
	default:
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
	}
}

func (s *Server) handleCreatePosition(w http.ResponseWriter, r *http.Request) {
	var req CreatePositionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}
	var response interface{}
	if req.SqrtPriceX96 != "" {
		initTx, err := BuildInitializePoolTx(req)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, err)
			return
		}
		addTx, err := BuildCreatePositionTx(req)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, err)
			return
		}
		response = map[string]interface{}{
			"initialize":       initTx,
			"addLiquidity":     addTx,
			"transactionCount": 2,
		}
	} else {
		tx, err := BuildCreatePositionTx(req)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, err)
			return
		}
		response = tx
	}
	s.writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleGetPosition(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	owner := q.Get("owner")
	chainID, _ := strconv.ParseUint(q.Get("chainId"), 10, 64)
	if owner == "" {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("owner is required"))
		return
	}
	ctx := s.requestContext(r)
	positions, err := s.router.GetPositions(ctx, PositionsRequest{ChainID: ChainID(chainID), Owner: owner})
	if err != nil {
		s.writeError(w, http.StatusInternalServerError, err)
		return
	}
	s.writeJSON(w, http.StatusOK, positions)
}

func (s *Server) handlePositionIncrease(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	var req IncreasePositionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}
	tx, err := BuildIncreasePositionTx(req)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}
	s.writeJSON(w, http.StatusOK, tx)
}

func (s *Server) handlePositionDecrease(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	var req DecreasePositionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}
	tx, err := BuildDecreasePositionTx(req)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}
	s.writeJSON(w, http.StatusOK, tx)
}

func (s *Server) handlePositionClaim(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	var req ClaimFeesRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}
	tx, err := BuildClaimFeesTx(req)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}
	s.writeJSON(w, http.StatusOK, tx)
}
