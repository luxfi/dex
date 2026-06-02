package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// registerApprovalRoutes registers ERC20 approval and Permit2 HTTP routes.
func (s *Server) registerApprovalRoutes() {
	s.mux.HandleFunc("/v1/approval/check", s.handleApprovalCheck)
	s.mux.HandleFunc("/v1/approval/build", s.handleApprovalBuild)
	s.mux.HandleFunc("/v1/permit2/check", s.handlePermit2Check)
	s.mux.HandleFunc("/v1/permit2/build", s.handlePermit2Build)
}

// handleApprovalCheck handles POST /v1/approval/check.
// Returns whether the given spender has sufficient ERC20 allowance.
// In the gateway layer (no RPC), this returns a conservative response
// indicating approval is always needed (the client should use the
// allowance calldata to query on-chain if it wants to check).
func (s *Server) handleApprovalCheck(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var req ApprovalRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	if err := req.Validate(); err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}

	// Build the approve tx so the client can submit if needed
	approveTx, err := BuildApproveTx(req)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}

	// Gateway has no RPC connection to read on-chain state;
	// return needsApproval=true with the unsigned tx and the
	// allowance check calldata so the client can query itself.
	resp := ApprovalResponse{
		Approved:      false,
		Allowance:     "0",
		NeedsApproval: true,
		ApproveTx:     approveTx,
	}

	s.writeJSON(w, http.StatusOK, resp)
}

// handleApprovalBuild handles POST /v1/approval/build.
// Returns an unsigned ERC20 approve() transaction.
func (s *Server) handleApprovalBuild(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var req ApprovalRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	tx, err := BuildApproveTx(req)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}

	s.writeJSON(w, http.StatusOK, tx)
}

// handlePermit2Check handles POST /v1/permit2/check.
// Returns whether the owner has granted sufficient Permit2 allowance for the spender.
// Like approval/check, this is a conservative response without RPC access.
func (s *Server) handlePermit2Check(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var req Permit2Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	if err := req.Validate(); err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}

	sigReq, err := BuildPermit2SignatureRequest(req)
	if err != nil {
		s.writeError(w, http.StatusBadRequest, err)
		return
	}

	resp := Permit2Response{
		Permit2Approved:  false,
		Permit2Allowance: "0",
		SignatureRequest: sigReq,
	}

	s.writeJSON(w, http.StatusOK, resp)
}

// handlePermit2Build handles POST /v1/permit2/build.
// Returns either:
//   - An unsigned Permit2 approve() tx (on-chain allowance grant), or
//   - An EIP-712 PermitSingle signature request (off-chain permit)
//
// Behavior is controlled by the "mode" query parameter:
//   - ?mode=tx  (default) returns an unsigned approve transaction
//   - ?mode=sig returns an EIP-712 signature request
func (s *Server) handlePermit2Build(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}

	var req Permit2Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid request body: %w", err))
		return
	}

	mode := r.URL.Query().Get("mode")
	if mode == "" {
		mode = "tx"
	}

	switch mode {
	case "tx":
		tx, err := BuildPermit2ApproveTx(req)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, err)
			return
		}
		s.writeJSON(w, http.StatusOK, tx)

	case "sig":
		sigReq, err := BuildPermit2SignatureRequest(req)
		if err != nil {
			s.writeError(w, http.StatusBadRequest, err)
			return
		}
		s.writeJSON(w, http.StatusOK, sigReq)

	default:
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("invalid mode %q: must be \"tx\" or \"sig\"", mode))
	}
}
