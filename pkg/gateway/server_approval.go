package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"time"
)

// handleApprovalCheck handles POST /v1/trade/approval/check.
// Returns whether the spender may already move the amount asked for.
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

	want, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || want.Sign() < 0 {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("amount %q is not a whole number of the token's smallest unit", req.Amount))
		return
	}

	held, err := s.read(s.requestContext(r), ChainID(req.ChainID), req.TokenAddress,
		BuildERC20AllowanceCalldata(req.Owner, req.Spender))
	if err != nil {
		s.writeError(w, http.StatusBadGateway, err)
		return
	}

	resp := ApprovalResponse{
		Allowance: held[0].String(),
		Approved:  held[0].Cmp(want) >= 0,
	}
	resp.NeedsApproval = !resp.Approved
	if resp.NeedsApproval {
		// Only when it is needed. A tx handed back beside "you already
		// approved" is a tx somebody sends.
		if resp.ApproveTx, err = BuildApproveTx(req); err != nil {
			s.writeError(w, http.StatusBadRequest, err)
			return
		}
	}

	s.writeJSON(w, http.StatusOK, resp)
}

// read asks a chain a question and returns the answer word by word.
//
// Both checks below reported allowance 0 and needsApproval on every request,
// under a comment saying the gateway had no way to read a chain. It has one per
// chain — the same endpoint the pool is quoted from, one field away. The cost
// of that guess was not a wrong number on a screen: a wallet told it needs an
// approval for a token it has already approved sends an approve transaction,
// and the user pays gas for a state change that changes nothing, before every
// swap, forever.
func (s *Server) read(ctx context.Context, chain ChainID, to string, data []byte) ([]*big.Int, error) {
	rpc := s.chains.RPC(chain)
	if rpc == "" {
		return nil, fmt.Errorf("no endpoint here reads chain %d", chain)
	}
	out, err := NewEVMClient(rpc).CallContract(ctx, to, data)
	if err != nil {
		return nil, err
	}
	if len(out) < 32 {
		return nil, fmt.Errorf("chain %d answered %d bytes for a call that returns words", chain, len(out))
	}
	words := make([]*big.Int, 0, len(out)/32)
	for i := 0; i+32 <= len(out); i += 32 {
		words = append(words, new(big.Int).SetBytes(out[i:i+32]))
	}
	return words, nil
}

// handleApprovalBuild handles POST /v1/trade/approval/build.
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

// handlePermit2Check handles POST /v1/trade/permit2/check.
// Returns whether the owner has granted the spender a live Permit2 allowance
// for the amount asked for.
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

	want, ok := new(big.Int).SetString(req.Amount, 10)
	if !ok || want.Sign() < 0 {
		s.writeError(w, http.StatusBadRequest, fmt.Errorf("amount %q is not a whole number of the token's smallest unit", req.Amount))
		return
	}

	// allowance(owner, token, spender) -> (uint160 amount, uint48 expiration,
	// uint48 nonce). The expiration is half the answer: a permit that has run
	// out is not an allowance, and reading only the amount reports a wallet as
	// ready to swap on a grant that expired.
	held, err := s.read(s.requestContext(r), ChainID(req.ChainID), Permit2Address,
		BuildPermit2AllowanceCalldata(req.Owner, req.TokenAddress, req.Spender))
	if err != nil {
		s.writeError(w, http.StatusBadGateway, err)
		return
	}
	if len(held) < 2 {
		s.writeError(w, http.StatusBadGateway, fmt.Errorf("permit2 on chain %d answered %d words, want 3", req.ChainID, len(held)))
		return
	}

	live := held[1].Int64() > time.Now().Unix()
	resp := Permit2Response{
		Permit2Allowance: held[0].String(),
		Permit2Approved:  live && held[0].Cmp(want) >= 0,
	}
	if !resp.Permit2Approved {
		if resp.SignatureRequest, err = BuildPermit2SignatureRequest(req); err != nil {
			s.writeError(w, http.StatusBadRequest, err)
			return
		}
	}

	s.writeJSON(w, http.StatusOK, resp)
}

// handlePermit2Build handles POST /v1/trade/permit2/build.
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
