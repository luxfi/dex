package gateway

import (
	"encoding/hex"
	"fmt"
	"math/big"
	"strings"
)

// Permit2 canonical address (same on all EVM chains)
const Permit2Address = "0x000000000022D473030F116dDEE9F6B43aC78BA3"

// ERC20 function selectors
var (
	// approve(address,uint256) = 0x095ea7b3
	selectorERC20Approve = [4]byte{0x09, 0x5e, 0xa7, 0xb3}

	// allowance(address,address) = 0xdd62ed3e
	selectorERC20Allowance = [4]byte{0xdd, 0x62, 0xed, 0x3e}
)

// Permit2 function selectors
var (
	// approve(address,address,uint160,uint48) = 0x87517c45
	selectorPermit2Approve = [4]byte{0x87, 0x51, 0x7c, 0x45}

	// allowance(address,address,address) = 0x927da105
	selectorPermit2Allowance = [4]byte{0x92, 0x7d, 0xa1, 0x05}
)

// Gas estimates for approval operations
const (
	gasERC20Approve   uint64 = 50_000
	gasPermit2Approve uint64 = 60_000
)

// MaxUint256 is the maximum approval amount (2^256 - 1)
var MaxUint256 = func() *big.Int {
	v := new(big.Int).Lsh(big.NewInt(1), 256)
	v.Sub(v, big.NewInt(1))
	return v
}()

// MaxUint160 is the maximum Permit2 allowance amount (2^160 - 1)
var MaxUint160 = func() *big.Int {
	v := new(big.Int).Lsh(big.NewInt(1), 160)
	v.Sub(v, big.NewInt(1))
	return v
}()

// MaxUint48 is the maximum Permit2 expiration (2^48 - 1)
var MaxUint48 = func() *big.Int {
	v := new(big.Int).Lsh(big.NewInt(1), 48)
	v.Sub(v, big.NewInt(1))
	return v
}()

// --- Request/Response types ---

// ApprovalRequest is the JSON body for approval check/build endpoints.
type ApprovalRequest struct {
	TokenAddress string `json:"tokenAddress"`
	Owner        string `json:"owner"`
	Spender      string `json:"spender"`
	Amount       string `json:"amount"`
	ChainID      uint64 `json:"chainId"`
}

// ApprovalResponse is returned from the approval check endpoint.
type ApprovalResponse struct {
	Approved      bool                `json:"approved"`
	Allowance     string              `json:"allowance"`
	NeedsApproval bool                `json:"needsApproval"`
	ApproveTx     *UnsignedTxResponse `json:"approveTx,omitempty"`
}

// Permit2Request is the JSON body for Permit2 check/build endpoints.
type Permit2Request struct {
	TokenAddress string `json:"tokenAddress"`
	Owner        string `json:"owner"`
	Spender      string `json:"spender"`
	Amount       string `json:"amount"`
	Deadline     int64  `json:"deadline"`
	ChainID      uint64 `json:"chainId"`
}

// Permit2Response is returned from the Permit2 check endpoint.
type Permit2Response struct {
	Permit2Approved  bool               `json:"permit2Approved"`
	Permit2Allowance string             `json:"permit2Allowance"`
	SignatureRequest *EIP712SignRequest `json:"signatureRequest,omitempty"`
}

// EIP712SignRequest contains the typed data for the user to sign.
type EIP712SignRequest struct {
	Types       map[string][]EIP712Field `json:"types"`
	PrimaryType string                   `json:"primaryType"`
	Domain      EIP712Domain             `json:"domain"`
	Message     map[string]interface{}   `json:"message"`
}

// --- Validation ---

func (r *ApprovalRequest) Validate() error {
	if r.TokenAddress == "" {
		return fmt.Errorf("tokenAddress is required")
	}
	if !isValidHexAddress(r.TokenAddress) {
		return fmt.Errorf("invalid tokenAddress")
	}
	if r.Owner == "" {
		return fmt.Errorf("owner is required")
	}
	if !isValidHexAddress(r.Owner) {
		return fmt.Errorf("invalid owner address")
	}
	if r.Spender == "" {
		return fmt.Errorf("spender is required")
	}
	if !isValidHexAddress(r.Spender) {
		return fmt.Errorf("invalid spender address")
	}
	if r.Amount == "" {
		return fmt.Errorf("amount is required")
	}
	amt := parseBigIntStr(r.Amount)
	if amt == nil || amt.Sign() <= 0 {
		return fmt.Errorf("amount must be a positive integer")
	}
	return nil
}

func (r *Permit2Request) Validate() error {
	if r.TokenAddress == "" {
		return fmt.Errorf("tokenAddress is required")
	}
	if !isValidHexAddress(r.TokenAddress) {
		return fmt.Errorf("invalid tokenAddress")
	}
	if r.Owner == "" {
		return fmt.Errorf("owner is required")
	}
	if !isValidHexAddress(r.Owner) {
		return fmt.Errorf("invalid owner address")
	}
	if r.Spender == "" {
		return fmt.Errorf("spender is required")
	}
	if !isValidHexAddress(r.Spender) {
		return fmt.Errorf("invalid spender address")
	}
	if r.Amount == "" {
		return fmt.Errorf("amount is required")
	}
	amt := parseBigIntStr(r.Amount)
	if amt == nil || amt.Sign() <= 0 {
		return fmt.Errorf("amount must be a positive integer")
	}
	// Permit2 amounts are uint160; reject anything larger
	if amt.Cmp(MaxUint160) > 0 {
		return fmt.Errorf("amount exceeds uint160 max for Permit2")
	}
	if r.Deadline <= 0 {
		return fmt.Errorf("deadline is required")
	}
	// Permit2 expiration is uint48; reject anything larger
	if big.NewInt(r.Deadline).Cmp(MaxUint48) > 0 {
		return fmt.Errorf("deadline exceeds uint48 max for Permit2")
	}
	return nil
}

// --- Calldata builders ---

// BuildERC20ApproveCalldata encodes ERC20 approve(spender, amount).
//
// ABI layout (after 4-byte selector):
//
//	[0:32]  spender (address, left-padded to 32 bytes)
//	[32:64] amount  (uint256)
func BuildERC20ApproveCalldata(spender string, amount *big.Int) []byte {
	data := make([]byte, 68) // 4 selector + 2*32 params
	copy(data[0:4], selectorERC20Approve[:])

	// spender address
	sb := addrBytes(spender)
	copy(data[4+12:4+32], sb)

	// amount (uint256)
	ab := amount.Bytes()
	copy(data[36+32-len(ab):36+32], ab)

	return data
}

// BuildERC20AllowanceCalldata encodes ERC20 allowance(owner, spender).
func BuildERC20AllowanceCalldata(owner, spender string) []byte {
	data := make([]byte, 68)
	copy(data[0:4], selectorERC20Allowance[:])

	ob := addrBytes(owner)
	copy(data[4+12:4+32], ob)

	sb := addrBytes(spender)
	copy(data[36+12:36+32], sb)

	return data
}

// BuildPermit2ApproveCalldata encodes Permit2 approve(token, spender, amount, expiration).
//
// ABI layout (after 4-byte selector):
//
//	[0:32]   token      (address)
//	[32:64]  spender    (address)
//	[64:96]  amount     (uint160)
//	[96:128] expiration (uint48)
func BuildPermit2ApproveCalldata(token, spender string, amount *big.Int, expiration int64) []byte {
	data := make([]byte, 132) // 4 + 4*32
	copy(data[0:4], selectorPermit2Approve[:])

	// token address
	tb := addrBytes(token)
	copy(data[4+12:4+32], tb)

	// spender address
	sb := addrBytes(spender)
	copy(data[36+12:36+32], sb)

	// amount (uint160, fits in 20 bytes but padded to 32)
	ab := amount.Bytes()
	copy(data[68+32-len(ab):68+32], ab)

	// expiration (uint48, fits in 6 bytes but padded to 32)
	exp := big.NewInt(expiration)
	eb := exp.Bytes()
	copy(data[100+32-len(eb):100+32], eb)

	return data
}

// BuildPermit2AllowanceCalldata encodes Permit2 allowance(owner, token, spender).
func BuildPermit2AllowanceCalldata(owner, token, spender string) []byte {
	data := make([]byte, 100) // 4 + 3*32
	copy(data[0:4], selectorPermit2Allowance[:])

	ob := addrBytes(owner)
	copy(data[4+12:4+32], ob)

	tb := addrBytes(token)
	copy(data[36+12:36+32], tb)

	sb := addrBytes(spender)
	copy(data[68+12:68+32], sb)

	return data
}

// --- Approval transaction builders ---

// BuildApproveTx builds an unsigned ERC20 approve transaction.
func BuildApproveTx(req ApprovalRequest) (*UnsignedTxResponse, error) {
	if err := (&req).Validate(); err != nil {
		return nil, err
	}

	amount := parseBigIntStr(req.Amount)
	calldata := BuildERC20ApproveCalldata(req.Spender, amount)

	return &UnsignedTxResponse{
		To:       strings.ToLower(req.TokenAddress),
		Data:     "0x" + hex.EncodeToString(calldata),
		Value:    "0",
		GasLimit: gasERC20Approve,
		ChainID:  req.ChainID,
	}, nil
}

// BuildPermit2ApproveTx builds an unsigned Permit2 approve transaction.
// The user calls this on the Permit2 contract to grant a sub-allowance.
func BuildPermit2ApproveTx(req Permit2Request) (*UnsignedTxResponse, error) {
	if err := (&req).Validate(); err != nil {
		return nil, err
	}

	amount := parseBigIntStr(req.Amount)
	calldata := BuildPermit2ApproveCalldata(req.TokenAddress, req.Spender, amount, req.Deadline)

	return &UnsignedTxResponse{
		To:       strings.ToLower(Permit2Address),
		Data:     "0x" + hex.EncodeToString(calldata),
		Value:    "0",
		GasLimit: gasPermit2Approve,
		ChainID:  req.ChainID,
	}, nil
}

// BuildPermit2SignatureRequest builds the EIP-712 typed data for a Permit2
// PermitSingle signature. The user signs this off-chain, and the spender
// submits the permit on-chain as part of the swap flow.
func BuildPermit2SignatureRequest(req Permit2Request) (*EIP712SignRequest, error) {
	if err := (&req).Validate(); err != nil {
		return nil, err
	}

	return &EIP712SignRequest{
		Types: map[string][]EIP712Field{
			"EIP712Domain": {
				{Name: "name", Type: "string"},
				{Name: "chainId", Type: "uint256"},
				{Name: "verifyingContract", Type: "address"},
			},
			"PermitSingle": {
				{Name: "details", Type: "PermitDetails"},
				{Name: "spender", Type: "address"},
				{Name: "sigDeadline", Type: "uint256"},
			},
			"PermitDetails": {
				{Name: "token", Type: "address"},
				{Name: "amount", Type: "uint160"},
				{Name: "expiration", Type: "uint48"},
				{Name: "nonce", Type: "uint48"},
			},
		},
		PrimaryType: "PermitSingle",
		Domain: EIP712Domain{
			Name:              "Permit2",
			ChainID:           fmt.Sprintf("%d", req.ChainID),
			VerifyingContract: Permit2Address,
		},
		Message: map[string]interface{}{
			"details": map[string]interface{}{
				"token":      strings.ToLower(req.TokenAddress),
				"amount":     req.Amount,
				"expiration": fmt.Sprintf("%d", req.Deadline),
				"nonce":      "0",
			},
			"spender":     strings.ToLower(req.Spender),
			"sigDeadline": fmt.Sprintf("%d", req.Deadline),
		},
	}, nil
}
