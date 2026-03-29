package gateway

import (
	"encoding/hex"
	"fmt"
	"math/big"
	"strings"
	"time"
)

// LP types for V4 position management

type CreatePositionRequest struct {
	Token0         string `json:"token0"`
	Token1         string `json:"token1"`
	Fee            uint32 `json:"fee"`
	TickLower      int32  `json:"tickLower"`
	TickUpper      int32  `json:"tickUpper"`
	Amount0Desired string `json:"amount0Desired"`
	Amount1Desired string `json:"amount1Desired"`
	Amount0Min     string `json:"amount0Min"`
	Amount1Min     string `json:"amount1Min"`
	Recipient      string `json:"recipient"`
	Deadline       int64  `json:"deadline"`
	SqrtPriceX96   string `json:"sqrtPriceX96,omitempty"`
	Salt           string `json:"salt,omitempty"`
	Hooks          string `json:"hooks,omitempty"`
	ChainID        uint64 `json:"chainId"`
}

type IncreasePositionRequest struct {
	Token0         string `json:"token0"`
	Token1         string `json:"token1"`
	Fee            uint32 `json:"fee"`
	TickLower      int32  `json:"tickLower"`
	TickUpper      int32  `json:"tickUpper"`
	Amount0Desired string `json:"amount0Desired"`
	Amount1Desired string `json:"amount1Desired"`
	Amount0Min     string `json:"amount0Min"`
	Amount1Min     string `json:"amount1Min"`
	Deadline       int64  `json:"deadline"`
	Salt           string `json:"salt,omitempty"`
	Hooks          string `json:"hooks,omitempty"`
	ChainID        uint64 `json:"chainId"`
}

type DecreasePositionRequest struct {
	Token0     string `json:"token0"`
	Token1     string `json:"token1"`
	Fee        uint32 `json:"fee"`
	TickLower  int32  `json:"tickLower"`
	TickUpper  int32  `json:"tickUpper"`
	Liquidity  string `json:"liquidity"`
	Amount0Min string `json:"amount0Min"`
	Amount1Min string `json:"amount1Min"`
	Deadline   int64  `json:"deadline"`
	Salt       string `json:"salt,omitempty"`
	Hooks      string `json:"hooks,omitempty"`
	ChainID    uint64 `json:"chainId"`
}

type ClaimFeesRequest struct {
	Token0    string `json:"token0"`
	Token1    string `json:"token1"`
	Fee       uint32 `json:"fee"`
	TickLower int32  `json:"tickLower"`
	TickUpper int32  `json:"tickUpper"`
	Recipient string `json:"recipient"`
	Salt      string `json:"salt,omitempty"`
	Hooks     string `json:"hooks,omitempty"`
	ChainID   uint64 `json:"chainId"`
}

type UnsignedTxResponse struct {
	To       string `json:"to"`
	Data     string `json:"data"`
	Value    string `json:"value"`
	GasLimit uint64 `json:"gasLimit"`
	ChainID  uint64 `json:"chainId"`
}

type PositionResponse struct {
	Owner                    string `json:"owner"`
	Token0                   string `json:"token0"`
	Token1                   string `json:"token1"`
	Fee                      uint32 `json:"fee"`
	TickLower                int32  `json:"tickLower"`
	TickUpper                int32  `json:"tickUpper"`
	Liquidity                string `json:"liquidity"`
	TokensOwed0              string `json:"tokensOwed0"`
	TokensOwed1              string `json:"tokensOwed1"`
	FeeGrowthInside0LastX128 string `json:"feeGrowthInside0LastX128"`
	FeeGrowthInside1LastX128 string `json:"feeGrowthInside1LastX128"`
}

type PoolStateResponse struct {
	SqrtPriceX96     string `json:"sqrtPriceX96"`
	Tick             int32  `json:"tick"`
	Liquidity        string `json:"liquidity"`
	FeeGrowthGlobal0 string `json:"feeGrowthGlobal0X128"`
	FeeGrowthGlobal1 string `json:"feeGrowthGlobal1X128"`
	Token0           string `json:"token0"`
	Token1           string `json:"token1"`
	Fee              uint32 `json:"fee"`
}

var (
	selectorInitialize      uint32 = 0x6276CBBE
	selectorModifyLiquidity uint32 = 0x5A6BCFDA
)

const lxPoolAddress = "0x0000000000000000000000000000000000009010"

var feeToTickSpacing = map[uint32]int32{100: 1, 500: 10, 3000: 60, 10000: 200}

const (
	gasCreatePosition  uint64 = 150_000
	gasModifyLiquidity uint64 = 80_000
	gasCollectFees     uint64 = 80_000
)

func tickSpacingForFee(fee uint32) (int32, error) {
	ts, ok := feeToTickSpacing[fee]
	if !ok {
		return 0, fmt.Errorf("unrecognized fee tier %d: must be one of 100, 500, 3000, 10000", fee)
	}
	return ts, nil
}

func writeSelector(buf []byte, sel uint32) {
	buf[0] = byte(sel >> 24)
	buf[1] = byte(sel >> 16)
	buf[2] = byte(sel >> 8)
	buf[3] = byte(sel)
}

func validateAddress(addr string) error {
	if addr == "" {
		return fmt.Errorf("address is empty")
	}
	clean := strings.TrimPrefix(addr, "0x")
	if len(clean) != 40 {
		return fmt.Errorf("invalid address length: expected 40 hex chars, got %d", len(clean))
	}
	_, err := hex.DecodeString(clean)
	if err != nil {
		return fmt.Errorf("invalid address hex: %w", err)
	}
	return nil
}

func validateTickRange(tickLower, tickUpper int32) error {
	if tickLower >= tickUpper {
		return fmt.Errorf("tickLower (%d) must be less than tickUpper (%d)", tickLower, tickUpper)
	}
	if tickLower < -887272 {
		return fmt.Errorf("tickLower (%d) below minimum (-887272)", tickLower)
	}
	if tickUpper > 887272 {
		return fmt.Errorf("tickUpper (%d) above maximum (887272)", tickUpper)
	}
	return nil
}

func validateTickAlignment(tickLower, tickUpper, tickSpacing int32) error {
	if tickSpacing <= 0 {
		return fmt.Errorf("invalid tick spacing: %d", tickSpacing)
	}
	if tickLower%tickSpacing != 0 {
		return fmt.Errorf("tickLower (%d) not aligned to tick spacing (%d)", tickLower, tickSpacing)
	}
	if tickUpper%tickSpacing != 0 {
		return fmt.Errorf("tickUpper (%d) not aligned to tick spacing (%d)", tickUpper, tickSpacing)
	}
	return nil
}

func validateDeadline(deadline int64) error {
	if deadline > 0 && deadline < time.Now().Unix() {
		return fmt.Errorf("deadline %d has already passed", deadline)
	}
	return nil
}

func validateCurrencySorted(token0, token1 string) error {
	a0 := strings.ToLower(strings.TrimPrefix(token0, "0x"))
	a1 := strings.ToLower(strings.TrimPrefix(token1, "0x"))
	if a0 >= a1 {
		return fmt.Errorf("token0 (%s) must be less than token1 (%s): currencies must be sorted", token0, token1)
	}
	return nil
}

func addressToBytes(addr string) [20]byte {
	clean := strings.TrimPrefix(addr, "0x")
	b, _ := hex.DecodeString(clean)
	var result [20]byte
	copy(result[20-len(b):], b)
	return result
}

func saltFromHex(s string) [32]byte {
	var salt [32]byte
	if s == "" {
		return salt
	}
	clean := strings.TrimPrefix(s, "0x")
	b, _ := hex.DecodeString(clean)
	if len(b) > 32 {
		b = b[:32]
	}
	copy(salt[32-len(b):], b)
	return salt
}

func encodePoolKey(token0, token1 string, fee uint32, tickSpacing int32, hooks string) []byte {
	data := make([]byte, 160)
	a0 := addressToBytes(token0)
	copy(data[12:32], a0[:])
	a1 := addressToBytes(token1)
	copy(data[44:64], a1[:])
	data[93] = byte(fee >> 16)
	data[94] = byte(fee >> 8)
	data[95] = byte(fee)
	if tickSpacing < 0 {
		for i := 96; i < 125; i++ {
			data[i] = 0xff
		}
	}
	data[125] = byte(tickSpacing >> 16)
	data[126] = byte(tickSpacing >> 8)
	data[127] = byte(tickSpacing)
	if hooks != "" && hooks != "0x0000000000000000000000000000000000000000" {
		ha := addressToBytes(hooks)
		copy(data[140:160], ha[:])
	}
	return data
}

func encodeModifyLiquidityParams(tickLower, tickUpper int32, liquidityDelta *big.Int, salt [32]byte) []byte {
	data := make([]byte, 128)
	if tickLower < 0 {
		for i := 0; i < 29; i++ {
			data[i] = 0xff
		}
	}
	data[29] = byte(tickLower >> 16)
	data[30] = byte(tickLower >> 8)
	data[31] = byte(tickLower)
	if tickUpper < 0 {
		for i := 32; i < 61; i++ {
			data[i] = 0xff
		}
	}
	data[61] = byte(tickUpper >> 16)
	data[62] = byte(tickUpper >> 8)
	data[63] = byte(tickUpper)
	copy(data[64:96], bigIntTo32BytesABI(liquidityDelta))
	copy(data[96:128], salt[:])
	return data
}

func bigIntTo32BytesABI(v *big.Int) []byte {
	result := make([]byte, 32)
	if v == nil || v.Sign() == 0 {
		return result
	}
	if v.Sign() > 0 {
		b := v.Bytes()
		if len(b) > 32 {
			b = b[len(b)-32:]
		}
		copy(result[32-len(b):], b)
	} else {
		tc := new(big.Int).Add(new(big.Int).Lsh(big.NewInt(1), 256), v)
		b := tc.Bytes()
		if len(b) > 32 {
			b = b[len(b)-32:]
		}
		copy(result[32-len(b):], b)
	}
	return result
}

func encodeInitializeCalldata(token0, token1 string, fee uint32, tickSpacing int32, hooks string, sqrtPriceX96 *big.Int) []byte {
	poolKey := encodePoolKey(token0, token1, fee, tickSpacing, hooks)
	calldata := make([]byte, 196)
	writeSelector(calldata, selectorInitialize)
	copy(calldata[4:164], poolKey)
	copy(calldata[164:196], bigIntTo32BytesABI(sqrtPriceX96))
	return calldata
}

func encodeModifyLiquidityCalldata(token0, token1 string, fee uint32, tickSpacing int32, hooks string, tickLower, tickUpper int32, liquidityDelta *big.Int, salt [32]byte) []byte {
	poolKey := encodePoolKey(token0, token1, fee, tickSpacing, hooks)
	params := encodeModifyLiquidityParams(tickLower, tickUpper, liquidityDelta, salt)
	calldata := make([]byte, 356)
	writeSelector(calldata, selectorModifyLiquidity)
	copy(calldata[4:164], poolKey)
	copy(calldata[164:292], params)
	copy(calldata[292:324], bigIntTo32BytesABI(big.NewInt(320)))
	return calldata
}

func BuildCreatePositionTx(req CreatePositionRequest) (*UnsignedTxResponse, error) {
	for _, addr := range []string{req.Token0, req.Token1, req.Recipient} {
		if err := validateAddress(addr); err != nil {
			return nil, fmt.Errorf("invalid address: %w", err)
		}
	}
	if req.Hooks != "" {
		if err := validateAddress(req.Hooks); err != nil {
			return nil, fmt.Errorf("invalid hooks address: %w", err)
		}
	}
	if err := validateCurrencySorted(req.Token0, req.Token1); err != nil {
		return nil, err
	}
	tickSpacing, err := tickSpacingForFee(req.Fee)
	if err != nil {
		return nil, err
	}
	if err := validateTickRange(req.TickLower, req.TickUpper); err != nil {
		return nil, err
	}
	if err := validateTickAlignment(req.TickLower, req.TickUpper, tickSpacing); err != nil {
		return nil, err
	}
	if err := validateDeadline(req.Deadline); err != nil {
		return nil, err
	}
	amount0 := parseBigIntStr(req.Amount0Desired)
	if amount0 == nil || amount0.Sign() <= 0 {
		return nil, fmt.Errorf("amount0Desired must be positive")
	}
	amount1 := parseBigIntStr(req.Amount1Desired)
	if amount1 == nil || amount1.Sign() <= 0 {
		return nil, fmt.Errorf("amount1Desired must be positive")
	}
	liquidityDelta := new(big.Int).Set(amount0)
	salt := saltFromHex(req.Salt)
	calldata := encodeModifyLiquidityCalldata(req.Token0, req.Token1, req.Fee, tickSpacing, req.Hooks, req.TickLower, req.TickUpper, liquidityDelta, salt)
	gas := gasModifyLiquidity
	if req.SqrtPriceX96 != "" {
		gas = gasCreatePosition
	}
	return &UnsignedTxResponse{To: lxPoolAddress, Data: "0x" + hex.EncodeToString(calldata), Value: "0", GasLimit: gas, ChainID: req.ChainID}, nil
}

func BuildInitializePoolTx(req CreatePositionRequest) (*UnsignedTxResponse, error) {
	if req.SqrtPriceX96 == "" {
		return nil, fmt.Errorf("sqrtPriceX96 is required for pool initialization")
	}
	sqrtPrice := parseBigIntStr(req.SqrtPriceX96)
	if sqrtPrice == nil || sqrtPrice.Sign() <= 0 {
		return nil, fmt.Errorf("invalid sqrtPriceX96: must be a positive integer")
	}
	tickSpacing, err := tickSpacingForFee(req.Fee)
	if err != nil {
		return nil, err
	}
	calldata := encodeInitializeCalldata(req.Token0, req.Token1, req.Fee, tickSpacing, req.Hooks, sqrtPrice)
	return &UnsignedTxResponse{To: lxPoolAddress, Data: "0x" + hex.EncodeToString(calldata), Value: "0", GasLimit: 50_000, ChainID: req.ChainID}, nil
}

func BuildIncreasePositionTx(req IncreasePositionRequest) (*UnsignedTxResponse, error) {
	for _, addr := range []string{req.Token0, req.Token1} {
		if err := validateAddress(addr); err != nil {
			return nil, fmt.Errorf("invalid address: %w", err)
		}
	}
	if req.Hooks != "" {
		if err := validateAddress(req.Hooks); err != nil {
			return nil, fmt.Errorf("invalid hooks address: %w", err)
		}
	}
	if err := validateCurrencySorted(req.Token0, req.Token1); err != nil {
		return nil, err
	}
	tickSpacing, err := tickSpacingForFee(req.Fee)
	if err != nil {
		return nil, err
	}
	if err := validateTickRange(req.TickLower, req.TickUpper); err != nil {
		return nil, err
	}
	if err := validateTickAlignment(req.TickLower, req.TickUpper, tickSpacing); err != nil {
		return nil, err
	}
	if err := validateDeadline(req.Deadline); err != nil {
		return nil, err
	}
	amount0 := parseBigIntStr(req.Amount0Desired)
	if amount0 == nil || amount0.Sign() <= 0 {
		return nil, fmt.Errorf("amount0Desired must be positive")
	}
	liquidityDelta := new(big.Int).Set(amount0)
	salt := saltFromHex(req.Salt)
	calldata := encodeModifyLiquidityCalldata(req.Token0, req.Token1, req.Fee, tickSpacing, req.Hooks, req.TickLower, req.TickUpper, liquidityDelta, salt)
	return &UnsignedTxResponse{To: lxPoolAddress, Data: "0x" + hex.EncodeToString(calldata), Value: "0", GasLimit: gasModifyLiquidity, ChainID: req.ChainID}, nil
}

func BuildDecreasePositionTx(req DecreasePositionRequest) (*UnsignedTxResponse, error) {
	for _, addr := range []string{req.Token0, req.Token1} {
		if err := validateAddress(addr); err != nil {
			return nil, fmt.Errorf("invalid address: %w", err)
		}
	}
	if req.Hooks != "" {
		if err := validateAddress(req.Hooks); err != nil {
			return nil, fmt.Errorf("invalid hooks address: %w", err)
		}
	}
	if err := validateCurrencySorted(req.Token0, req.Token1); err != nil {
		return nil, err
	}
	tickSpacing, err := tickSpacingForFee(req.Fee)
	if err != nil {
		return nil, err
	}
	if err := validateTickRange(req.TickLower, req.TickUpper); err != nil {
		return nil, err
	}
	if err := validateTickAlignment(req.TickLower, req.TickUpper, tickSpacing); err != nil {
		return nil, err
	}
	if err := validateDeadline(req.Deadline); err != nil {
		return nil, err
	}
	liquidity := parseBigIntStr(req.Liquidity)
	if liquidity == nil || liquidity.Sign() <= 0 {
		return nil, fmt.Errorf("liquidity must be positive")
	}
	liquidityDelta := new(big.Int).Neg(liquidity)
	salt := saltFromHex(req.Salt)
	calldata := encodeModifyLiquidityCalldata(req.Token0, req.Token1, req.Fee, tickSpacing, req.Hooks, req.TickLower, req.TickUpper, liquidityDelta, salt)
	return &UnsignedTxResponse{To: lxPoolAddress, Data: "0x" + hex.EncodeToString(calldata), Value: "0", GasLimit: gasModifyLiquidity, ChainID: req.ChainID}, nil
}

func BuildClaimFeesTx(req ClaimFeesRequest) (*UnsignedTxResponse, error) {
	for _, addr := range []string{req.Token0, req.Token1, req.Recipient} {
		if err := validateAddress(addr); err != nil {
			return nil, fmt.Errorf("invalid address: %w", err)
		}
	}
	if req.Hooks != "" {
		if err := validateAddress(req.Hooks); err != nil {
			return nil, fmt.Errorf("invalid hooks address: %w", err)
		}
	}
	if err := validateCurrencySorted(req.Token0, req.Token1); err != nil {
		return nil, err
	}
	tickSpacing, err := tickSpacingForFee(req.Fee)
	if err != nil {
		return nil, err
	}
	if err := validateTickRange(req.TickLower, req.TickUpper); err != nil {
		return nil, err
	}
	if err := validateTickAlignment(req.TickLower, req.TickUpper, tickSpacing); err != nil {
		return nil, err
	}
	liquidityDelta := big.NewInt(0)
	salt := saltFromHex(req.Salt)
	calldata := encodeModifyLiquidityCalldata(req.Token0, req.Token1, req.Fee, tickSpacing, req.Hooks, req.TickLower, req.TickUpper, liquidityDelta, salt)
	return &UnsignedTxResponse{To: lxPoolAddress, Data: "0x" + hex.EncodeToString(calldata), Value: "0", GasLimit: gasCollectFees, ChainID: req.ChainID}, nil
}
