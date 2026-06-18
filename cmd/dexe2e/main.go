// dexe2e — LIVE EVM 0x9010 end-to-end against a running C-Chain RPC.
//
// Drives REAL signed C-Chain transactions through the 0x9010 DEX precompile
// (LP-9010 V4 = CLOB facade) -> NewZAPEngine -> the live dchain-venue D-Chain
// matcher, exercising the full custody lifecycle:
//
//	initialize  -> open the market (clob_open_market binds the pool)
//	deposit     -> native LUX into the D-Chain ledger (available)
//	modifyLiquidity -> rest a maker order (available -> locked)
//	swap        -> marketable taker cross (consensus fills, settle in-ledger)
//	cancel (modifyLiquidity negative delta) -> unlock unfilled
//	withdraw    -> burn available, release native from the 0x9010 vault
//
// Asserts the conservation invariant I = A + L + F + E across the cycle and that
// no PoolManager native reserve accrues (vault == Σ available + Σ locked).
//
// Funded signer: derived from LUX_MNEMONIC on the canonical genesis path
// m/44'/9000'/0'/0/<idx> (the path pkg/genesis funds in the C-Chain alloc).
// NO EWOQ KEY. The mnemonic is read from env (operator-managed, in-cluster).
package main

import (
	"context"
	"crypto/ecdsa"
	"encoding/hex"
	"fmt"
	"math/big"
	"os"
	"strconv"
	"strings"
	"time"

	ethereum "github.com/luxfi/geth"
	"github.com/luxfi/crypto"
	"github.com/luxfi/geth/common"
	"github.com/luxfi/geth/core/types"
	"github.com/luxfi/geth/ethclient"
	bip32 "github.com/luxfi/go-bip32"
	bip39 "github.com/luxfi/go-bip39"
)

// 0x9010 DEX precompile.
var dexAddr = common.HexToAddress("0x0000000000000000000000000000000000009010")

// Selectors (keccak4 of the V4 signatures; mirror precompile/dex/module.go).
const (
	selInitialize      = "6276cbbe" // initialize((address,address,uint24,int24,address),uint160)
	selSwap            = "f3cd914c" // swap((...),(bool,int256,uint160),bytes)
	selModifyLiquidity = "5a6bcfda" // modifyLiquidity((...),(int24,int24,int256,bytes32),bytes)
	selDeposit         = "47e7ef24" // deposit(address,uint256)
	selWithdraw        = "f3fef3a3" // withdraw(address,uint256)
	selBalanceOf       = "f7888aec" // balanceOf(address,address)
)

// Market: native LUX (currency0 = 0x0) / "LUSD" sentinel (0xC0DE), fee 0.30%,
// tickSpacing 60 — identical to the precompile clob e2e pool key.
var (
	nativeLUX = common.Address{}
	lusd      = common.HexToAddress("0x000000000000000000000000000000000000C0DE")
	fee030    = uint64(3000)
	tick030   = int64(60)
)

func must(err error, ctx string) {
	if err != nil {
		fmt.Printf("FATAL %s: %v\n", ctx, err)
		os.Exit(1)
	}
}

func pad32(b []byte) []byte {
	out := make([]byte, 32)
	copy(out[32-len(b):], b)
	return out
}

func addr32(a common.Address) []byte {
	out := make([]byte, 32)
	copy(out[12:], a.Bytes())
	return out
}

func uint256ABI(x *big.Int) []byte { return pad32(x.Bytes()) }

// int256ABI two's-complement 32-byte encoding of a possibly-negative big.Int.
func int256ABI(x *big.Int) []byte {
	if x.Sign() >= 0 {
		return pad32(x.Bytes())
	}
	mod := new(big.Int).Lsh(big.NewInt(1), 256)
	tc := new(big.Int).Add(mod, x) // mod - |x|
	return pad32(tc.Bytes())
}

// poolKeyABI encodes the 160-byte V4 PoolKey: currency0, currency1, fee,
// tickSpacing, hooks (each a 32-byte slot).
func poolKeyABI() []byte {
	buf := make([]byte, 0, 160)
	buf = append(buf, addr32(nativeLUX)...)                     // [0:32]   currency0
	buf = append(buf, addr32(lusd)...)                          // [32:64]  currency1
	buf = append(buf, uint256ABI(big.NewInt(int64(fee030)))...) // [64:96]  fee
	buf = append(buf, int256ABI(big.NewInt(tick030))...)        // [96:128] tickSpacing
	buf = append(buf, addr32(common.Address{})...)              // [128:160] hooks
	return buf
}

func sel(s string) []byte { b, _ := hex.DecodeString(s); return b }

func deriveKey(mnemonic string, idx uint32) (*ecdsa.PrivateKey, common.Address) {
	mnemonic = strings.TrimSpace(mnemonic)
	if !bip39.IsMnemonicValid(mnemonic) {
		fmt.Println("FATAL: invalid LUX_MNEMONIC")
		os.Exit(1)
	}
	seed := bip39.NewSeed(mnemonic, "")
	master, err := bip32.NewMasterKey(seed)
	must(err, "master key")
	purpose, _ := master.NewChildKey(bip32.FirstHardenedChild + 44)
	coin, _ := purpose.NewChildKey(bip32.FirstHardenedChild + 9000)
	acct, _ := coin.NewChildKey(bip32.FirstHardenedChild + 0)
	change, _ := acct.NewChildKey(0)
	child, err := change.NewChildKey(idx)
	must(err, "child key")
	pk, err := crypto.ToECDSA(child.Key)
	must(err, "ToECDSA")
	cAddr := crypto.PubkeyToAddress(pk.PublicKey) // luxfi/crypto/common.Address
	return pk, common.BytesToAddress(cAddr[:])    // -> luxfi/geth/common.Address
}

type runner struct {
	ctx     context.Context
	cl      *ethclient.Client
	chainID *big.Int
	signer  types.Signer
	key     *ecdsa.PrivateKey
	from    common.Address
	gasTip  *big.Int
	gasFee  *big.Int
}

// send signs+broadcasts a tx to 0x9010 and waits for the receipt. value is the
// native LUX attached (msg.value), used for deposit.
func (r *runner) send(label string, data []byte, value *big.Int) *types.Receipt {
	nonce, err := r.cl.PendingNonceAt(r.ctx, r.from)
	must(err, label+": nonce")
	tx := types.NewTx(&types.DynamicFeeTx{
		ChainID:   r.chainID,
		Nonce:     nonce,
		GasTipCap: r.gasTip,
		GasFeeCap: r.gasFee,
		Gas:       1_500_000,
		To:        &dexAddr,
		Value:     value,
		Data:      data,
	})
	signed, err := types.SignTx(tx, r.signer, r.key)
	must(err, label+": sign")
	must(r.cl.SendTransaction(r.ctx, signed), label+": send")
	rec := r.wait(label, signed.Hash())
	status := "OK"
	if rec.Status != types.ReceiptStatusSuccessful {
		status = "REVERTED"
	}
	fmt.Printf("  [%s] tx=%s status=%s gasUsed=%d block=%d\n",
		label, signed.Hash().Hex(), status, rec.GasUsed, rec.BlockNumber.Uint64())
	return rec
}

func (r *runner) wait(label string, h common.Hash) *types.Receipt {
	deadline := time.Now().Add(90 * time.Second)
	for time.Now().Before(deadline) {
		rec, err := r.cl.TransactionReceipt(r.ctx, h)
		if err == nil && rec != nil {
			return rec
		}
		time.Sleep(500 * time.Millisecond)
	}
	fmt.Printf("FATAL %s: receipt timeout for %s\n", label, h.Hex())
	os.Exit(1)
	return nil
}

// availOf reads the AVAILABLE D-Chain balance for (account, asset) via 0x9010
// read-only balanceOf.
func (r *runner) availOf(account, asset common.Address) *big.Int {
	data := append(sel(selBalanceOf), addr32(account)...)
	data = append(data, addr32(asset)...)
	out, err := r.cl.CallContract(r.ctx, ethereum.CallMsg{From: r.from, To: &dexAddr, Data: data}, nil)
	if err != nil {
		fmt.Printf("  balanceOf(%s,%s) call err: %v\n", account.Hex(), asset.Hex(), err)
		return big.NewInt(0)
	}
	return new(big.Int).SetBytes(out)
}

func (r *runner) nativeBal(a common.Address) *big.Int {
	b, err := r.cl.BalanceAt(r.ctx, a, nil)
	must(err, "BalanceAt")
	return b
}

// fundAccount sends `amount` native LUX from this runner's signer to `to` (used
// to fund a fresh taker from the treasury). Waits for the receipt.
func (r *runner) fundAccount(to common.Address, amount *big.Int) {
	nonce, err := r.cl.PendingNonceAt(r.ctx, r.from)
	must(err, "fund: nonce")
	tx := types.NewTx(&types.DynamicFeeTx{
		ChainID: r.chainID, Nonce: nonce,
		GasTipCap: r.gasTip, GasFeeCap: r.gasFee,
		Gas: 21000, To: &to, Value: amount,
	})
	signed, err := types.SignTx(tx, r.signer, r.key)
	must(err, "fund: sign")
	must(r.cl.SendTransaction(r.ctx, signed), "fund: send")
	r.wait("fund", signed.Hash())
	fmt.Printf("  funded %s with %s LUX (tx=%s)\n", to.Hex(), weiToLUX(amount), signed.Hash().Hex())
}

// childRunner clones this runner with a different signing key/address (same RPC,
// chain, gas params).
func (r *runner) childRunner(key *ecdsa.PrivateKey, from common.Address) *runner {
	c := *r
	c.key = key
	c.from = from
	return &c
}

// freshKey generates a brand-new random secp256k1 key (the taker counterparty).
func freshKey() (*ecdsa.PrivateKey, common.Address) {
	k, err := crypto.GenerateKey()
	must(err, "GenerateKey")
	cAddr := crypto.PubkeyToAddress(k.PublicKey)
	return k, common.BytesToAddress(cAddr[:])
}

func main() {
	rpcURL := envOr("DEX_RPC", "http://127.0.0.1:9650/ext/bc/C/rpc")
	mnemonic := os.Getenv("LUX_MNEMONIC")
	if mnemonic == "" && os.Getenv("LUX_PRIVATE_KEY") == "" {
		fmt.Println("FATAL: set LUX_PRIVATE_KEY or LUX_MNEMONIC")
		os.Exit(1)
	}
	keyIdx := uint32(0)
	if v := os.Getenv("KEY_IDX"); v != "" {
		n, _ := strconv.Atoi(v)
		keyIdx = uint32(n)
	}

	ctx := context.Background()
	cl, err := ethclient.DialContext(ctx, rpcURL)
	must(err, "dial "+rpcURL)
	chainID, err := cl.ChainID(ctx)
	must(err, "chainID")

	// Prefer an explicit funded private key (LUX_PRIVATE_KEY from the
	// operator-managed deployer secret) over mnemonic derivation. Both are
	// genesis-funded accounts; NEITHER is an ewoq key.
	var key *ecdsa.PrivateKey
	var from common.Address
	if pkHex := strings.TrimPrefix(strings.TrimSpace(os.Getenv("LUX_PRIVATE_KEY")), "0x"); pkHex != "" {
		raw, derr := hex.DecodeString(pkHex)
		must(derr, "decode LUX_PRIVATE_KEY")
		key, err = crypto.ToECDSA(raw)
		must(err, "ToECDSA(LUX_PRIVATE_KEY)")
		cAddr := crypto.PubkeyToAddress(key.PublicKey)
		from = common.BytesToAddress(cAddr[:])
		fmt.Println("(using LUX_PRIVATE_KEY)")
	} else {
		key, from = deriveKey(mnemonic, keyIdx)
	}

	fmt.Println("================ LIVE EVM 0x9010 e2e (devnet) ================")
	fmt.Printf("rpc=%s\n", rpcURL)
	fmt.Printf("chainID=%s  signer=%s (m/44'/9000'/0'/0/%d, NOT ewoq)\n", chainID, from.Hex(), keyIdx)

	r := &runner{
		ctx: ctx, cl: cl, chainID: chainID,
		signer: types.LatestSignerForChainID(chainID),
		key:    key, from: from,
		gasTip: big.NewInt(1_000_000_000),  // 1 gwei tip
		gasFee: big.NewInt(80_000_000_000), // 80 gwei cap (devnet base fee ~25 gwei)
	}

	bal := r.nativeBal(from)
	fmt.Printf("signer native balance: %s wei (%s LUX)\n", bal, weiToLUX(bal))
	if bal.Sign() == 0 {
		fmt.Println("FATAL: signer not funded at idx", keyIdx, "— try a different KEY_IDX")
		os.Exit(1)
	}

	// DEXE2E_MODE=idor runs the live cancel-authorization (RED H4) IDOR check
	// instead of the full lifecycle; default runs the full lifecycle.
	if envOr("DEXE2E_MODE", "full") == "idor" {
		runIDOR(r)
		return
	}

	run(r)
}

func envOr(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func weiToLUX(w *big.Int) string {
	f := new(big.Float).Quo(new(big.Float).SetInt(w), big.NewFloat(1e18))
	return f.Text('f', 6)
}
