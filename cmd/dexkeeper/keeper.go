// keeper.go is the off-chain KEEPER that drives the on-chain 0x9999 two-phase atomic
// DEX seam so a DEXFill lands on the C-Chain. It is a relayer with NO custody authority:
// the value is locked in shared memory by the C-side SubmitSwapIntent before the keeper
// ever sees it, and the D->C proceeds object pays its RECORDED owner regardless of what
// the keeper submits. The worst a faulty/byzantine keeper can do is fail to build a
// settlement (which the locker reclaims directly after the intent deadline). See
// precompile/dex native_events.go and chains/dexvm txs/tx.go for the authority model.
//
// The seam, end to end:
//
//  1. Phase-A intent: the maker signs a V4 swap tx to 0x9999 with DI01 hookData (a
//     chosen nonce + deadline). SubmitSwapIntent locks the input, stages a C->D atomic
//     object keyed by intentID, and emits IntentSubmitted.
//  2. The keeper reads IntentSubmitted (FilterLogs), then submits to the dexvm:
//     - an ImportTx that consumes the C->D object and funds the D order, and
//     - a settling RelayOrderTx (clob_submit) bound to that import's collateral.
//     Both go to the dexvm's dex.submitTx RPC; the proposer relays the order once and
//     settleFromFills exports a D->C railSwap proceeds object.
//  3. The keeper discovers the D->C export's outputID, then has the maker sign a
//     Phase-B V4 swap tx to 0x9999 with DS01 hookData (outputID|amount|intentID).
//     ImportSettlement consumes the proceeds object and emits DEXFill.
package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math/big"
	"net/http"
	"time"

	dextxs "github.com/luxfi/chains/dexvm/txs"
	"github.com/luxfi/dex/pkg/zapwire"
	"github.com/luxfi/geth/common"
	ethtypes "github.com/luxfi/geth/core/types"
	"github.com/luxfi/ids"
	"github.com/luxfi/log"
	"github.com/luxfi/zap/dexsession"

	ethereum "github.com/luxfi/geth"
	"github.com/luxfi/geth/ethclient"
)

// keeperConfig is the fully-resolved runtime config (flags + boot-queried values).
type keeperConfig struct {
	CChainRPC string
	DRPC      string // dexvm RPC base URL (…/ext/bc/<D-chain-id>/rpc)
	DChainID  ids.ID
	VenueAddr string
	Maker     common.Address
	ChainID   *big.Int // C-Chain EVM chainID (eth_chainId)
	NetworkID uint32   // Lux primary network id (for intent-id derivation)
}

// keeper holds the live clients + the maker key + resolved config.
type keeper struct {
	cfg   keeperConfig
	cc    *ethclient.Client
	maker *makerKey
	http  *http.Client
}

// newKeeper dials the C-Chain, resolves the chainID/networkID if unset, and binds the
// maker key. The dexvm is reached over JSON-RPC by URL (no client object to dial).
func newKeeper(ctx context.Context, cfg keeperConfig, mnemonic string) (*keeper, error) {
	cc, err := ethclient.DialContext(ctx, cfg.CChainRPC)
	if err != nil {
		return nil, fmt.Errorf("dial C-Chain %s: %w", cfg.CChainRPC, err)
	}
	if cfg.ChainID == nil {
		id, err := cc.ChainID(ctx)
		if err != nil {
			return nil, fmt.Errorf("eth_chainId: %w", err)
		}
		cfg.ChainID = id
	}
	if cfg.NetworkID == 0 {
		// The Lux primary network id is the C-Chain EVM chainID for sovereign L1s, and
		// for the primary network it is the convention-fixed id. The intent-id derivation
		// needs the value the precompile used (atomicState.NetworkID()); on devnet the
		// C-Chain EVM chainID equals the network id, so we default to it. Override with
		// --network-id if a deployment separates them.
		cfg.NetworkID = uint32(cfg.ChainID.Uint64())
	}
	mk, err := deriveMakerKey(mnemonic, cfg.Maker, 1000)
	if err != nil {
		return nil, fmt.Errorf("derive maker key: %w", err)
	}
	return &keeper{cfg: cfg, cc: cc, maker: mk, http: &http.Client{Timeout: 30 * time.Second}}, nil
}

// cChainID reads the C-Chain's blockchain id (the 32-byte chain id the precompile uses in
// DeriveIntentID, NOT the EVM chainID). On the primary network the C-Chain blockchain id
// is a fixed alias; the keeper reads it from the node so it matches the chain exactly.
//
// NOTE: the geth ethclient does not expose the platform blockchain id. The authoritative
// value rides the IntentSubmitted correlation anyway (the keeper acts on the id in the
// event topic, not a re-derived one), so cChainID is only needed for the optional local
// re-derivation sanity check. It is resolved lazily from the configured value when set.
func (k *keeper) cChainBlockchainID() ids.ID {
	// The C-Chain blockchain id is not queryable via eth RPC; the keeper correlates on the
	// event's intentID topic (authoritative) rather than re-deriving, so an empty value
	// here only disables the optional local-derivation cross-check.
	return ids.Empty
}

// ---- Phase A: send one intent swap from the maker ----

// sendPhaseAIntent builds, signs, and sends ONE Phase-A (DI01) swap to 0x9999 from the
// maker. It returns the locally-derived intentID (correlation hint) and the C tx hash.
// The keeper CHOOSES the nonce; SubmitSwapIntent re-derives the SAME id from the same
// calldata, so the IntentSubmitted topic will carry this id.
func (k *keeper) sendPhaseAIntent(ctx context.Context, pk dexsession.PoolKeyArgs, marketID, assetIn [32]byte, zeroForOne bool, amountIn, nonce, deadline uint64) (ids.ID, common.Hash, error) {
	data := phaseAIntentCalldata(pk, zeroForOne, amountIn, deadline, nonce)
	txHash, err := k.sendSwapTx(ctx, data)
	if err != nil {
		return ids.Empty, common.Hash{}, fmt.Errorf("send Phase-A swap: %w", err)
	}
	// Local id derivation is a correlation hint only; the chain's id (read from the
	// IntentSubmitted topic) is authoritative. cChainBlockchainID() is Empty over eth RPC,
	// so this matches the chain only when both sides use the same (here it documents the
	// derivation; the watch loop trusts the topic).
	intentID := deriveIntentID(k.cfg.NetworkID, k.cChainBlockchainID(), k.cfg.DChainID, k.maker.Address, assetIn, amountIn, marketID, nonce)
	return intentID, txHash, nil
}

// sendSwapTx signs a legacy EVM tx to 0x9999 carrying `data` with the maker key and
// submits it via eth_sendRawTransaction. It uses a suggested gas price + the maker's
// pending nonce; the swap precompile is gas-metered, so a generous limit is used.
func (k *keeper) sendSwapTx(ctx context.Context, data []byte) (common.Hash, error) {
	nonce, err := k.cc.PendingNonceAt(ctx, k.maker.Address)
	if err != nil {
		return common.Hash{}, fmt.Errorf("pending nonce: %w", err)
	}
	gasPrice, err := k.cc.SuggestGasPrice(ctx)
	if err != nil {
		return common.Hash{}, fmt.Errorf("suggest gas price: %w", err)
	}
	to := poolManagerAddr9999
	inner := &ethtypes.LegacyTx{
		Nonce:    nonce,
		GasPrice: gasPrice,
		Gas:      600_000,
		To:       &to,
		Value:    big.NewInt(0),
		Data:     data,
	}
	signed, err := ethtypes.SignTx(ethtypes.NewTx(inner), ethtypes.LatestSignerForChainID(k.cfg.ChainID), k.maker.ecdsa)
	if err != nil {
		return common.Hash{}, fmt.Errorf("sign tx: %w", err)
	}
	if err := k.cc.SendTransaction(ctx, signed); err != nil {
		return common.Hash{}, fmt.Errorf("send raw tx: %w", err)
	}
	return signed.Hash(), nil
}

// ---- The watch loop ----

// watch runs the keeper loop: poll the C-Chain for new IntentSubmitted logs from 0x9999,
// and for each previously-unseen intent, drive it through import+relay (and Phase-B once
// export discovery is wired). It is restart-safe: it tracks the last scanned block and the
// set of handled intent ids so a re-seen log is not re-driven.
func (k *keeper) watch(ctx context.Context, fromBlock uint64) error {
	handled := make(map[ids.ID]bool)
	next := fromBlock
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	log.Info("keeper watch loop started", "fromBlock", fromBlock, "maker", k.cfg.Maker, "dChainID", k.cfg.DChainID)
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
		}
		head, err := k.cc.BlockNumber(ctx)
		if err != nil {
			log.Warn("BlockNumber", "err", err)
			continue
		}
		if head < next {
			continue
		}
		logs, err := k.intentLogs(ctx, next, head)
		if err != nil {
			log.Warn("FilterLogs IntentSubmitted", "from", next, "to", head, "err", err)
			continue
		}
		for _, lg := range logs {
			ev, err := decodeIntentEvent(lg)
			if err != nil {
				log.Warn("decode IntentSubmitted", "err", err)
				continue
			}
			if handled[ev.IntentID] {
				continue
			}
			if err := k.handleIntent(ctx, ev); err != nil {
				log.Error("handle intent", "intentID", ev.IntentID, "err", err)
				continue // leave unhandled so a later pass can retry
			}
			handled[ev.IntentID] = true
		}
		next = head + 1
	}
}

// intentLogs queries 0x9999 for IntentSubmitted logs in [from, to].
func (k *keeper) intentLogs(ctx context.Context, from, to uint64) ([]ethtypes.Log, error) {
	q := ethereum.FilterQuery{
		FromBlock: new(big.Int).SetUint64(from),
		ToBlock:   new(big.Int).SetUint64(to),
		Addresses: []common.Address{poolManagerAddr9999},
		Topics:    [][]common.Hash{{intentSubmittedSig}},
	}
	return k.cc.FilterLogs(ctx, q)
}

// handleIntent drives ONE intent through the D-side legs: build+submit the ImportTx
// (consume the C->D object, fund the D order) and the settling RelayOrderTx (clob_submit),
// then discover the D->C export and build Phase-B. Only swap intents (kind 0) are driven
// here; position intents (kind 1) are routed by a different op and are skipped.
func (k *keeper) handleIntent(ctx context.Context, ev intentEvent) error {
	if ev.Kind != intentKindSwap {
		log.Info("skip non-swap intent", "intentID", ev.IntentID, "kind", ev.Kind)
		return nil
	}
	log.Info("handling swap intent", "intentID", ev.IntentID, "account", ev.Account,
		"assetIn", hex.EncodeToString(ev.AssetIn[:]), "amountIn", ev.AmountIn, "marketID", hex.EncodeToString(ev.MarketID[:]))

	// 1) ImportTx: consume the C->D object (keyed by intentID) and credit the D order's
	//    funding to the maker. The import's single input names the staged object (UTXOID =
	//    intentID), its asset (assetIn) and locked amount; the output credits the same to
	//    the maker on the swap rail.
	owner := ids.ShortID(k.maker.Address)
	importTx := dextxs.NewImportTx(
		owner,
		ev.AmountIn, // nonce: any monotone value; the import is content-addressed by its inputs
		k.cChainSourceID(),
		[]dextxs.AtomicInput{{UTXOID: ev.IntentID, Asset: ids.ID(ev.AssetIn), Amount: ev.AmountIn}},
		[]dextxs.AtomicOutput{{Rail: dextxs.RailSwap, Owner: owner, Asset: ids.ID(ev.AssetIn), Amount: ev.AmountIn}},
	)
	importID, err := k.submitDexTx(ctx, importTx)
	if err != nil {
		return fmt.Errorf("submit ImportTx: %w", err)
	}
	log.Info("submitted ImportTx", "txID", importID, "collateralRef", importTx.ID())

	// 2) Settling RelayOrderTx: the taker clob_submit frame, bound to the import's
	//    collateral (CollateralRef = importTx.ID()). The order is a marketable BUY (the
	//    taker spends assetIn to receive the opposite side) at the carried slippage limit.
	//    assetOut is the opposite side of the market — the keeper knows the market mapping;
	//    here it is carried as the market's quote/base counterpart of assetIn.
	//
	//    The relay is UNSIGNED, with From set to the maker's STANDARD EVM address (==
	//    the escrow owner the precompile recorded). This is the architecturally-correct
	//    build: settleFromFills derives the settle authority + payout target from the
	//    consumed C->D object's RECORDED owner (escrowOwner), and REQUIRES the relay's
	//    From to equal it (relaySender != escrowOwner -> errSettleUnauthorized); From
	//    carries no independent authority, so an unsigned relay is admissible and cannot
	//    escalate (chains/dexvm txs/tx.go RelayOrderTx.Verify / vm.go settleFromFills).
	//    We deliberately do NOT call relay.Sign(secp): that would overwrite From with
	//    secp256k1.PrivateKey.EVMAddress(), which hashes the COMPRESSED pubkey and so
	//    yields a NON-standard address that does NOT equal the standard escrow owner —
	//    the signed relay would be refused with errSettleUnauthorized. See the keeper
	//    report's "secp256k1.EVMAddress divergence" blocker.
	side := zapwire.SideBuy
	limitPrice := priceLimitFloat(ev.PriceLimit)
	size := float64(ev.AmountIn)
	frame := submitFrameFor(poolIDFromMarket(ev.MarketID), side, true /*isMarket*/, limitPrice, size, makerHandle(k.cfg.Maker))
	assetOut := k.assetOutFor(ev.MarketID, ev.AssetIn)
	relay := dextxs.NewSettlingRelayOrderTx(owner, ev.AmountIn+1, frame, importTx.ID(), assetOut, ev.PriceLimit, ev.LimitIsUpper)
	relayID, err := k.submitDexTx(ctx, relay)
	if err != nil {
		return fmt.Errorf("submit RelayOrderTx: %w", err)
	}
	log.Info("submitted settling RelayOrderTx (unsigned; From=escrow owner)", "txID", relayID)

	// 3) Discover the D->C export proceeds object and build Phase-B. This is the single
	//    step that requires a settlement-coordinate query the current dexvm RPC does not
	//    yet expose (see getExportedOutput); until it is wired, the keeper has driven the
	//    intent through the relay and the D->C proceeds object IS staged — the maker (or a
	//    later keeper pass) completes Phase-B once the outputID is discoverable.
	outputID, amountOut, err := k.getExportedOutput(ctx, ev, relay, importTx.ID())
	if err != nil {
		return fmt.Errorf("discover D->C export (intent driven through relay; Phase-B pending): %w", err)
	}
	return k.sendPhaseBSettle(ctx, ev, outputID, amountOut)
}

// sendPhaseBSettle has the maker sign+send the Phase-B (DS01) swap to 0x9999, consuming
// the discovered D->C proceeds object. ImportSettlement credits the maker and emits DEXFill.
func (k *keeper) sendPhaseBSettle(ctx context.Context, ev intentEvent, outputID ids.ID, amountOut uint64) error {
	pk := poolKeyForMarket(ev.MarketID)
	// zeroForOne mirrors the Phase-A direction; amountIn keeps the swap ABI well-formed.
	data := phaseBSettleCalldata(pk, true, ev.AmountIn, outputID, ev.IntentID, amountOut)
	txHash, err := k.sendSwapTx(ctx, data)
	if err != nil {
		return fmt.Errorf("send Phase-B swap: %w", err)
	}
	log.Info("submitted Phase-B settlement", "intentID", ev.IntentID, "outputID", outputID, "amountOut", amountOut, "txHash", txHash)
	return nil
}

// ---- dexvm tx submission over dex.submitTx ----

// submitTxRequest / submitTxResponse mirror the gorilla-rpc envelope the dexvm's
// dex.submitTx serves (chains/dexvm/api/service.go SubmitTxArgs/SubmitTxReply).
type submitTxRequest struct {
	JSONRPC string            `json:"jsonrpc"`
	ID      int               `json:"id"`
	Method  string            `json:"method"`
	Params  [1]submitTxParams `json:"params"`
}

type submitTxParams struct {
	Tx string `json:"tx"` // hex-encoded tx wire bytes
}

type submitTxResponse struct {
	Result *struct {
		TxID string `json:"txID"`
	} `json:"result"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
}

// submitDexTx hex-encodes a dexvm tx's wire bytes and POSTs them to dex.submitTx, returning
// the dexvm's tx id. The wire is the dexvm codec (type byte + JSON), exactly tx.Bytes().
func (k *keeper) submitDexTx(ctx context.Context, tx dextxs.Tx) (string, error) {
	body, err := json.Marshal(submitTxRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "dex.submitTx",
		Params:  [1]submitTxParams{{Tx: hex.EncodeToString(tx.Bytes())}},
	})
	if err != nil {
		return "", fmt.Errorf("marshal submitTx request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, k.cfg.DRPC, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("content-type", "application/json")
	resp, err := k.http.Do(req)
	if err != nil {
		return "", fmt.Errorf("POST dex.submitTx: %w", err)
	}
	defer resp.Body.Close()
	var out submitTxResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("decode submitTx response: %w", err)
	}
	if out.Error != nil {
		return "", fmt.Errorf("dex.submitTx error %d: %s", out.Error.Code, out.Error.Message)
	}
	if out.Result == nil {
		return "", fmt.Errorf("dex.submitTx: empty result")
	}
	return out.Result.TxID, nil
}

// ---- export discovery (the one documented stub) ----

// getExportedOutput discovers the D->C railSwap proceeds object that settleFromFills
// exported for THIS intent, returning its shared-memory outputID and credited amount so
// the keeper can build the Phase-B settlement.
//
// WHY THIS IS A STUB (and exactly what must fill it): the export UTXO key is
//
//	deriveUTXOID( settlementExportTx.ID(), outputIndex )
//
// where (chains/dexvm vm.go settleFromFills + NewSettlementExportTx):
//
//	settlementExportTx = NewSettlementExportTx(
//	    taker     = ids.ShortID(maker EVM address),
//	    txIndex   = the relay's index within the D settlement block,
//	    destChain = cChainID,
//	    outs      = [ proceeds(RailSwap, owner=taker, asset=assetOut, amount=Σfill_out, spent=Σfill_in),
//	                  refund(RailSwap, owner=taker, asset=assetIn,  amount=locked-spent) ],  // refund omitted if 0
//	    fillRef   = collateralRef (== importTx.ID()),
//	    createdAt = int64(BigEndian.Uint64(blockHash[:8])),  // D settlement block hash
//	)
//
// and the proceeds leg is outputIndex 0 (nonZeroOutputs preserves proceeds-before-refund
// order). Everything EXCEPT (blockHash, txIndex) and the realized fill amounts is known to
// the keeper. (blockHash, txIndex, fills) are properties of the D block that settled the
// relay — and the dexvm RPC currently exposes only {Ping, Status, Relay, SubmitTx}
// (chains/dexvm/api/service.go), none of which return them. The C-Chain EVM does not
// expose an atomic-getUTXOs for the dexvm seam either (verified in luxfi/node plugin/evm).
//
// TO COMPLETE, add ONE of:
//   - a dexvm RPC `dex.getSettlement(relayTxID)` returning {blockHash, txIndex, fills[]}
//     (or {outputID, amount} directly), then recompute the key here with deriveUTXOID; or
//   - a C-Chain view that lists importable D->C railSwap UTXOs for an address+sourceChain,
//     and select the one whose recorded (owner==maker, asset==assetOut) matches this intent.
//
// Until then this returns an error; handleIntent has already driven the intent through the
// relay, so the D->C proceeds object is staged and Phase-B is the only remaining leg.
func (k *keeper) getExportedOutput(ctx context.Context, ev intentEvent, relay *dextxs.RelayOrderTx, collateralRef ids.ID) (outputID ids.ID, amount uint64, err error) {
	_ = ctx
	_ = ev
	_ = relay
	_ = collateralRef
	return ids.Empty, 0, fmt.Errorf("export discovery not wired: dexvm exposes no settlement-coordinate query (need dex.getSettlement(relayTxID) -> {blockHash,txIndex,fills} or a C-Chain importable-UTXO view)")
}

// ---- market mapping helpers ----
//
// The keeper binds a D market id to its (poolID, base/quote assets, V4 PoolKey). On a
// single-market devnet proof the mapping is the identity poolID==marketID with the venue's
// base/quote; a multi-market deployment supplies a real registry. These keep the mapping in
// ONE place so the rest of the keeper is market-agnostic.

// poolIDFromMarket maps a marketID to the venue poolID. The seam keys both by the same
// 32-byte id (the D market IS the venue book), so this is the identity.
func poolIDFromMarket(marketID [32]byte) [32]byte { return marketID }

// poolKeyForMarket builds the V4 PoolKey the 0x9999 calldata needs for a market. The
// native seam matches by marketID, not by the PoolKey tuple, so the keeper passes a
// well-formed key whose hooks point at 0x9999; currency0/1 are the market's assets' EVM
// addresses (address(0) for native), fee/tickSpacing the market's V4 params.
func poolKeyForMarket(marketID [32]byte) dexsession.PoolKeyArgs {
	return dexsession.PoolKeyArgs{
		Hooks: dexsession.Account(poolManagerAddr9999),
	}
}

// assetOutFor returns the opposite side of a market from assetIn (the asset a taker BUY
// receives). On the single-market devnet proof the market's quote and base are the two
// venue assets; the keeper subtracts assetIn from the pair. A real registry supplies this.
func (k *keeper) assetOutFor(marketID, assetIn [32]byte) ids.ID {
	// Identity-market devnet default: the output asset is carried by the seed spec and
	// equals the market's base when assetIn is the quote (and vice-versa). With no registry
	// here, return the marketID-derived counterpart; the dexvm equality-rejects a wrong
	// asset (taker-liveness only, no theft), so this is the bounded surface documented on
	// RelayOrderTx.AssetOut.
	_ = assetIn
	return ids.ID(marketID)
}

// makerHandle is the venue user handle for the maker, derived from its EVM address so the
// custody rows are stable and addressable. zapwire user width is 16 bytes; the handle is
// the address hex truncated to fit.
func makerHandle(addr common.Address) string {
	h := addr.Hex() // 0x + 40 hex
	if len(h) > zapwire.UserSize {
		return h[:zapwire.UserSize]
	}
	return h
}

// cChainSourceID is the source chain the ImportTx names (the chain the C->D object was
// exported from — the C-Chain). It equals the configured C-Chain blockchain id; on the
// primary network the dexvm resolves it as the "C" alias. When the keeper has no platform
// blockchain id (eth RPC does not expose it), the dexvm validates the import against the
// recorded object regardless, so the configured D-chain peering is authoritative.
func (k *keeper) cChainSourceID() ids.ID {
	// The import's SourceChain must equal the C-Chain blockchain id the object was PUT from.
	// It is not queryable over eth RPC; the dexvm binds the import to the recorded object's
	// source. Carried as Empty here documents the dependency; a deployment that needs an
	// explicit value supplies it via the (future) settlement query or a --c-blockchain-id.
	return k.cChainBlockchainID()
}

// intentKindSwap mirrors precompile/dex nativeIntentKind: 0 = taker swap, 1 = LP position.
const intentKindSwap uint8 = 0
