# dexkeeper — devnet 0x9999 DEX-seam deploy

The keeper that makes the on-chain **0x9999 two-phase atomic DEX swap flow**
operational on **devnet** so the first real `DEXFill` lands and the V4/DEX markets
populate. DEVNET ONLY.

## The seam (what the keeper drives)

```
C Phase-A intent (signed swap to 0x9999, DI01 hookData)
  -> SubmitSwapIntent locks input + stages C->D object (key=intentID) + emits IntentSubmitted
KEEPER watches IntentSubmitted
  -> dexvm ImportTx (consume C->D object, fund D order)        via dex.submitTx
  -> dexvm settling RelayOrderTx (clob_submit, CollateralRef=intentID)  via dex.submitTx
dexvm proposer relays ONCE (obtainFills) -> fills carried -> settleFromFills exports D->C proceeds
KEEPER polls dex.getSettlement(intentID) -> {outputID, amount}
  -> C Phase-B settle (signed swap to 0x9999, DS01 outputID|amount|intentID)
  -> ImportSettlement credits maker + emits DEXFill   <-- the goal
```

## Prerequisites (status)

1. **dexvm relay wired** — DONE. All 5 `lux-devnet` validators log
   `relayConfigured:true -> dchain-venue:9099` (per-chain config.json seeded by the
   `luxd-startup` CM; mirrored to source in `universe` `feat/dexvm-relay-devnet`).
2. **dexvm RPC seam** — code in `chains` `feat/dexvm-submit-tx-rpc`: `dex.submitTx`
   (mempool entry) + `dex.getSettlement` (proceeds-coordinate query). The luxd node
   image must be rebuilt with the new `dexvm-linux-amd64` plugin (S3) and the
   validators rolled onto it.
3. **keeper image** — `ghcr.io/luxfi/dexkeeper` (this repo, `docker.yml` matrix +
   `Dockerfile.dexkeeper`). Requires `zap >= v0.8.10` and a `chains` tag carrying the
   RPC seam to be cut first (the keeper imports both).
4. **maker key in KMS** — the devnet mnemonic is already in lux KMS; `kmssecret-mnemonic.yaml`
   syncs it to the `dexkeeper-mnemonic` secret. The keeper derives the maker
   (`0x9011…`, `m/44'/9000'/0'/0/i`) in-process.

## Deploy sequence (CI/CD + operator only)

```sh
# 0) Cut tags so the keeper image builds from published modules:
#    - luxfi/zap v0.8.10   (EncodeIntentHookData / EncodeSettlementHookData)
#    - luxfi/chains <tag>  (dexvm dex.submitTx + dex.getSettlement)
# 1) Rebuild the node image + dexvm-linux-amd64 plugin (chains tag), publish to S3,
#    bump the lux-devnet LuxNetwork CR pluginSource, roll the 5 luxd pods (OnDelete).
# 2) Build the keeper image (push to main / tag v* triggers docker.yml).
# 3) Sync the maker key + deploy the keeper:
kubectl --context do-sfo3-lux-k8s apply -f kmssecret-mnemonic.yaml
kubectl --context do-sfo3-lux-k8s apply -f deployment.yaml
# 4) Seed venue liquidity + drive the first swap:
kubectl --context do-sfo3-lux-k8s apply -f job-drive-first-swap.yaml
```

## Verify

```sh
# first DEXFill on devnet C (report the tx hash):
curl -s -X POST https://api.lux-dev.network/ext/bc/C/rpc -H 'content-type:application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"eth_getLogs","params":[{"address":"0x0000000000000000000000000000000000009999","fromBlock":"0x0","toBlock":"latest"}]}'
# DEXFill topic0 = keccak256("DEXFill(bytes32,address,uint256,uint256)")

# dex subgraph shows the market + fill (not markets:null):
curl -s -X POST https://explorer.lux-dev.network/v1/graph/cchain/dex/graphql \
  -H 'content-type:application/json' -d '{"query":"{ markets { id baseAsset quoteAsset } fills(first:3){ id } }"}'

# UI: lux.exchange?network=devnet renders the V4/DEX market.
```
