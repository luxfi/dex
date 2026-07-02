# LX Architecture

## Overview

The LX is built on a highly scalable, modular architecture designed to achieve planet-scale performance with sub-microsecond latency. The system uses a multi-layer approach with specialized components for different aspects of trading.

## System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Client Layer                          │
│  Web UI | Mobile | API Clients | Trading Bots | SDKs     │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────────┐
│                    Gateway Layer                         │
│  Load Balancer | Rate Limiter | Auth | WebSocket/gRPC   │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────────┐
│                  Application Layer                       │
│  Order Management | Risk Engine | Market Data | Admin    │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────────┐
│                   Core Engine Layer                      │
│  Matching Engine | Clearing | Settlement | Liquidation   │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────────┐
│                  Consensus Layer                         │
│  DAG Consensus | Validator Network | State Machine       │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────┴─────────────────────────────────────┐
│                   Storage Layer                          │
│  Order DB | Trade History | State DB | Archive          │
└──────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Matching Engine

The heart of the DEX, responsible for order matching with extreme performance.

#### Architecture
```go
type MatchingEngine struct {
    orderBooks   map[string]*OrderBook  // Per-symbol order books
    lockFreePool *LockFreeOrderPool     // Lock-free order allocation
    tradeBuffer  *CircularTradeBuffer   // Zero-copy trade buffer
    backend      Backend                // Go/C++/GPU backend
}
```

#### Key Features
- **Lock-free data structures**: Atomic operations for concurrency
- **Memory pooling**: Reuse allocations to minimize GC
- **Multiple backends**: Auto-select optimal implementation
- **Price-time priority**: Fair order matching algorithm

#### Performance Optimizations
- **Integer price levels**: Avoid floating-point operations
- **B-tree for price levels**: O(log n) operations
- **Circular buffers**: Zero-copy trade recording
- **SIMD operations**: Vectorized computations (C++ backend)

### 2. Order Book

Each trading pair has its own order book with optimized data structures.

#### Structure
```
        BID SIDE                    ASK SIDE
    ┌─────────────┐            ┌─────────────┐
    │ Price: 50100│            │ Price: 50101│
    │ Orders: [...│            │ Orders: [...│
    └──────┬──────┘            └──────┬──────┘
           │                          │
    ┌──────┴──────┐            ┌──────┴──────┐
    │ Price: 50099│            │ Price: 50102│
    │ Orders: [...│            │ Orders: [...│
    └──────┬──────┘            └──────┬──────┘
           │                          │
         (B-Tree)                  (B-Tree)
```

#### Order Types
- **Market**: Execute immediately at best price
- **Limit**: Execute at specified price or better
- **Stop**: Trigger when price reaches threshold
- **Iceberg**: Show only partial size
- **Hidden**: Completely hidden from book
- **Pegged**: Track reference price

### 3. Risk Engine

Pre-trade and post-trade risk management.

#### Pre-trade Checks
- Balance verification
- Position limits
- Leverage constraints
- Rate limiting
- Order validation

#### Post-trade Processing
- PnL calculation
- Margin requirements
- Liquidation monitoring
- Insurance fund management

### 4. Clearing & Settlement

Handles the finalization of trades.

#### Process Flow
1. **Trade Matching**: Orders matched in engine
2. **Clearing**: Verify and record trade
3. **Settlement**: Update balances
4. **Confirmation**: Notify participants

#### Atomic Settlement
All operations in single transaction:
```go
func (c *Clearinghouse) SettleTrade(trade *Trade) error {
    tx := c.BeginTransaction()
    defer tx.Rollback()
    
    // Update buyer balance
    tx.DebitAccount(trade.Buyer, trade.QuoteAmount)
    tx.CreditAccount(trade.Buyer, trade.BaseAmount)
    
    // Update seller balance
    tx.DebitAccount(trade.Seller, trade.BaseAmount)
    tx.CreditAccount(trade.Seller, trade.QuoteAmount)
    
    // Record trade
    tx.RecordTrade(trade)
    
    return tx.Commit()
}
```

### On-chain settlement (C↔D)

The matching engine above runs on the **D-Chain** (`dexvm`). It does not move
EVM balances directly. Instead, **D matches and C settles**: the D-Chain matches
an order, emits a `DFillReceipt`, and a D-validator quorum BLS-signs the receipt
root; the **C-Chain** receipt-settlement precompile at `0x9999` then verifies that
certificate inline and debits/credits EVM balances.

```
┌──────────────── D-CHAIN (dexvm, ~10 validators) ────────────────┐
│  match order  →  fill  →  DFillReceipt  →  BLS-sign receipt root │
└───────────────────────────────┬─────────────────────────────────┘
                                 │ receipt + Merkle proof + BLS cert
                                 │ (in V4 hookData / predicate rail)
┌───────────────────────────────▼─────────────────────────────────┐
│  C-CHAIN 0x9999 (cEVM, ~1000 validators)                        │
│    verify cert INLINE (deterministic → fork-safe)               │
│    debit tokenIn(sender) · credit tokenOut(recipient)           │
│    mark consumedReceipt[receiptID]  ·  Block-STM parallel        │
└──────────────────────────────────────────────────────────────────┘

   D matches · C settles · BLS certifies day-1 · Q later · X finalizes
```

Key properties:

- **Build vs verify**: only the build/propose path touches a live matcher; the
  C-Chain verify path settles **solely** from the certificate, never calling a
  live matcher — this is what keeps a single-tx synchronous swap fork-safe.
- **Uniswap-V4 ABI**: `0x9999` exposes the V4 `PoolManager` ABI unchanged, so
  existing routers/clients work with only an address change (`0x9010` is
  deprecated; same impl, same namespace).
- **Block-STM**: fine-grained per-account/asset/receipt/pool keys, no global hot
  slots, so thousands of independent fills settle in parallel.
- **Upgradeable cert**: `certType` (BLS day-1 → Q / PQ / ZK) is resolved via an
  on-chain verifier registry with no ABI change.
- **Halt**: multi-layer, fund-preserving default (stop new swaps; allow cancel,
  settle of safe receipts, and withdrawals).

The normative specification is **LP-9999** (`DEX V4 Receipt-Settlement
Precompile`); this document references it and does not duplicate it.

### 5. DAG Consensus

Directed Acyclic Graph consensus for parallel processing.

#### Architecture
```
    ┌───────┐
    │ Node A│───────┐
    └───┬───┘       │
        │       ┌───▼───┐
        │       │ Node C│
        │       └───┬───┘
    ┌───▼───┐       │
    │ Node B│───────┘
    └───────┘
```

#### Features
- **Parallel validation**: Multiple chains process simultaneously
- **Fast finality**: 50ms consensus time
- **Quantum-resistant**: Post-quantum signatures ready

### 6. Cross-Chain Bridge

Universal bridge for asset transfers between chains.

#### Supported Chains
- Ethereum & EVM-compatible
- Cosmos ecosystem
- Solana
- Bitcoin (via wrapped tokens)
- Lux native chain

#### Bridge Flow
1. **Lock**: Assets locked on source chain
2. **Verify**: Validators confirm transaction
3. **Mint**: Equivalent assets minted on destination
4. **Complete**: User receives bridged assets

## Data Flow

### Order Lifecycle

```
Client Request
     │
     ▼
[Gateway Layer]
     │
     ├─> Rate Limiting
     ├─> Authentication
     └─> Validation
     │
     ▼
[Risk Engine]
     │
     ├─> Balance Check
     ├─> Position Limits
     └─> Margin Requirements
     │
     ▼
[Matching Engine]
     │
     ├─> Order Book Update
     ├─> Match Orders
     └─> Generate Trades
     │
     ▼
[Clearinghouse]
     │
     ├─> Clear Trades
     ├─> Update Balances
     └─> Settlement
     │
     ▼
[Consensus Layer]
     │
     ├─> Validate Block
     ├─> Achieve Consensus
     └─> Finalize State
     │
     ▼
[Storage Layer]
     │
     └─> Persist Data
```

## Performance Architecture

### Multi-Engine Design

The DEX supports multiple execution backends. Order-matching throughput is
11.88M orders/sec (C++, 10 threads, 169 ns avg match) / 2.2M orders/sec (pure
Go) on the default CPU build, and up to 12.76B orders/sec (AMD 8060S) / 9.13B
(GB10) on the GPU-native per-book matcher (backend 4). The per-engine msgs/sec
and latency figures below are **stale December-2024 FIX wire encode/decode
rates** (a separate axis from matching) and have not been re-verified.

1. **Pure Go Engine**
   - Portable and maintainable
   - 163K-332K msgs/sec (FIX protocol)
   - 33.5μs average latency

2. **C++ Engine**
   - SIMD optimizations
   - Lock-free algorithms
   - 444K-1.08M msgs/sec (FIX protocol)
   - 8.2μs average latency

3. **Rust Engine**
   - Memory safety with performance
   - 232K-586K msgs/sec (FIX protocol)
   - 11.9μs average latency

4. **GPU-native per-book matcher** (`pkg/lx/orderbook_gpu.go`, CGO_ENABLED=1)
   - Unified `lux-gpu` backend, runtime-select CUDA > HIP > Metal > CPU
   - Byte-identical to the CPU oracle (`MatchOrderCPU`), parity-verified
     (`pkg/lxgpu/orderbook_parity_test.go`, `three_mode_parity_test.go`)
   - Up to 12.76B orders/sec (AMD 8060S) / 9.13B (GB10) / 5.60B (M4 Max) /
     2.80B (M1 Max); 21.9B on a two-node fabric — deterministic per-book,
     one thread per book across millions of books
   - Kernels ship prebuilt from luxcpp/dex

### FIX Protocol Performance (December 2024 — stale, not re-verified this session)

FIX wire encode/decode rates only (a separate axis from order matching); treat
as historical, not current.

| Engine | NewOrderSingle | ExecutionReport | MarketDataSnapshot |
|--------|----------------|-----------------|-------------------|
| Pure Go | 163K/sec | 124K/sec | 332K/sec |
| Hybrid Go/C++ | 167K/sec | 378K/sec | 616K/sec |
| Pure C++ | 444K/sec | 804K/sec | 1.08M/sec |
| Rust | 484K/sec | 232K/sec | 586K/sec |

### Memory Management

#### Object Pooling
```go
var orderPool = sync.Pool{
    New: func() interface{} {
        return &Order{}
    },
}

func GetOrder() *Order {
    return orderPool.Get().(*Order)
}

func PutOrder(order *Order) {
    order.Reset()
    orderPool.Put(order)
}
```

#### Zero-Copy Techniques
- Use of `unsafe` for direct memory access
- Circular buffers for trades
- Memory-mapped files for large datasets

### Concurrency Model

#### Lock-Free Operations
```go
type AtomicCounter struct {
    value atomic.Int64
}

func (c *AtomicCounter) Increment() int64 {
    return c.value.Add(1)
}
```

#### Goroutine Management
- Worker pools for order processing
- Bounded channels for backpressure
- Context for graceful shutdown

## Scalability

### Horizontal Scaling

#### Sharding Strategy
- **By Symbol**: Each shard handles specific trading pairs
- **By User**: Users distributed across shards
- **By Region**: Geographic distribution

#### Load Balancing
- Consistent hashing for shard selection
- Health-based routing
- Automatic failover

### Vertical Scaling

#### Hardware Optimization
- CPU pinning for critical threads
- NUMA-aware memory allocation
- Kernel bypass networking (DPDK)

## Security Architecture

### Cryptography
- **Ed25519**: Digital signatures
- **BLS**: Aggregate signatures
- **Post-Quantum**: Lattice-based crypto ready

### Access Control
- **JWT**: API authentication
- **RBAC**: Role-based permissions
- **2FA**: Two-factor authentication

### Network Security
- **TLS 1.3**: Encrypted connections
- **DDoS Protection**: Rate limiting and filtering
- **Firewall Rules**: Strict ingress/egress

## Monitoring & Observability

### Metrics Collection
```go
var (
    ordersProcessed = prometheus.NewCounter(
        prometheus.CounterOpts{
            Name: "orders_processed_total",
            Help: "Total number of orders processed",
        },
    )
    
    orderLatency = prometheus.NewHistogram(
        prometheus.HistogramOpts{
            Name: "order_latency_seconds",
            Help: "Order processing latency",
        },
    )
)
```

### Distributed Tracing
- OpenTelemetry integration
- Request flow visualization
- Performance bottleneck identification

### Logging
- Structured logging with context
- Log aggregation (ELK stack)
- Real-time alerting

## Deployment Architecture

### Container Strategy
```yaml
services:
  matching-engine:
    replicas: 3
    resources:
      limits:
        cpu: "4"
        memory: "8Gi"
    
  risk-engine:
    replicas: 2
    resources:
      limits:
        cpu: "2"
        memory: "4Gi"
```

### Kubernetes Deployment
- StatefulSets for order books
- Horizontal Pod Autoscaling
- Persistent volume claims for state

### High Availability
- Multi-region deployment
- Automatic failover
- Data replication
- Disaster recovery

## Future Enhancements

### Planned Features
1. **Layer 2 Scaling**: Optimistic rollups
2. **Advanced Order Types**: Options, futures
3. **AI Market Making**: ML-based liquidity
4. **Quantum Computing**: Quantum-resistant fully
5. **Hardware Acceleration**: FPGA/ASIC support