# LX Trading SDK

Multi-language SDK for building high-performance trading applications.

## 🚀 Supported Languages

- **C++** - Ultra-fast, zero-copy, lock-free (10M+ msgs/sec)
- **Go** - Fast, concurrent, easy deployment (1M+ msgs/sec)
- **TypeScript/JavaScript** - Web/Node.js compatible (100K+ msgs/sec)
- **Python** - Data science friendly (50K+ msgs/sec)
- **Rust** - Memory safe, high performance (5M+ msgs/sec)

## Performance by Language

| Language | Max Throughput | Latency | Use Case |
|----------|---------------|---------|----------|
| C++ | 10M+ msgs/sec | <1μs | HFT, Market Making |
| Rust | 5M msgs/sec | <10μs | Safe HFT Systems |
| Go | 1M msgs/sec | <100μs | Exchange Backends |
| TypeScript | 100K msgs/sec | <1ms | Web Trading |
| Python | 50K msgs/sec | <10ms | Analytics, Bots |

## Quick Start

### C++ SDK
```cpp
#include "lx_trading.hpp"

auto client = lx::TradingClient("tcp://localhost:5555");
client.send_order("BTC/USD", 50000, 0.1, lx::Side::BUY);
```

### Go SDK
```go
import "github.com/luxfi/dex/sdk/go"

client := lxsdk.NewClient("localhost:5555")
client.SendOrder("BTC/USD", 50000, 0.1, lxsdk.Buy)
```

### TypeScript SDK
```typescript
import { TradingClient } from '@luxfi/trading-sdk';

const client = new TradingClient('ws://localhost:5555');
await client.sendOrder('BTC/USD', 50000, 0.1, 'buy');
```

### Python SDK
```python
from lx_trading import TradingClient

client = TradingClient('localhost:5555')
client.send_order('BTC/USD', 50000, 0.1, 'buy')
```

### Rust SDK
```rust
use lx_trading::TradingClient;

let client = TradingClient::new("localhost:5555");
client.send_order("BTC/USD", 50000.0, 0.1, Side::Buy);
```
