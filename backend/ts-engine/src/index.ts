#!/usr/bin/env node

import { OrderBook } from './orderbook';

// Simple TypeScript engine server
class TypeScriptEngine {
  private orderbooks: Map<string, OrderBook> = new Map();

  constructor() {
    console.log('LX TypeScript Engine v1.0.0');
  }

  start(port: number = 50054) {
    console.log(`TypeScript engine would listen on port ${port}`);
    console.log('Note: gRPC server not implemented yet - this is a benchmark-only build');
  }

  getOrderBook(symbol: string): OrderBook {
    if (!this.orderbooks.has(symbol)) {
      this.orderbooks.set(symbol, new OrderBook(symbol));
    }
    return this.orderbooks.get(symbol)!;
  }
}

// Main entry point
if (require.main === module) {
  const engine = new TypeScriptEngine();
  const port = parseInt(process.env.PORT || '50054');
  engine.start(port);
}

export { TypeScriptEngine, OrderBook };