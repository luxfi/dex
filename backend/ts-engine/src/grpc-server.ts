import * as grpc from '@grpc/grpc-js';
import * as protoLoader from '@grpc/proto-loader';
import { OrderBook } from './orderbook';
import path from 'path';

// Load proto file
const PROTO_PATH = path.join(__dirname, '../../proto/lx_engine.proto');
const packageDefinition = protoLoader.loadSync(PROTO_PATH, {
  keepCase: true,
  longs: String,
  enums: String,
  defaults: true,
  oneofs: true,
});

const proto = grpc.loadPackageDefinition(packageDefinition) as any;

interface Order {
  id: string;
  price: number;
  quantity: number;
  side: 'BUY' | 'SELL';
  timestamp: number;
}

class TypeScriptEngine {
  private orderBooks: Map<string, OrderBook> = new Map();
  private orderCounter = 0;
  private stats = {
    totalOrders: 0,
    totalCancels: 0,
    totalErrors: 0,
  };

  private getOrderBook(symbol: string): OrderBook {
    if (!this.orderBooks.has(symbol)) {
      this.orderBooks.set(symbol, new OrderBook(symbol));
    }
    return this.orderBooks.get(symbol)!;
  }

  handleSubmitOrder(call: any, callback: any) {
    try {
      const request = call.request;
      const book = this.getOrderBook(request.symbol);
      
      const orderId = `order-${++this.orderCounter}`;
      
      // For now, just track the order ID
      // The actual order book implementation would handle this
      book.addOrder({
        id: orderId,
        userId: 0,
        symbol: request.symbol,
        price: request.price,
        quantity: request.quantity,
        filledQuantity: 0,
        side: request.side === 1 ? 'BUY' : 'SELL',
        type: 'LIMIT',
        status: 'NEW',
        timestamp: Date.now(),
      } as any);
      book.matchOrders();
      
      this.stats.totalOrders++;
      
      callback(null, {
        order_id: orderId,
        status: 'ORDER_STATUS_NEW',
        timestamp: Date.now(),
      });
    } catch (error) {
      this.stats.totalErrors++;
      callback(error);
    }
  }

  handleCancelOrder(call: any, callback: any) {
    try {
      const request = call.request;
      let success = false;
      
      // Try to cancel in all order books
      for (const book of this.orderBooks.values()) {
        if (book.cancelOrder(request.order_id)) {
          success = true;
          break;
        }
      }
      
      this.stats.totalCancels++;
      
      callback(null, {
        success: success,
      });
    } catch (error) {
      this.stats.totalErrors++;
      callback(error);
    }
  }

  handleGetOrderBook(call: any, callback: any) {
    try {
      const request = call.request;
      const book = this.getOrderBook(request.symbol);
      const depth = book.getDepth(request.depth || 10);
      
      callback(null, {
        symbol: request.symbol,
        bids: depth.bids.map((b: any) => ({
          price: b.price,
          quantity: b.quantity,
        })),
        asks: depth.asks.map((a: any) => ({
          price: a.price,
          quantity: a.quantity,
        })),
        timestamp: Date.now(),
      });
    } catch (error) {
      this.stats.totalErrors++;
      callback(error);
    }
  }

  handleStreamOrderBook(call: any) {
    const request = call.request;
    const book = this.getOrderBook(request.symbol);
    
    // Send updates every 100ms
    const interval = setInterval(() => {
      const depth = book.getDepth(request.depth || 5);
      
      call.write({
        symbol: request.symbol,
        bid_updates: depth.bids.map((b: any) => ({
          price: b.price,
          quantity: b.quantity,
        })),
        ask_updates: depth.asks.map((a: any) => ({
          price: a.price,
          quantity: a.quantity,
        })),
        timestamp: Date.now(),
      });
    }, 100);
    
    // Clean up on client disconnect
    call.on('cancelled', () => {
      clearInterval(interval);
    });
    
    // Stop after 10 seconds
    setTimeout(() => {
      clearInterval(interval);
      call.end();
    }, 10000);
  }

  printStats() {
    console.log(`Stats: Orders=${this.stats.totalOrders}, Cancels=${this.stats.totalCancels}, Errors=${this.stats.totalErrors}`);
  }
}

export function startGrpcServer(port: number = 50053) {
  const server = new grpc.Server({
    'grpc.max_receive_message_length': 50 * 1024 * 1024,
    'grpc.max_send_message_length': 50 * 1024 * 1024,
  });
  
  const engine = new TypeScriptEngine();
  
  server.addService(proto.lx_engine.EngineService.service, {
    SubmitOrder: engine.handleSubmitOrder.bind(engine),
    CancelOrder: engine.handleCancelOrder.bind(engine),
    GetOrderBook: engine.handleGetOrderBook.bind(engine),
    StreamOrderBook: engine.handleStreamOrderBook.bind(engine),
  });
  
  server.bindAsync(
    `0.0.0.0:${port}`,
    grpc.ServerCredentials.createInsecure(),
    (err, port) => {
      if (err) {
        console.error('Failed to start server:', err);
        return;
      }
      console.log(`TypeScript Engine listening on port ${port}`);
      server.start();
      
      // Print stats every 5 seconds
      setInterval(() => {
        engine.printStats();
      }, 5000);
    }
  );
}

// Start server if run directly
if (require.main === module) {
  const port = parseInt(process.env.PORT || '50053');
  startGrpcServer(port);
}