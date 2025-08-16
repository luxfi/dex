// TypeScript OrderBook implementation
// Simple but functional for benchmarking

export enum OrderSide {
  Buy = 0,
  Sell = 1
}

export enum OrderType {
  Market = 0,
  Limit = 1
}

export enum OrderStatus {
  Pending = 0,
  PartiallyFilled = 1,
  Filled = 2,
  Cancelled = 3
}

export interface Order {
  id: number;
  userId: number;
  symbol: string;
  price: number;
  quantity: number;
  filledQuantity: number;
  side: OrderSide;
  type: OrderType;
  status: OrderStatus;
  timestamp: number;
}

export interface Trade {
  id: number;
  buyOrderId: number;
  sellOrderId: number;
  price: number;
  quantity: number;
  timestamp: number;
}

export interface PriceLevel {
  price: number;
  quantity: number;
  orderCount: number;
}

export interface OrderBookDepth {
  bids: PriceLevel[];
  asks: PriceLevel[];
}

// Binary heap for price-time priority
class OrderHeap {
  private orders: Order[] = [];
  private isAsk: boolean;

  constructor(isAsk: boolean) {
    this.isAsk = isAsk;
  }

  push(order: Order): void {
    this.orders.push(order);
    this.bubbleUp(this.orders.length - 1);
  }

  pop(): Order | undefined {
    if (this.orders.length === 0) return undefined;
    
    const result = this.orders[0];
    const end = this.orders.pop();
    
    if (this.orders.length > 0 && end) {
      this.orders[0] = end;
      this.bubbleDown(0);
    }
    
    return result;
  }

  peek(): Order | undefined {
    return this.orders[0];
  }

  size(): number {
    return this.orders.length;
  }

  private compare(a: Order, b: Order): boolean {
    if (this.isAsk) {
      // For asks, lower price has priority
      if (a.price !== b.price) {
        return a.price < b.price;
      }
    } else {
      // For bids, higher price has priority
      if (a.price !== b.price) {
        return a.price > b.price;
      }
    }
    // Same price, earlier order has priority
    return a.timestamp < b.timestamp;
  }

  private bubbleUp(index: number): void {
    const element = this.orders[index];
    
    while (index > 0) {
      const parentIndex = Math.floor((index - 1) / 2);
      const parent = this.orders[parentIndex];
      
      if (this.compare(element, parent)) {
        this.orders[index] = parent;
        index = parentIndex;
      } else {
        break;
      }
    }
    
    this.orders[index] = element;
  }

  private bubbleDown(index: number): void {
    const length = this.orders.length;
    const element = this.orders[index];
    
    while (true) {
      let swapIndex = -1;
      const leftChildIndex = 2 * index + 1;
      const rightChildIndex = 2 * index + 2;
      
      if (leftChildIndex < length) {
        const leftChild = this.orders[leftChildIndex];
        if (this.compare(leftChild, element)) {
          swapIndex = leftChildIndex;
        }
      }
      
      if (rightChildIndex < length) {
        const rightChild = this.orders[rightChildIndex];
        if (this.compare(
          rightChild,
          swapIndex === -1 ? element : this.orders[swapIndex]
        )) {
          swapIndex = rightChildIndex;
        }
      }
      
      if (swapIndex === -1) break;
      
      this.orders[index] = this.orders[swapIndex];
      index = swapIndex;
    }
    
    this.orders[index] = element;
  }
}

export class OrderBook {
  private symbol: string;
  private orders: Map<number, Order> = new Map();
  private bids: OrderHeap = new OrderHeap(false);
  private asks: OrderHeap = new OrderHeap(true);
  private nextTradeId: number = 1;
  private totalVolume: number = 0;

  constructor(symbol: string) {
    this.symbol = symbol;
  }

  addOrder(order: Order): number {
    this.orders.set(order.id, order);
    
    if (order.type === OrderType.Limit) {
      if (order.side === OrderSide.Buy) {
        this.bids.push(order);
      } else {
        this.asks.push(order);
      }
    }
    
    return order.id;
  }

  cancelOrder(orderId: number): boolean {
    const order = this.orders.get(orderId);
    if (!order) return false;
    
    order.status = OrderStatus.Cancelled;
    this.orders.delete(orderId);
    
    // Note: In production, we'd remove from heaps efficiently
    // For simplicity, we'll handle during matching
    
    return true;
  }

  matchOrders(): Trade[] {
    const trades: Trade[] = [];
    
    while (this.bids.size() > 0 && this.asks.size() > 0) {
      const bestBid = this.bids.peek();
      const bestAsk = this.asks.peek();
      
      if (!bestBid || !bestAsk) break;
      
      // Skip cancelled orders
      if (!this.orders.has(bestBid.id)) {
        this.bids.pop();
        continue;
      }
      if (!this.orders.has(bestAsk.id)) {
        this.asks.pop();
        continue;
      }
      
      // Check if orders can match
      if (bestBid.price >= bestAsk.price) {
        const bidRemaining = bestBid.quantity - bestBid.filledQuantity;
        const askRemaining = bestAsk.quantity - bestAsk.filledQuantity;
        const matchQuantity = Math.min(bidRemaining, askRemaining);
        
        // Determine match price (price-time priority)
        const matchPrice = bestBid.timestamp < bestAsk.timestamp 
          ? bestBid.price 
          : bestAsk.price;
        
        // Create trade
        const trade: Trade = {
          id: this.nextTradeId++,
          buyOrderId: bestBid.id,
          sellOrderId: bestAsk.id,
          price: matchPrice,
          quantity: matchQuantity,
          timestamp: Date.now()
        };
        trades.push(trade);
        
        // Update orders
        bestBid.filledQuantity += matchQuantity;
        bestAsk.filledQuantity += matchQuantity;
        this.totalVolume += matchQuantity;
        
        // Update status and remove if filled
        if (bestBid.filledQuantity >= bestBid.quantity) {
          bestBid.status = OrderStatus.Filled;
          this.bids.pop();
        } else {
          bestBid.status = OrderStatus.PartiallyFilled;
        }
        
        if (bestAsk.filledQuantity >= bestAsk.quantity) {
          bestAsk.status = OrderStatus.Filled;
          this.asks.pop();
        } else {
          bestAsk.status = OrderStatus.PartiallyFilled;
        }
      } else {
        break; // No more matches possible
      }
    }
    
    return trades;
  }

  getBestBid(): number {
    const bid = this.bids.peek();
    return bid ? bid.price : 0;
  }

  getBestAsk(): number {
    const ask = this.asks.peek();
    return ask ? ask.price : 0;
  }

  getDepth(maxLevels: number): OrderBookDepth {
    const bidLevels = new Map<number, PriceLevel>();
    const askLevels = new Map<number, PriceLevel>();
    
    // Aggregate bids
    for (const [_, order] of this.orders) {
      if (order.side === OrderSide.Buy && 
          (order.status === OrderStatus.Pending || 
           order.status === OrderStatus.PartiallyFilled)) {
        const remaining = order.quantity - order.filledQuantity;
        const level = bidLevels.get(order.price) || 
          { price: order.price, quantity: 0, orderCount: 0 };
        level.quantity += remaining;
        level.orderCount++;
        bidLevels.set(order.price, level);
      }
    }
    
    // Aggregate asks
    for (const [_, order] of this.orders) {
      if (order.side === OrderSide.Sell && 
          (order.status === OrderStatus.Pending || 
           order.status === OrderStatus.PartiallyFilled)) {
        const remaining = order.quantity - order.filledQuantity;
        const level = askLevels.get(order.price) || 
          { price: order.price, quantity: 0, orderCount: 0 };
        level.quantity += remaining;
        level.orderCount++;
        askLevels.set(order.price, level);
      }
    }
    
    // Sort and limit
    const bids = Array.from(bidLevels.values())
      .sort((a, b) => b.price - a.price)
      .slice(0, maxLevels);
    
    const asks = Array.from(askLevels.values())
      .sort((a, b) => a.price - b.price)
      .slice(0, maxLevels);
    
    return { bids, asks };
  }

  getVolume(): number {
    return this.totalVolume;
  }

  getOrderCount(): number {
    return this.orders.size;
  }
}