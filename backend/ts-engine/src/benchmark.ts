#!/usr/bin/env node

import { OrderBook, Order, OrderSide, OrderType, OrderStatus } from './orderbook';
import * as fs from 'fs';
import * as path from 'path';
import { parse } from 'csv-parse/sync';

interface FIXMessage {
  MsgType: string;
  Symbol: string;
  OrderID: string;
  Side: string;
  OrderType: string;
  Price: number;
  Quantity: number;
  Timestamp: string;
}

interface BenchmarkResult {
  engineName: string;
  messagesProcessed: number;
  ordersCreated: number;
  tradesExecuted: number;
  duration: number; // milliseconds
  throughput: number; // messages per second
  avgLatency: number; // microseconds
  p99Latency: number; // microseconds
}

class Benchmark {
  private verbose: boolean = false;

  constructor(verbose: boolean = false) {
    this.verbose = verbose;
  }

  async run(dataFile: string | null, iterations: number = 1, warmup: number = 1000): Promise<void> {
    console.log('=== TypeScript Engine Benchmark ===\n');

    // Load or generate data
    let messages: FIXMessage[];
    if (dataFile && fs.existsSync(dataFile)) {
      messages = this.loadFIXData(dataFile);
      console.log(`Loaded ${messages.length} FIX messages from ${dataFile}`);
    } else {
      messages = this.generateSyntheticData(100000);
      console.log(`Generated ${messages.length} synthetic FIX messages`);
    }

    // Run benchmark
    const result = this.benchmarkEngine(messages, iterations, warmup);
    
    // Print results
    this.printResults([result]);
  }

  private loadFIXData(filePath: string): FIXMessage[] {
    const content = fs.readFileSync(filePath, 'utf-8');
    
    // Check if CSV
    if (content.includes(',')) {
      const records = parse(content, {
        columns: true,
        skip_empty_lines: true
      });
      
      return records.map((r: any) => ({
        MsgType: r.MsgType,
        Symbol: r.Symbol,
        OrderID: r.OrderID,
        Side: r.Side,
        OrderType: r.OrderType,
        Price: parseFloat(r.Price) || 0,
        Quantity: parseFloat(r.Quantity) || 0,
        Timestamp: r.Timestamp
      }));
    }
    
    // Otherwise parse as FIX
    return this.parseFIXMessages(content);
  }

  private parseFIXMessages(content: string): FIXMessage[] {
    const messages: FIXMessage[] = [];
    const lines = content.split('\n');
    
    for (const line of lines) {
      if (!line.trim()) continue;
      
      const fields = line.split('\x01'); // SOH separator
      const msg: any = {};
      
      for (const field of fields) {
        const [tag, value] = field.split('=');
        switch (tag) {
          case '35': msg.MsgType = value; break;
          case '55': msg.Symbol = value; break;
          case '37': msg.OrderID = value; break;
          case '54': msg.Side = value; break;
          case '40': msg.OrderType = value; break;
          case '44': msg.Price = parseFloat(value); break;
          case '38': msg.Quantity = parseFloat(value); break;
          case '52': msg.Timestamp = value; break;
        }
      }
      
      if (msg.MsgType && msg.Symbol) {
        messages.push(msg as FIXMessage);
      }
    }
    
    return messages;
  }

  private generateSyntheticData(count: number): FIXMessage[] {
    const messages: FIXMessage[] = [];
    const basePrice = 50000; // BTC price
    
    for (let i = 0; i < count; i++) {
      // Mix of order types
      let msgType = 'D'; // New Order
      if (i % 10 === 0) msgType = 'F'; // Cancel
      else if (i % 20 === 0) msgType = 'G'; // Modify
      
      // Randomize price
      const spread = 100;
      const price = basePrice + ((i % 100) - 50) * spread / 50;
      
      // Randomize quantity
      const quantity = (1 + (i % 10)) * 0.1;
      
      // Alternate sides
      const side = i % 2 === 0 ? '1' : '2';
      
      messages.push({
        MsgType: msgType,
        Symbol: 'BTC-USD',
        OrderID: i.toString(),
        Side: side,
        OrderType: '2', // Limit
        Price: price,
        Quantity: quantity,
        Timestamp: new Date().toISOString()
      });
    }
    
    return messages;
  }

  private benchmarkEngine(messages: FIXMessage[], iterations: number, warmup: number): BenchmarkResult {
    let totalDuration = 0;
    let totalOrders = 0;
    let totalTrades = 0;
    const latencies: number[] = [];
    
    for (let iter = 0; iter < iterations; iter++) {
      if (this.verbose) {
        console.log(`Iteration ${iter + 1}/${iterations}`);
      }
      
      const orderBook = new OrderBook('BTC-USD');
      let ordersCreated = 0;
      let tradesExecuted = 0;
      
      const startTime = Date.now();
      
      for (let i = 0; i < messages.length; i++) {
        const msg = messages[i];
        const msgStart = process.hrtime.bigint();
        
        switch (msg.MsgType) {
          case 'D': // New Order
            const order: Order = {
              id: i,
              userId: 1,
              symbol: msg.Symbol,
              price: msg.Price,
              quantity: msg.Quantity,
              filledQuantity: 0,
              side: msg.Side === '1' ? OrderSide.Buy : OrderSide.Sell,
              type: msg.OrderType === '1' ? OrderType.Market : OrderType.Limit,
              status: OrderStatus.Pending,
              timestamp: Date.now()
            };
            
            orderBook.addOrder(order);
            ordersCreated++;
            
            // Match orders
            const trades = orderBook.matchOrders();
            tradesExecuted += trades.length;
            break;
            
          case 'F': // Cancel
            const cancelId = parseInt(msg.OrderID);
            if (!isNaN(cancelId)) {
              orderBook.cancelOrder(cancelId);
            }
            break;
            
          case 'G': // Modify
            // For simplicity, cancel and re-add
            const modifyId = parseInt(msg.OrderID);
            if (!isNaN(modifyId)) {
              orderBook.cancelOrder(modifyId);
              const newOrder: Order = {
                id: modifyId,
                userId: 1,
                symbol: msg.Symbol,
                price: msg.Price,
                quantity: msg.Quantity,
                filledQuantity: 0,
                side: msg.Side === '1' ? OrderSide.Buy : OrderSide.Sell,
                type: OrderType.Limit,
                status: OrderStatus.Pending,
                timestamp: Date.now()
              };
              orderBook.addOrder(newOrder);
            }
            break;
        }
        
        if (i >= warmup) {
          const msgEnd = process.hrtime.bigint();
          const latencyNs = Number(msgEnd - msgStart);
          latencies.push(latencyNs / 1000); // Convert to microseconds
        }
      }
      
      const iterDuration = Date.now() - startTime;
      totalDuration += iterDuration;
      totalOrders += ordersCreated;
      totalTrades += tradesExecuted;
      
      if (this.verbose) {
        console.log(`  Orders: ${ordersCreated}, Trades: ${tradesExecuted}, Duration: ${iterDuration}ms`);
      }
    }
    
    // Calculate statistics
    const avgDuration = totalDuration / iterations;
    const throughput = (messages.length * iterations * 1000) / totalDuration; // messages per second
    
    const avgLatency = latencies.length > 0 
      ? latencies.reduce((a, b) => a + b, 0) / latencies.length 
      : 0;
    
    const p99Latency = this.calculateP99(latencies);
    
    return {
      engineName: 'TypeScript',
      messagesProcessed: messages.length * iterations,
      ordersCreated: totalOrders,
      tradesExecuted: totalTrades,
      duration: totalDuration,
      throughput: throughput,
      avgLatency: avgLatency,
      p99Latency: p99Latency
    };
  }

  private calculateP99(latencies: number[]): number {
    if (latencies.length === 0) return 0;
    
    latencies.sort((a, b) => a - b);
    const index = Math.floor(latencies.length * 0.99);
    return latencies[Math.min(index, latencies.length - 1)];
  }

  private printResults(results: BenchmarkResult[]): void {
    console.log('\n=== Benchmark Results ===');
    console.log('╔═══════════════════╦════════════════╦═══════════╦═══════════╦════════════╦════════════╦════════════╗');
    console.log('║ Engine            ║ Messages/sec   ║ Orders    ║ Trades    ║ Avg Latency║ P99 Latency║ Duration   ║');
    console.log('╠═══════════════════╬════════════════╬═══════════╬═══════════╬════════════╬════════════╬════════════╣');
    
    for (const r of results) {
      console.log(
        `║ ${r.engineName.padEnd(17)} ║ ${r.throughput.toFixed(0).padStart(14)} ║ ${
          r.ordersCreated.toString().padStart(9)} ║ ${
          r.tradesExecuted.toString().padStart(9)} ║ ${
          (r.avgLatency + 'μs').padStart(10)} ║ ${
          (r.p99Latency + 'μs').padStart(10)} ║ ${
          (r.duration + 'ms').padStart(10)} ║`
      );
    }
    
    console.log('╚═══════════════════╩════════════════╩═══════════╩═══════════╩════════════╩════════════╩════════════╝');
  }
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  let dataFile: string | null = null;
  let iterations = 1;
  let warmup = 1000;
  let verbose = false;
  
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--data':
      case '-d':
        dataFile = args[++i];
        break;
      case '--iter':
      case '-i':
        iterations = parseInt(args[++i]);
        break;
      case '--warmup':
      case '-w':
        warmup = parseInt(args[++i]);
        break;
      case '--verbose':
      case '-v':
        verbose = true;
        break;
    }
  }
  
  const benchmark = new Benchmark(verbose);
  benchmark.run(dataFile, iterations, warmup).catch(console.error);
}