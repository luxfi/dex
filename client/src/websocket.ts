/**
 * WebSocket Manager for Lux Exchange
 */

import { EventEmitter } from 'eventemitter3';
import { WSEventType, LXClientConfig } from './types';

export class WebSocketManager extends EventEmitter {
  private ws: WebSocket | null = null;
  private url: string;
  private config: LXClientConfig;
  private reconnectAttempts: number = 0;
  private subscriptions: Map<string, Set<string>> = new Map();
  private pingInterval: NodeJS.Timeout | null = null;
  private isClosing: boolean = false;

  constructor(url: string, config: LXClientConfig) {
    super();
    this.url = url;
    this.config = config;
  }

  async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          this.isClosing = false;
          this.emit(WSEventType.CONNECTED);
          this.startPing();
          this.resubscribe();
          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onerror = (error) => {
          this.emit(WSEventType.ERROR, error);
          reject(error);
        };

        this.ws.onclose = () => {
          this.stopPing();
          this.emit(WSEventType.DISCONNECTED);
          
          if (!this.isClosing && this.config.reconnect) {
            this.reconnect();
          }
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  async disconnect(): Promise<void> {
    this.isClosing = true;
    this.stopPing();
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  private reconnect(): void {
    if (this.reconnectAttempts >= (this.config.maxReconnectAttempts || 10)) {
      this.emit(WSEventType.ERROR, new Error('Max reconnection attempts reached'));
      return;
    }

    this.reconnectAttempts++;
    const delay = (this.config.reconnectDelay || 1000) * Math.pow(2, this.reconnectAttempts - 1);
    
    setTimeout(() => {
      this.connect().catch((error) => {
        console.error('Reconnection failed:', error);
      });
    }, delay);
  }

  private startPing(): void {
    this.pingInterval = setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.send({ type: 'ping' });
      }
    }, 30000);
  }

  private stopPing(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  private handleMessage(data: string): void {
    try {
      const message = JSON.parse(data);
      
      switch (message.type) {
        case 'pong':
          // Heartbeat response
          break;
        
        case 'orderbook':
          this.emit(WSEventType.ORDERBOOK, message.data);
          break;
        
        case 'trade':
          this.emit(WSEventType.TRADE, message.data);
          break;
        
        case 'ticker':
          this.emit(WSEventType.TICKER, message.data);
          break;
        
        case 'order':
          this.emit(WSEventType.ORDER_UPDATE, message.data);
          break;
        
        case 'position':
          this.emit(WSEventType.POSITION_UPDATE, message.data);
          break;
        
        case 'balance':
          this.emit(WSEventType.BALANCE_UPDATE, message.data);
          break;
        
        default:
          console.warn('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
    }
  }

  subscribe(channel: string, symbols: string[]): void {
    // Store subscription for reconnection
    const subs = this.subscriptions.get(channel) || new Set();
    symbols.forEach(symbol => subs.add(symbol));
    this.subscriptions.set(channel, subs);

    // Send subscription message
    this.send({
      type: 'subscribe',
      channel,
      symbols
    });
  }

  unsubscribe(channel: string, symbols: string[]): void {
    // Update stored subscriptions
    const subs = this.subscriptions.get(channel);
    if (subs) {
      symbols.forEach(symbol => subs.delete(symbol));
      if (subs.size === 0) {
        this.subscriptions.delete(channel);
      }
    }

    // Send unsubscribe message
    this.send({
      type: 'unsubscribe',
      channel,
      symbols
    });
  }

  private resubscribe(): void {
    // Resubscribe to all channels after reconnection
    for (const [channel, symbols] of this.subscriptions.entries()) {
      if (symbols.size > 0) {
        this.send({
          type: 'subscribe',
          channel,
          symbols: Array.from(symbols)
        });
      }
    }
  }

  private send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}