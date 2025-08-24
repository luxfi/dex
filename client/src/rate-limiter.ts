/**
 * Rate Limiter for Lux Exchange API
 */

import { RateLimitConfig } from './types';

export class RateLimiter {
  private config: RateLimitConfig;
  private requestTokens: number;
  private orderTokens: number;
  private lastRefill: number;

  constructor(config?: RateLimitConfig) {
    this.config = config || {
      maxRequestsPerSecond: 20,
      maxOrdersPerSecond: 10,
      burstCapacity: 100
    };
    
    this.requestTokens = this.config.burstCapacity;
    this.orderTokens = this.config.maxOrdersPerSecond * 10;
    this.lastRefill = Date.now();
  }

  async checkRequestLimit(): Promise<void> {
    await this.checkLimit('request');
  }

  async checkOrderLimit(): Promise<void> {
    await this.checkLimit('order');
  }

  private async checkLimit(type: 'request' | 'order'): Promise<void> {
    this.refillTokens();
    
    const tokens = type === 'request' ? this.requestTokens : this.orderTokens;
    const required = 1;
    
    if (tokens < required) {
      const waitTime = this.calculateWaitTime(type);
      await this.sleep(waitTime);
      return this.checkLimit(type);
    }
    
    if (type === 'request') {
      this.requestTokens -= required;
    } else {
      this.orderTokens -= required;
    }
  }

  private refillTokens(): void {
    const now = Date.now();
    const elapsed = (now - this.lastRefill) / 1000;
    
    if (elapsed > 0) {
      const requestRefill = elapsed * this.config.maxRequestsPerSecond;
      const orderRefill = elapsed * this.config.maxOrdersPerSecond;
      
      this.requestTokens = Math.min(
        this.requestTokens + requestRefill,
        this.config.burstCapacity
      );
      
      this.orderTokens = Math.min(
        this.orderTokens + orderRefill,
        this.config.maxOrdersPerSecond * 10
      );
      
      this.lastRefill = now;
    }
  }

  private calculateWaitTime(type: 'request' | 'order'): number {
    const rate = type === 'request' 
      ? this.config.maxRequestsPerSecond 
      : this.config.maxOrdersPerSecond;
    
    return Math.ceil(1000 / rate);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  reset(): void {
    this.requestTokens = this.config.burstCapacity;
    this.orderTokens = this.config.maxOrdersPerSecond * 10;
    this.lastRefill = Date.now();
  }
}