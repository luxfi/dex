/**
 * Cross-Chain Arbitrage Transports.
 *
 * 1. WARP (Lux Native)
 *    - Only works WITHIN Lux ecosystem (between subnets)
 *    - Sub-second message delivery
 *    - Use for: LX DEX <-> LX AMM <-> Other Lux subnets
 *    - Cannot reach external chains
 *
 * 2. TELEPORT (EVM Bridge)
 *    - Works with ANY EVM-compatible chain
 *    - Lux <-> Ethereum, BSC, Arbitrum, Polygon, etc.
 *    - ~30 second finality (depends on source chain)
 *    - Uses validator attestations
 *
 * 3. CEX API
 *    - No bridging needed - just API calls
 *    - Sub-second execution
 *    - Settlement via withdraw/deposit (slow but doesn't block arb)
 *
 * 4. FOR OMNICHAIN ARBITRAGE:
 *    - Lux internal: Warp (instant)
 *    - External EVM: Teleport (~30s)
 *    - CEX: Direct API (instant trade, later settle)
 */

import { Decimal } from 'decimal.js';
import type {
  ChainType,
  CrossChainConfig,
  CrossChainTransport,
  UnifiedOpportunity,
} from './types.js';

/**
 * Warp client interface for Lux-native messaging.
 */
export interface WarpClient {
  /** Send a Warp message to another Lux subnet */
  sendMessage(destSubnet: string, payload: Uint8Array): Promise<string>;
  /** Receive a Warp message */
  receiveMessage(messageId: string): Promise<Uint8Array>;
  /** Get this subnet's ID */
  getBlockchainId(): string;
}

/**
 * Teleport client interface for EVM bridging.
 */
export interface TeleportClient {
  /** Bridge assets to another EVM chain */
  bridge(destChain: string, token: string, amount: Decimal): Promise<string>;
  /** Get bridge transaction status */
  getBridgeStatus(txId: string): Promise<BridgeStatus>;
  /** Estimate bridge fee */
  estimateBridgeFee(destChain: string, token: string, amount: Decimal): Promise<Decimal>;
}

export interface BridgeStatus {
  txId: string;
  status: 'pending' | 'confirming' | 'completed' | 'failed';
  sourceChain: string;
  destChain: string;
  amount: Decimal;
  fee: Decimal;
  sourceTx: string;
  destTx?: string;
  timestamp: number;
}

/**
 * Cross-chain router for determining optimal transport.
 */
export class CrossChainRouter {
  private _warp?: WarpClient;
  private _teleport?: TeleportClient;

  constructor(public readonly config: CrossChainConfig) {}

  /**
   * Set the Warp client.
   */
  setWarpClient(client: WarpClient): void {
    this._warp = client;
  }

  /**
   * Set the Teleport client.
   */
  setTeleportClient(client: TeleportClient): void {
    this._teleport = client;
  }

  /**
   * Get the Warp client.
   */
  get warp(): WarpClient | undefined {
    return this._warp;
  }

  /**
   * Get the Teleport client.
   */
  get teleport(): TeleportClient | undefined {
    return this._teleport;
  }

  /**
   * Determine the best transport between two chains.
   */
  determineTransport(sourceChain: string, destChain: string): CrossChainTransport {
    const src = this.config.chains.get(sourceChain);
    const dst = this.config.chains.get(destChain);

    // Same chain = direct
    if (sourceChain === destChain) {
      return 'direct' as CrossChainTransport;
    }

    // CEX = API
    if (src?.chainType === ('cex' as ChainType) || dst?.chainType === ('cex' as ChainType)) {
      return 'cex_api' as CrossChainTransport;
    }

    // Both Lux subnets = Warp (fastest)
    if (
      src?.chainType === ('lux_subnet' as ChainType) &&
      dst?.chainType === ('lux_subnet' as ChainType)
    ) {
      if (src.warpSupported && dst.warpSupported && this.config.warpEnabled) {
        return 'warp' as CrossChainTransport;
      }
    }

    // Both EVM or mixed = Teleport
    if (src?.teleportSupported && dst?.teleportSupported && this.config.teleportEnabled) {
      return 'teleport' as CrossChainTransport;
    }

    // No viable transport
    return '' as CrossChainTransport;
  }

  /**
   * Estimate latency for cross-chain message.
   */
  estimateLatency(sourceChain: string, destChain: string): number {
    const transport = this.determineTransport(sourceChain, destChain);

    switch (transport) {
      case 'direct':
        return 0;
      case 'warp':
        return 500; // Sub-second
      case 'cex_api':
        return 100; // API call
      case 'teleport': {
        const src = this.config.chains.get(sourceChain);
        return (src?.finalityMs ?? 0) + 10000; // Finality + processing
      }
      default:
        return 3600000; // Unknown/unsupported (1 hour)
    }
  }

  /**
   * Estimate cost for cross-chain transfer.
   */
  async estimateCost(
    sourceChain: string,
    destChain: string,
    token: string,
    amount: Decimal
  ): Promise<Decimal> {
    const transport = this.determineTransport(sourceChain, destChain);

    switch (transport) {
      case 'direct':
        return new Decimal(0);
      case 'warp':
        return new Decimal(0.001); // Nearly free
      case 'cex_api':
        return new Decimal(0); // No bridge cost
      case 'teleport':
        if (this.teleport) {
          return this.teleport.estimateBridgeFee(destChain, token, amount);
        }
        return new Decimal(1.0); // Estimate $1
      default:
        return new Decimal(0);
    }
  }

  /**
   * Get chain ID from venue name.
   */
  venueToChain(venue: string): string {
    for (const [chainId, info] of this.config.chains) {
      if (info.venues.includes(venue)) {
        return chainId;
      }
    }
    return venue; // Fallback to venue name
  }

  /**
   * Enhance an opportunity with routing information.
   */
  async enhanceOpportunity(
    opp: UnifiedOpportunity
  ): Promise<EnhancedOpportunity> {
    const buyChain = this.venueToChain(opp.buyVenue);
    const sellChain = this.venueToChain(opp.sellVenue);

    const transport = this.determineTransport(buyChain, sellChain);
    const estimatedLatency = this.estimateLatency(buyChain, sellChain);
    const bridgeCost = await this.estimateCost(
      buyChain,
      sellChain,
      opp.symbol,
      opp.maxSize
    );

    return {
      ...opp,
      transport,
      estimatedLatency,
      bridgeCost,
      adjustedNetProfit: opp.netProfit.minus(bridgeCost),
    };
  }
}

export interface EnhancedOpportunity extends UnifiedOpportunity {
  transport: CrossChainTransport;
  estimatedLatency: number;
  bridgeCost: Decimal;
  adjustedNetProfit: Decimal;
}
