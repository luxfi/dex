/**
 * LX-First Arbitrage Bot Example
 *
 * This bot uses the LX-First strategy where LX DEX prices are treated
 * as the "truth" (fastest venue with nanosecond updates, 200ms blocks).
 * Other venues are always stale by comparison.
 *
 * Arbitrage = exploiting stale venues before they catch up to LX prices.
 *
 * Cross-chain transport options:
 * - Warp: For Lux subnet communication (<500ms)
 * - Teleport: For EVM chain bridging (~30s)
 * - CEX API: Direct trading (instant)
 *
 * NO SMART CONTRACTS - just coordinated trades through unified SDK.
 */

import { Decimal } from 'decimal.js'
import {
  LxDex,
  LxFirstArbitrage,
  LxFirstConfig,
  LxFirstOpportunity,
  LxPrice,
  VenuePrice,
  UnifiedArbitrage,
  UnifiedArbConfig,
  UnifiedOpportunity,
  Scanner,
  ScannerConfig,
  ArbitrageOpportunity,
  CrossChainRouter,
  CrossChainConfig,
  CrossChainTransport,
  ChainType,
} from '@luxfi/dex-sdk'

// ============================================
// Configuration
// ============================================

const LX_DEX_ENDPOINT = process.env.LX_DEX_ENDPOINT || 'wss://dex.lux.network/ws'
const LX_API_KEY = process.env.LX_API_KEY

// ============================================
// Arbitrage Bot Class
// ============================================

class ArbitrageBot {
  private dex: LxDex | null = null
  private lxFirst: LxFirstArbitrage | null = null
  private unified: UnifiedArbitrage | null = null
  private scanner: Scanner | null = null
  private router: CrossChainRouter | null = null
  private running = false
  private totalOpportunities = 0
  private totalExecutions = 0
  private totalPnl = new Decimal(0)

  async start(): Promise<void> {
    console.log('='.repeat(60))
    console.log('LX-FIRST ARBITRAGE BOT')
    console.log('='.repeat(60))
    console.log()

    // Initialize DEX client
    this.dex = new LxDex({
      endpoint: LX_DEX_ENDPOINT,
      apiKey: LX_API_KEY,
    })
    await this.dex.connect()
    console.log('[OK] Connected to LX DEX')

    // Initialize LX-First strategy
    const lxConfig: LxFirstConfig = {
      maxStalenessMs: 2000,
      minDivergenceBps: new Decimal(10),
      minProfit: new Decimal(5),
      maxPositionSize: new Decimal(10000),
      symbols: ['BTC-USDC', 'ETH-USDC', 'LUX-USDC'],
      venueLatencies: {
        binance: 50,
        mexc: 100,
        okx: 80,
        uniswap: 12000,
        pancakeswap: 3000,
      },
    }
    this.lxFirst = new LxFirstArbitrage(lxConfig)
    this.lxFirst.onOpportunity((opp) => this.onLxFirstOpportunity(opp))
    console.log('[OK] LX-First strategy initialized')

    // Initialize Unified Arbitrage
    const unifiedConfig: UnifiedArbConfig = {
      minSpreadBps: new Decimal(10),
      minProfit: new Decimal(5),
      maxPositionSize: new Decimal(10000),
      maxTotalExposure: new Decimal(100000),
      symbols: ['BTC-USDC', 'ETH-USDC', 'LUX-USDC'],
      venuePriority: ['lx_dex', 'binance', 'mexc', 'lx_amm'],
      scanIntervalMs: 100,
      executeTimeoutMs: 5000,
      maxDailyLoss: new Decimal(1000),
      maxTradesPerDay: 100,
    }
    this.unified = new UnifiedArbitrage(this.dex, unifiedConfig)
    this.unified.onOpportunity((opp) => this.onUnifiedOpportunity(opp))
    console.log('[OK] Unified arbitrage initialized')

    // Initialize Scanner
    const scannerConfig: ScannerConfig = {
      minSpreadBps: new Decimal(10),
      minProfitUsd: new Decimal(10),
      maxPriceAgeMs: 5000,
      symbols: ['BTC', 'ETH', 'LUX', 'SOL', 'AVAX'],
      chainIds: ['lux', 'ethereum', 'bsc', 'arbitrum', 'polygon'],
      scanIntervalMs: 100,
      maxConcurrency: 50,
    }
    this.scanner = new Scanner(scannerConfig)
    this.scanner.onOpportunity((opp) => this.onScannerOpportunity(opp))
    console.log('[OK] Scanner initialized')

    // Initialize Cross-Chain Router
    const routerConfig: CrossChainConfig = {
      warpEnabled: true,
      teleportEnabled: true,
      chains: {
        lux_mainnet: {
          chainId: 'lux_mainnet',
          name: 'Lux Mainnet',
          chainType: ChainType.LUX_SUBNET,
          blockTimeMs: 400,
          finalityMs: 400,
          warpSupported: true,
          teleportSupported: true,
          venues: ['lx_dex', 'lx_amm'],
        },
        ethereum: {
          chainId: '1',
          name: 'Ethereum',
          chainType: ChainType.EVM,
          blockTimeMs: 12000,
          finalityMs: 900000,
          warpSupported: false,
          teleportSupported: true,
          venues: ['uniswap', 'sushiswap'],
        },
        bsc: {
          chainId: '56',
          name: 'BNB Smart Chain',
          chainType: ChainType.EVM,
          blockTimeMs: 3000,
          finalityMs: 45000,
          warpSupported: false,
          teleportSupported: true,
          venues: ['pancakeswap'],
        },
        binance: {
          chainId: 'binance',
          name: 'Binance',
          chainType: ChainType.CEX,
          blockTimeMs: 0,
          finalityMs: 0,
          warpSupported: false,
          teleportSupported: false,
          venues: ['binance'],
        },
      },
    }
    this.router = new CrossChainRouter(routerConfig)
    console.log('[OK] Cross-chain router initialized')

    // Start all systems
    this.lxFirst.start()
    await this.unified.start()
    await this.scanner.start()
    this.running = true

    console.log()
    console.log('='.repeat(60))
    console.log('BOT RUNNING - Press Ctrl+C to stop')
    console.log('='.repeat(60))
    console.log()
    console.log(`Monitoring symbols: ${lxConfig.symbols.join(', ')}`)
    console.log(`Min divergence: ${lxConfig.minDivergenceBps} bps`)
    console.log(`Min profit: $${lxConfig.minProfit}`)
    console.log()

    // Start price feed simulation
    this.simulatePriceFeeds()

    // Start stats reporter
    this.reportStats()
  }

  async stop(): Promise<void> {
    console.log('\nShutting down...')
    this.running = false

    this.lxFirst?.stop()
    await this.unified?.stop()
    await this.scanner?.stop()

    this.printFinalStats()
  }

  private onLxFirstOpportunity(opp: LxFirstOpportunity): void {
    this.totalOpportunities++

    console.log()
    console.log('='.repeat(50))
    console.log('LX-FIRST OPPORTUNITY DETECTED')
    console.log('='.repeat(50))
    console.log(`Symbol:          ${opp.symbol}`)
    console.log(`LX Price:        $${opp.lxPrice.mid}`)
    console.log(`Stale Venue:     ${opp.staleVenue}`)
    console.log(`Stale Bid/Ask:   $${opp.stalePrice.bid} / $${opp.stalePrice.ask}`)
    console.log(`Staleness:       ${opp.staleness}ms`)
    console.log(`Side:            ${opp.side.toUpperCase()}`)
    console.log(`Divergence:      ${opp.divergenceBps} bps`)
    console.log(`Expected Profit: $${opp.expectedProfit}`)
    console.log(`Confidence:      ${(opp.confidence * 100).toFixed(1)}%`)
    console.log('='.repeat(50))

    // Execute if confidence is high enough
    if (opp.confidence > 0.8) {
      this.executeLxFirst(opp)
    }
  }

  private onUnifiedOpportunity(opp: UnifiedOpportunity): void {
    console.log(
      `[UNIFIED] ${opp.symbol}: Buy ${opp.buyVenue} @ $${opp.buyPrice} -> ` +
      `Sell ${opp.sellVenue} @ $${opp.sellPrice} | Net: $${opp.netProfit}`
    )
  }

  private onScannerOpportunity(opp: ArbitrageOpportunity): void {
    console.log(
      `[SCANNER] ${opp.type}: ${opp.buySource.venue} -> ` +
      `${opp.sellSource.venue} | Spread: ${opp.spreadBps} bps | ` +
      `Net PnL: $${opp.netPnl}`
    )
  }

  private async executeLxFirst(opp: LxFirstOpportunity): Promise<void> {
    try {
      console.log(`\n[EXECUTING] ${opp.id}...`)

      // Determine cross-chain transport
      const buyChain = this.router!.venueToChain(opp.staleVenue)
      const sellChain = 'lux_mainnet'
      const transport = this.router!.determineTransport(buyChain, sellChain)
      const latency = this.router!.estimateLatency(buyChain, sellChain)

      console.log(`  Transport: ${transport}`)
      console.log(`  Est. Latency: ${latency}ms`)

      if (opp.side === 'buy') {
        console.log(`  Buying on ${opp.staleVenue}...`)
        // In production: place actual order
        // const order = await cexClient.placeOrder(...)

        console.log('  Hedging on LX DEX...')
        // In production: place hedge order
        // const hedge = await this.dex.spot.sell(...)
      } else {
        console.log(`  Selling on ${opp.staleVenue}...`)
        // In production: place actual order

        console.log('  Hedging on LX DEX...')
        // In production: place hedge order
      }

      // Simulate successful execution
      this.totalExecutions++
      const profit = opp.expectedProfit.mul(0.8) // Simulate slippage
      this.totalPnl = this.totalPnl.add(profit)

      console.log(`[SUCCESS] Executed ${opp.id} | Profit: $${profit.toFixed(2)}`)
    } catch (error) {
      console.log(`[FAILED] ${opp.id}: ${error}`)
    }
  }

  private simulatePriceFeeds(): void {
    const basePrices: Record<string, Decimal> = {
      'BTC-USDC': new Decimal(50000),
      'ETH-USDC': new Decimal(3000),
      'LUX-USDC': new Decimal(25),
    }

    const interval = setInterval(() => {
      if (!this.running) {
        clearInterval(interval)
        return
      }

      for (const [symbol, base] of Object.entries(basePrices)) {
        // Simulate LX DEX price (the oracle)
        const lxMid = base.mul(1 + (Math.random() - 0.5) * 0.002)
        this.lxFirst?.updateLxPrice({
          symbol,
          bid: lxMid.mul(0.9999),
          ask: lxMid.mul(1.0001),
          mid: lxMid,
          timestamp: Date.now(),
          blockNum: Math.floor(Math.random() * 1000000) + 1000000,
        })

        // Simulate stale CEX prices
        for (const [venue, latency] of [['binance', 50], ['mexc', 100]] as const) {
          const divergence = (Math.random() - 0.5) * 0.004
          const venueMid = base.mul(1 + divergence)

          this.lxFirst?.updateVenuePrice({
            venue,
            symbol,
            bid: venueMid.mul(0.9998),
            ask: venueMid.mul(1.0002),
            timestamp: Date.now() - latency,
            latency,
            stale: false,
          })
        }
      }
    }, 100) // 10 updates per second
  }

  private reportStats(): void {
    const interval = setInterval(() => {
      if (!this.running) {
        clearInterval(interval)
        return
      }

      console.log()
      console.log('-'.repeat(40))
      console.log('STATS')
      console.log(`  Opportunities: ${this.totalOpportunities}`)
      console.log(`  Executions:    ${this.totalExecutions}`)
      console.log(`  Total PnL:     $${this.totalPnl.toFixed(2)}`)
      if (this.totalExecutions > 0) {
        const avgPnl = this.totalPnl.div(this.totalExecutions)
        console.log(`  Avg PnL:       $${avgPnl.toFixed(2)}`)
      }
      console.log('-'.repeat(40))
    }, 30000) // Every 30 seconds
  }

  private printFinalStats(): void {
    console.log()
    console.log('='.repeat(50))
    console.log('FINAL STATISTICS')
    console.log('='.repeat(50))
    console.log(`Total Opportunities: ${this.totalOpportunities}`)
    console.log(`Total Executions:    ${this.totalExecutions}`)
    console.log(`Total PnL:           $${this.totalPnl.toFixed(2)}`)
    if (this.totalExecutions > 0) {
      const winRate = (this.totalExecutions / this.totalOpportunities) * 100
      const avgPnl = this.totalPnl.div(this.totalExecutions)
      console.log(`Execution Rate:      ${winRate.toFixed(1)}%`)
      console.log(`Avg PnL per Trade:   $${avgPnl.toFixed(2)}`)
    }
    console.log('='.repeat(50))
  }
}

// ============================================
// Main Entry Point
// ============================================

async function main(): Promise<void> {
  const bot = new ArbitrageBot()

  // Handle Ctrl+C gracefully
  process.on('SIGINT', async () => {
    await bot.stop()
    process.exit(0)
  })

  process.on('SIGTERM', async () => {
    await bot.stop()
    process.exit(0)
  })

  try {
    await bot.start()

    // Keep running
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 1000))
    }
  } catch (error) {
    console.error('Fatal error:', error)
    await bot.stop()
    process.exit(1)
  }
}

main()
