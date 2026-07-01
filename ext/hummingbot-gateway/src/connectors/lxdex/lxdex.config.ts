/**
 * LX Gateway Connector Configuration
 *
 * Configuration interface and loader for the LX connector.
 * Supports Router, AMM, CLMM, and OrderBook trading types.
 */

export interface AvailableNetworks {
  chain: string;
  networks: string[];
}

export interface LXDexConfig {
  // Network configuration
  chain: string;
  networks: string[];
  availableNetworks: AvailableNetworks[];

  // Trading parameters
  tradingTypes: string[];
  slippagePct: number;
  maxHops: number;

  // API configuration
  apiEndpoint: string;
  wsEndpoint: string;
  grpcEndpoint: string;

  // Rate limits
  requestsPerSecond: number;

  // Fees
  defaultFeeRate: number;

  // Timeouts (ms)
  orderTimeout: number;
  connectionTimeout: number;
}

/**
 * Default configuration values
 */
export const LXDexConfigDefaults: LXDexConfig = {
  chain: 'lux',
  networks: ['mainnet', 'testnet'],
  availableNetworks: [
    { chain: 'lux', networks: ['mainnet', 'testnet'] },
  ],
  tradingTypes: ['ROUTER', 'AMM', 'CLMM', 'OrderBook'],
  slippagePct: 0.5,
  maxHops: 4,
  apiEndpoint: 'https://api.dex.lux.network',
  wsEndpoint: 'wss://ws.dex.lux.network',
  grpcEndpoint: 'grpc.dex.lux.network:443',
  requestsPerSecond: 50,
  defaultFeeRate: 0.003, // 0.3% (30 bps)
  orderTimeout: 30000,
  connectionTimeout: 10000,
};

/**
 * Network-specific configurations
 */
export const LX_DEX_NETWORKS: Record<string, Partial<LXDexConfig>> = {
  mainnet: {
    apiEndpoint: 'https://api.dex.lux.network',
    wsEndpoint: 'wss://ws.dex.lux.network',
    grpcEndpoint: 'grpc.dex.lux.network:443',
  },
  testnet: {
    apiEndpoint: 'https://api.testnet.dex.lux.network',
    wsEndpoint: 'wss://ws.testnet.dex.lux.network',
    grpcEndpoint: 'grpc.testnet.dex.lux.network:443',
  },
};

/**
 * Get configuration for a specific network
 */
export function getLXDexConfig(
  network: string = 'mainnet',
  overrides?: Partial<LXDexConfig>
): LXDexConfig {
  const networkConfig = LX_DEX_NETWORKS[network] || LX_DEX_NETWORKS.mainnet;

  return {
    ...LXDexConfigDefaults,
    ...networkConfig,
    ...overrides,
  };
}

/**
 * Get network endpoints for a specific network
 */
export function getNetworkEndpoints(network: string): {
  api: string;
  ws: string;
  grpc: string;
} {
  const config = getLXDexConfig(network);
  return {
    api: config.apiEndpoint,
    ws: config.wsEndpoint,
    grpc: config.grpcEndpoint,
  };
}

/**
 * Validate configuration
 */
export function validateConfig(config: LXDexConfig): boolean {
  if (!config.apiEndpoint) {
    throw new Error('LXDex config: apiEndpoint is required');
  }
  if (!config.wsEndpoint) {
    throw new Error('LXDex config: wsEndpoint is required');
  }
  if (config.slippagePct < 0 || config.slippagePct > 100) {
    throw new Error('LXDex config: slippagePct must be between 0 and 100');
  }
  if (config.maxHops < 1 || config.maxHops > 10) {
    throw new Error('LXDex config: maxHops must be between 1 and 10');
  }
  return true;
}
