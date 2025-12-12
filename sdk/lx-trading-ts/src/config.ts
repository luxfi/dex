/**
 * Configuration for LX Trading SDK.
 */

import { Decimal } from 'decimal.js';

// =============================================================================
// General Config
// =============================================================================

export interface GeneralConfig {
  logLevel: 'debug' | 'info' | 'warn' | 'error';
  timeoutMs: number;
  smartRouting: boolean;
  venuePriority: string[];
  minImprovementBps: number;
}

function defaultGeneralConfig(): GeneralConfig {
  return {
    logLevel: 'info',
    timeoutMs: 30000,
    smartRouting: true,
    venuePriority: [],
    minImprovementBps: 5,
  };
}

// =============================================================================
// Risk Config
// =============================================================================

export interface RiskConfig {
  enabled: boolean;
  maxPositionSize: Decimal;
  maxOrderSize: Decimal;
  maxDailyLoss: Decimal;
  maxOpenOrders: number;
  killSwitchEnabled: boolean;
  positionLimits: Map<string, Decimal>;
}

function defaultRiskConfig(): RiskConfig {
  return {
    enabled: true,
    maxPositionSize: new Decimal(0),
    maxOrderSize: new Decimal(0),
    maxDailyLoss: new Decimal(0),
    maxOpenOrders: 100,
    killSwitchEnabled: false,
    positionLimits: new Map(),
  };
}

// =============================================================================
// Native Venue Config
// =============================================================================

export class NativeVenueConfig {
  venueType: 'dex' | 'amm' = 'dex';
  apiUrl = '';
  wsUrl?: string;
  apiKey?: string;
  apiSecret?: string;
  walletAddress?: string;
  privateKey?: string;
  network = 'mainnet';
  chainId = 96369;
  streaming = true;
  makerFee?: Decimal;
  takerFee?: Decimal;

  private constructor() {}

  static lxDex(apiUrl: string): NativeVenueConfig {
    const cfg = new NativeVenueConfig();
    cfg.venueType = 'dex';
    cfg.apiUrl = apiUrl;
    return cfg;
  }

  static lxAmm(apiUrl: string): NativeVenueConfig {
    const cfg = new NativeVenueConfig();
    cfg.venueType = 'amm';
    cfg.apiUrl = apiUrl;
    return cfg;
  }

  withCredentials(apiKey: string, apiSecret: string): this {
    this.apiKey = apiKey;
    this.apiSecret = apiSecret;
    return this;
  }

  withWallet(address: string, privateKey: string): this {
    this.walletAddress = address;
    this.privateKey = privateKey;
    return this;
  }

  testnet(): this {
    this.network = 'testnet';
    this.chainId = 8888;
    return this;
  }
}

// =============================================================================
// CCXT Config
// =============================================================================

export class CcxtConfig {
  exchangeId = '';
  apiKey?: string;
  apiSecret?: string;
  password?: string;
  sandbox = false;
  rateLimit = true;
  options: Record<string, unknown> = {};

  private constructor() {}

  static create(exchangeId: string): CcxtConfig {
    const cfg = new CcxtConfig();
    cfg.exchangeId = exchangeId;
    return cfg;
  }

  withCredentials(apiKey: string, apiSecret: string): this {
    this.apiKey = apiKey;
    this.apiSecret = apiSecret;
    return this;
  }

  withPassword(password: string): this {
    this.password = password;
    return this;
  }

  withSandbox(): this {
    this.sandbox = true;
    return this;
  }
}

// =============================================================================
// Hummingbot Config
// =============================================================================

export class HummingbotConfig {
  host = 'localhost';
  port = 15888;
  https = false;
  connector = '';
  chain = 'lux';
  network = 'mainnet';
  walletAddress?: string;

  private constructor() {}

  static create(connector: string): HummingbotConfig {
    const cfg = new HummingbotConfig();
    cfg.connector = connector;
    return cfg;
  }

  withWallet(address: string): this {
    this.walletAddress = address;
    return this;
  }

  withEndpoint(host: string, port: number): this {
    this.host = host;
    this.port = port;
    return this;
  }

  withHttps(): this {
    this.https = true;
    return this;
  }

  get baseUrl(): string {
    const scheme = this.https ? 'https' : 'http';
    return `${scheme}://${this.host}:${this.port}`;
  }
}

// =============================================================================
// Main Config
// =============================================================================

export class Config {
  general: GeneralConfig;
  risk: RiskConfig;
  native: Map<string, NativeVenueConfig> = new Map();
  ccxt: Map<string, CcxtConfig> = new Map();
  hummingbot: Map<string, HummingbotConfig> = new Map();

  constructor() {
    this.general = defaultGeneralConfig();
    this.risk = defaultRiskConfig();
  }

  static fromObject(data: ConfigData): Config {
    const config = new Config();

    // General
    if (data.general) {
      config.general = {
        logLevel: data.general.log_level ?? 'info',
        timeoutMs: data.general.timeout_ms ?? 30000,
        smartRouting: data.general.smart_routing ?? true,
        venuePriority: data.general.venue_priority ?? [],
        minImprovementBps: data.general.min_improvement_bps ?? 5,
      };
    }

    // Risk
    if (data.risk) {
      const posLimits = new Map<string, Decimal>();
      if (data.risk.position_limits) {
        for (const [k, v] of Object.entries(data.risk.position_limits)) {
          posLimits.set(k, new Decimal(v));
        }
      }

      config.risk = {
        enabled: data.risk.enabled ?? true,
        maxPositionSize: new Decimal(data.risk.max_position_size ?? 0),
        maxOrderSize: new Decimal(data.risk.max_order_size ?? 0),
        maxDailyLoss: new Decimal(data.risk.max_daily_loss ?? 0),
        maxOpenOrders: data.risk.max_open_orders ?? 100,
        killSwitchEnabled: data.risk.kill_switch_enabled ?? false,
        positionLimits: posLimits,
      };
    }

    // Native venues
    if (data.native) {
      for (const [name, cfg] of Object.entries(data.native)) {
        const native =
          cfg.venue_type === 'amm'
            ? NativeVenueConfig.lxAmm(cfg.api_url ?? '')
            : NativeVenueConfig.lxDex(cfg.api_url ?? '');

        native.wsUrl = cfg.ws_url;
        native.apiKey = cfg.api_key;
        native.apiSecret = cfg.api_secret;
        native.walletAddress = cfg.wallet_address;
        native.privateKey = cfg.private_key;
        native.network = cfg.network ?? 'mainnet';
        native.chainId = cfg.chain_id ?? 96369;
        native.streaming = cfg.streaming ?? true;

        config.native.set(name, native);
      }
    }

    // CCXT exchanges
    if (data.ccxt) {
      for (const [name, cfg] of Object.entries(data.ccxt)) {
        const ccxt = CcxtConfig.create(cfg.exchange_id ?? name);
        ccxt.apiKey = cfg.api_key;
        ccxt.apiSecret = cfg.api_secret;
        ccxt.password = cfg.password;
        ccxt.sandbox = cfg.sandbox ?? false;
        ccxt.rateLimit = cfg.rate_limit ?? true;
        ccxt.options = cfg.options ?? {};

        config.ccxt.set(name, ccxt);
      }
    }

    // Hummingbot gateways
    if (data.hummingbot) {
      for (const [name, cfg] of Object.entries(data.hummingbot)) {
        const hb = HummingbotConfig.create(cfg.connector ?? '');
        hb.host = cfg.host ?? 'localhost';
        hb.port = cfg.port ?? 15888;
        hb.https = cfg.https ?? false;
        hb.chain = cfg.chain ?? 'lux';
        hb.network = cfg.network ?? 'mainnet';
        hb.walletAddress = cfg.wallet_address;

        config.hummingbot.set(name, hb);
      }
    }

    return config;
  }

  withNative(name: string, cfg: NativeVenueConfig): this {
    this.native.set(name, cfg);
    return this;
  }

  withCcxt(name: string, cfg: CcxtConfig): this {
    this.ccxt.set(name, cfg);
    return this;
  }

  withHummingbot(name: string, cfg: HummingbotConfig): this {
    this.hummingbot.set(name, cfg);
    return this;
  }

  withSmartRouting(enabled: boolean): this {
    this.general.smartRouting = enabled;
    return this;
  }

  withVenuePriority(venues: string[]): this {
    this.general.venuePriority = venues;
    return this;
  }

  withRisk(risk: Partial<RiskConfig>): this {
    this.risk = { ...this.risk, ...risk };
    return this;
  }
}

// =============================================================================
// Config Data Types (for parsing)
// =============================================================================

export interface ConfigData {
  general?: {
    log_level?: 'debug' | 'info' | 'warn' | 'error';
    timeout_ms?: number;
    smart_routing?: boolean;
    venue_priority?: string[];
    min_improvement_bps?: number;
  };
  risk?: {
    enabled?: boolean;
    max_position_size?: number | string;
    max_order_size?: number | string;
    max_daily_loss?: number | string;
    max_open_orders?: number;
    kill_switch_enabled?: boolean;
    position_limits?: Record<string, number | string>;
  };
  native?: Record<
    string,
    {
      venue_type?: 'dex' | 'amm';
      api_url?: string;
      ws_url?: string;
      api_key?: string;
      api_secret?: string;
      wallet_address?: string;
      private_key?: string;
      network?: string;
      chain_id?: number;
      streaming?: boolean;
    }
  >;
  ccxt?: Record<
    string,
    {
      exchange_id?: string;
      api_key?: string;
      api_secret?: string;
      password?: string;
      sandbox?: boolean;
      rate_limit?: boolean;
      options?: Record<string, unknown>;
    }
  >;
  hummingbot?: Record<
    string,
    {
      host?: string;
      port?: number;
      https?: boolean;
      connector?: string;
      chain?: string;
      network?: string;
      wallet_address?: string;
    }
  >;
}
