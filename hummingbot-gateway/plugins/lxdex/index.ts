/**
 * Hummingbot Gateway Plugin for LX DEX
 *
 * This plugin registers the LX DEX connector with Hummingbot Gateway.
 */

import { FastifyInstance, FastifyPluginOptions } from 'fastify';
import { lxdexRoutes } from '@lxdex/hummingbot-gateway-connector';

export interface LXDexPluginOptions extends FastifyPluginOptions {
  /** Custom prefix for routes (default: none) */
  prefix?: string;
}

/**
 * Register the LX DEX plugin with Hummingbot Gateway
 */
export async function lxdexPlugin(
  fastify: FastifyInstance,
  options: LXDexPluginOptions
): Promise<void> {
  // Register the LX DEX routes
  await fastify.register(lxdexRoutes, {
    prefix: options.prefix,
  });

  fastify.log.info('LX DEX plugin registered successfully');
}

export default lxdexPlugin;

// Re-export connector components for direct use
export {
  LXDex,
  LXDexConfig,
  getLXDexConfig,
  LX_DEX_NETWORKS,
  lxdexRoutes,
} from '@lxdex/hummingbot-gateway-connector';
