/**
 * LX Gateway Connector Routes
 *
 * Fastify route registration for all LX endpoints.
 * Implements Router, AMM, CLMM, and Order Book schemas.
 */

import { FastifyInstance, FastifyPluginOptions } from 'fastify';
import { LXDex } from './lxdex';
import * as routerRoutes from './router-routes';
import * as ammRoutes from './amm-routes';
import * as clmmRoutes from './clmm-routes';

export async function lxdexRoutes(
  fastify: FastifyInstance,
  _options: FastifyPluginOptions
): Promise<void> {
  // Health check
  fastify.get('/lxdex/health', async (_request, reply) => {
    const connector = LXDex.getInstance('mainnet');
    const ready = await connector.ready();
    return reply.send({
      status: ready ? 'healthy' : 'unhealthy',
      network: connector.getNetwork(),
      timestamp: Date.now(),
    });
  });

  // Router Schema Routes (Swaps)
  fastify.register(routerRoutes.routes, { prefix: '/lxdex/router' });

  // AMM Schema Routes (Liquidity Pools)
  fastify.register(ammRoutes.routes, { prefix: '/lxdex/amm' });

  // CLMM Schema Routes (Concentrated Liquidity)
  fastify.register(clmmRoutes.routes, { prefix: '/lxdex/clmm' });

  // Order Book Routes
  fastify.post('/lxdex/orderbook', async (request, reply) => {
    const { network, symbol, depth } = request.body as {
      network?: string;
      symbol: string;
      depth?: number;
    };

    const connector = LXDex.getInstance(network || 'mainnet');
    const result = await connector.getOrderBook({ symbol, depth });
    return reply.send(result);
  });

  // Place Order
  fastify.post('/lxdex/order', async (request, reply) => {
    const body = request.body as {
      network?: string;
      walletAddress: string;
      symbol: string;
      side: 'BUY' | 'SELL';
      type: 'LIMIT' | 'MARKET' | 'STOP' | 'STOP_LIMIT';
      price?: string;
      size: string;
      timeInForce?: 'GTC' | 'IOC' | 'FOK' | 'GTT';
      clientOrderId?: string;
    };

    const connector = LXDex.getInstance(body.network || 'mainnet');
    const result = await connector.placeOrder(body);
    return reply.send(result);
  });

  // Cancel Order
  fastify.delete('/lxdex/order/:orderId', async (request, reply) => {
    const { orderId } = request.params as { orderId: string };
    const { network, walletAddress } = request.query as {
      network?: string;
      walletAddress: string;
    };

    const connector = LXDex.getInstance(network || 'mainnet');
    const result = await connector.cancelOrder({ walletAddress, orderId });
    return reply.send(result);
  });

  // Get Orders
  fastify.get('/lxdex/orders', async (request, reply) => {
    const query = request.query as {
      network?: string;
      walletAddress: string;
      symbol?: string;
      status?: string;
      limit?: string;
    };

    const connector = LXDex.getInstance(query.network || 'mainnet');
    const result = await connector.getOrders({
      walletAddress: query.walletAddress,
      symbol: query.symbol,
      status: query.status?.split(','),
      limit: query.limit ? parseInt(query.limit) : undefined,
    });
    return reply.send(result);
  });

  // Trading Pairs
  fastify.get('/lxdex/pairs', async (request, reply) => {
    const { network } = request.query as { network?: string };
    const connector = LXDex.getInstance(network || 'mainnet');
    const pairs = await connector.getTradingPairs();
    return reply.send({ pairs });
  });

  // Token Info
  fastify.get('/lxdex/token/:token', async (request, reply) => {
    const { token } = request.params as { token: string };
    const { network } = request.query as { network?: string };

    const connector = LXDex.getInstance(network || 'mainnet');
    const tokenInfo = await connector.getToken(token);

    if (!tokenInfo) {
      return reply.status(404).send({ error: 'Token not found' });
    }

    return reply.send(tokenInfo);
  });
}

export default lxdexRoutes;
