/**
 * LX CLMM Schema Routes
 *
 * Implements the CLMM (Concentrated Liquidity Market Maker) schema endpoints.
 * - GET /pool-info: Get concentrated liquidity pool information
 * - GET /positions-owned: Get user's concentrated liquidity positions
 * - POST /quote-position: Calculate position amounts for a price range
 * - POST /open-position: Open a new concentrated liquidity position
 * - POST /close-position: Close an existing position
 * - POST /collect-fees: Collect accumulated fees from a position
 */

import { FastifyInstance, FastifyPluginOptions } from 'fastify';
import { LXDex } from '../lxdex';
import {
  LXDexCLMMPoolInfoRequest,
  LXDexPositionsOwnedRequest,
  LXDexQuotePositionRequest,
  LXDexOpenPositionRequest,
  LXDexClosePositionRequest,
  LXDexCollectFeesRequest,
} from '../schemas';

export async function routes(
  fastify: FastifyInstance,
  _options: FastifyPluginOptions
): Promise<void> {
  /**
   * GET /pool-info
   *
   * Get information about a concentrated liquidity pool.
   * Returns current tick, price, liquidity, and fee tier.
   */
  fastify.get<{
    Querystring: LXDexCLMMPoolInfoRequest;
  }>(
    '/pool-info',
    {
      schema: {
        querystring: {
          type: 'object',
          required: ['tokenA', 'tokenB'],
          properties: {
            network: { type: 'string' },
            tokenA: { type: 'string' },
            tokenB: { type: 'string' },
            fee: { type: 'number' },
          },
        },
        response: {
          200: {
            type: 'object',
            properties: {
              pools: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    address: { type: 'string' },
                    tokenA: { type: 'object' },
                    tokenB: { type: 'object' },
                    fee: { type: 'number' },
                    tickSpacing: { type: 'number' },
                    currentTick: { type: 'number' },
                    currentPrice: { type: 'string' },
                    liquidity: { type: 'string' },
                    volume24h: { type: 'string' },
                    tvlUSD: { type: 'string' },
                  },
                },
              },
            },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.query.network || 'mainnet');
      const result = await connector.getCLMMPoolInfo(request.query);
      return reply.send(result);
    }
  );

  /**
   * GET /positions-owned
   *
   * Get all concentrated liquidity positions owned by a wallet.
   */
  fastify.get<{
    Querystring: LXDexPositionsOwnedRequest;
  }>(
    '/positions-owned',
    {
      schema: {
        querystring: {
          type: 'object',
          required: ['walletAddress'],
          properties: {
            network: { type: 'string' },
            walletAddress: { type: 'string' },
          },
        },
        response: {
          200: {
            type: 'object',
            properties: {
              positions: {
                type: 'array',
                items: {
                  type: 'object',
                  properties: {
                    tokenId: { type: 'string' },
                    poolAddress: { type: 'string' },
                    tokenA: { type: 'object' },
                    tokenB: { type: 'object' },
                    tickLower: { type: 'number' },
                    tickUpper: { type: 'number' },
                    liquidity: { type: 'string' },
                    amountA: { type: 'string' },
                    amountB: { type: 'string' },
                    unclaimedFeesA: { type: 'string' },
                    unclaimedFeesB: { type: 'string' },
                    inRange: { type: 'boolean' },
                    valueUSD: { type: 'string' },
                  },
                },
              },
              totalValueUSD: { type: 'string' },
            },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.query.network || 'mainnet');
      const result = await connector.getPositionsOwned(request.query);
      return reply.send(result);
    }
  );

  /**
   * POST /quote-position
   *
   * Calculate the amounts needed for a position in a given price range.
   */
  fastify.post<{
    Body: LXDexQuotePositionRequest;
  }>(
    '/quote-position',
    {
      schema: {
        body: {
          type: 'object',
          required: ['tokenA', 'tokenB', 'fee', 'tickLower', 'tickUpper'],
          properties: {
            network: { type: 'string' },
            tokenA: { type: 'string' },
            tokenB: { type: 'string' },
            fee: { type: 'number' },
            tickLower: { type: 'number' },
            tickUpper: { type: 'number' },
            amountA: { type: 'string' },
            amountB: { type: 'string' },
          },
        },
        response: {
          200: {
            type: 'object',
            properties: {
              estimatedAmountA: { type: 'string' },
              estimatedAmountB: { type: 'string' },
              estimatedLiquidity: { type: 'string' },
              priceRange: {
                type: 'object',
                properties: {
                  lower: { type: 'string' },
                  upper: { type: 'string' },
                  current: { type: 'string' },
                },
              },
              inRange: { type: 'boolean' },
            },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.body.network || 'mainnet');
      const result = await connector.quotePosition(request.body);
      return reply.send(result);
    }
  );

  /**
   * POST /open-position
   *
   * Open a new concentrated liquidity position.
   */
  fastify.post<{
    Body: LXDexOpenPositionRequest;
  }>(
    '/open-position',
    {
      schema: {
        body: {
          type: 'object',
          required: [
            'walletAddress',
            'tokenA',
            'tokenB',
            'fee',
            'tickLower',
            'tickUpper',
            'amountA',
            'amountB',
          ],
          properties: {
            network: { type: 'string' },
            walletAddress: { type: 'string' },
            tokenA: { type: 'string' },
            tokenB: { type: 'string' },
            fee: { type: 'number' },
            tickLower: { type: 'number' },
            tickUpper: { type: 'number' },
            amountA: { type: 'string' },
            amountB: { type: 'string' },
            slippagePct: { type: 'number' },
          },
        },
        response: {
          200: {
            type: 'object',
            properties: {
              txHash: { type: 'string' },
              status: { type: 'string' },
              tokenId: { type: 'string' },
              liquidity: { type: 'string' },
              amountA: { type: 'string' },
              amountB: { type: 'string' },
            },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.body.network || 'mainnet');
      const result = await connector.openPosition(request.body);
      return reply.send(result);
    }
  );

  /**
   * POST /close-position
   *
   * Close an existing concentrated liquidity position.
   * This withdraws all liquidity and collects fees.
   */
  fastify.post<{
    Body: LXDexClosePositionRequest;
  }>(
    '/close-position',
    {
      schema: {
        body: {
          type: 'object',
          required: ['walletAddress', 'tokenId'],
          properties: {
            network: { type: 'string' },
            walletAddress: { type: 'string' },
            tokenId: { type: 'string' },
            slippagePct: { type: 'number' },
          },
        },
        response: {
          200: {
            type: 'object',
            properties: {
              txHash: { type: 'string' },
              status: { type: 'string' },
              amountA: { type: 'string' },
              amountB: { type: 'string' },
              feesCollectedA: { type: 'string' },
              feesCollectedB: { type: 'string' },
            },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.body.network || 'mainnet');
      const result = await connector.closePosition(request.body);
      return reply.send(result);
    }
  );

  /**
   * POST /collect-fees
   *
   * Collect accumulated trading fees from a position.
   */
  fastify.post<{
    Body: LXDexCollectFeesRequest;
  }>(
    '/collect-fees',
    {
      schema: {
        body: {
          type: 'object',
          required: ['walletAddress', 'tokenId'],
          properties: {
            network: { type: 'string' },
            walletAddress: { type: 'string' },
            tokenId: { type: 'string' },
          },
        },
        response: {
          200: {
            type: 'object',
            properties: {
              txHash: { type: 'string' },
              status: { type: 'string' },
              feesCollectedA: { type: 'string' },
              feesCollectedB: { type: 'string' },
            },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.body.network || 'mainnet');
      const result = await connector.collectFees(request.body);
      return reply.send(result);
    }
  );
}

export default routes;
