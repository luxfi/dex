/**
 * LX DEX AMM Schema Routes
 *
 * Implements the AMM schema endpoints for liquidity pool operations.
 * - GET /pool-info: Get pool reserves and pricing
 * - GET /position-info: Get current liquidity position details
 * - POST /quote-liquidity: Calculate liquidity provision amounts
 * - POST /add-liquidity: Add liquidity to pool
 * - POST /remove-liquidity: Remove liquidity from pool
 */

import { FastifyInstance, FastifyPluginOptions } from 'fastify';
import { LXDex } from '../lxdex';
import {
  LXDexPoolInfoRequest,
  LXDexPositionInfoRequest,
  LXDexAddLiquidityRequest,
  LXDexRemoveLiquidityRequest,
} from '../schemas';

export async function routes(
  fastify: FastifyInstance,
  _options: FastifyPluginOptions
): Promise<void> {
  /**
   * GET /pool-info
   *
   * Get information about a liquidity pool.
   * Returns reserves, liquidity, fees, and pricing.
   */
  fastify.get<{
    Querystring: LXDexPoolInfoRequest;
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
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.query.network || 'mainnet');
      const result = await connector.getPoolInfo(request.query);
      return reply.send(result);
    }
  );

  /**
   * GET /position-info
   *
   * Get current LP position details for a wallet.
   */
  fastify.get<{
    Querystring: LXDexPositionInfoRequest;
  }>(
    '/position-info',
    {
      schema: {
        querystring: {
          type: 'object',
          required: ['walletAddress'],
          properties: {
            network: { type: 'string' },
            walletAddress: { type: 'string' },
            poolAddress: { type: 'string' },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.query.network || 'mainnet');
      const result = await connector.getPositionInfo(request.query);
      return reply.send(result);
    }
  );

  /**
   * POST /quote-liquidity
   *
   * Calculate optimal amounts for adding liquidity.
   * Returns estimated LP tokens and pool share.
   */
  fastify.post<{
    Body: {
      network?: string;
      tokenA: string;
      tokenB: string;
      amountA?: string;
      amountB?: string;
    };
  }>(
    '/quote-liquidity',
    {
      schema: {
        body: {
          type: 'object',
          required: ['tokenA', 'tokenB'],
          properties: {
            network: { type: 'string' },
            tokenA: { type: 'string' },
            tokenB: { type: 'string' },
            amountA: { type: 'string' },
            amountB: { type: 'string' },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.body.network || 'mainnet');

      // Get pool info to calculate quote
      const poolInfo = await connector.getPoolInfo({
        tokenA: request.body.tokenA,
        tokenB: request.body.tokenB,
      });

      if (!poolInfo.pools.length) {
        return reply.status(404).send({ error: 'Pool not found' });
      }

      const pool = poolInfo.pools[0];
      const reserveA = BigInt(pool.reserveA);
      const reserveB = BigInt(pool.reserveB);
      const totalLiquidity = BigInt(pool.totalLiquidity);

      let amountA: bigint;
      let amountB: bigint;
      let liquidityEstimate: bigint;

      if (request.body.amountA) {
        amountA = BigInt(request.body.amountA);
        amountB = (amountA * reserveB) / reserveA;
        liquidityEstimate = (amountA * totalLiquidity) / reserveA;
      } else if (request.body.amountB) {
        amountB = BigInt(request.body.amountB);
        amountA = (amountB * reserveA) / reserveB;
        liquidityEstimate = (amountB * totalLiquidity) / reserveB;
      } else {
        return reply.status(400).send({
          error: 'Either amountA or amountB must be provided',
        });
      }

      const sharePercent =
        Number((liquidityEstimate * 10000n) / (totalLiquidity + liquidityEstimate)) / 100;

      return reply.send({
        amountA: amountA.toString(),
        amountB: amountB.toString(),
        estimatedLiquidity: liquidityEstimate.toString(),
        sharePercent: sharePercent.toFixed(4),
        pool: {
          address: pool.address,
          fee: pool.fee,
        },
      });
    }
  );

  /**
   * POST /add-liquidity
   *
   * Add liquidity to a pool.
   */
  fastify.post<{
    Body: LXDexAddLiquidityRequest;
  }>(
    '/add-liquidity',
    {
      schema: {
        body: {
          type: 'object',
          required: ['walletAddress', 'tokenA', 'tokenB', 'amountA', 'amountB'],
          properties: {
            network: { type: 'string' },
            walletAddress: { type: 'string' },
            tokenA: { type: 'string' },
            tokenB: { type: 'string' },
            amountA: { type: 'string' },
            amountB: { type: 'string' },
            slippagePct: { type: 'number' },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.body.network || 'mainnet');
      const result = await connector.addLiquidity(request.body);
      return reply.send(result);
    }
  );

  /**
   * POST /remove-liquidity
   *
   * Remove liquidity from a pool.
   */
  fastify.post<{
    Body: LXDexRemoveLiquidityRequest;
  }>(
    '/remove-liquidity',
    {
      schema: {
        body: {
          type: 'object',
          required: ['walletAddress', 'poolAddress', 'liquidity'],
          properties: {
            network: { type: 'string' },
            walletAddress: { type: 'string' },
            poolAddress: { type: 'string' },
            liquidity: { type: 'string' },
            slippagePct: { type: 'number' },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.body.network || 'mainnet');
      const result = await connector.removeLiquidity(request.body);
      return reply.send(result);
    }
  );
}

export default routes;
