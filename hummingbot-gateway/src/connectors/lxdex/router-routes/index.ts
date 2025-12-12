/**
 * LX DEX Router Schema Routes
 *
 * Implements the Router schema endpoints for swap operations.
 * - GET /quote-swap: Get optimal swap quote
 * - POST /execute-swap: Execute swap directly
 * - POST /execute-quote: Execute pre-fetched quote
 */

import { FastifyInstance, FastifyPluginOptions } from 'fastify';
import { LXDex } from '../lxdex';
import {
  LXDexQuoteSwapRequest,
  LXDexExecuteSwapRequest,
  LXDexExecuteQuoteRequest,
  quoteSwapRequestSchema,
  executeSwapRequestSchema,
} from '../schemas';

export async function routes(
  fastify: FastifyInstance,
  _options: FastifyPluginOptions
): Promise<void> {
  /**
   * GET /quote-swap
   *
   * Get an optimal swap quote with routing details.
   * Returns the best route across all available liquidity sources.
   */
  fastify.get<{
    Querystring: LXDexQuoteSwapRequest;
  }>(
    '/quote-swap',
    {
      schema: {
        querystring: quoteSwapRequestSchema,
        response: {
          200: {
            type: 'object',
            properties: {
              quoteId: { type: 'string' },
              tokenIn: { type: 'object' },
              tokenOut: { type: 'object' },
              amountIn: { type: 'string' },
              amountOut: { type: 'string' },
              price: { type: 'string' },
              priceImpactPct: { type: 'string' },
              minAmountOut: { type: 'string' },
              maxAmountIn: { type: 'string' },
              route: { type: 'object' },
              estimatedGas: { type: 'string' },
              expiresAt: { type: 'number' },
            },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.query.network || 'mainnet');
      const quote = await connector.getQuote(request.query);
      return reply.send(quote);
    }
  );

  /**
   * POST /execute-swap
   *
   * Execute a swap directly without pre-fetching a quote.
   * The swap is executed at the best available price.
   */
  fastify.post<{
    Body: LXDexExecuteSwapRequest;
  }>(
    '/execute-swap',
    {
      schema: {
        body: executeSwapRequestSchema,
        response: {
          200: {
            type: 'object',
            properties: {
              txHash: { type: 'string' },
              status: { type: 'string' },
              tokenIn: { type: 'object' },
              tokenOut: { type: 'object' },
              amountIn: { type: 'string' },
              amountOut: { type: 'string' },
              price: { type: 'string' },
              fee: { type: 'string' },
              gasUsed: { type: 'string' },
              blockNumber: { type: 'number' },
            },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.body.network || 'mainnet');
      const result = await connector.executeSwap(request.body);
      return reply.send(result);
    }
  );

  /**
   * POST /execute-quote
   *
   * Execute a previously fetched quote.
   * The quote must not be expired.
   */
  fastify.post<{
    Body: LXDexExecuteQuoteRequest;
  }>(
    '/execute-quote',
    {
      schema: {
        body: {
          type: 'object',
          required: ['walletAddress', 'quoteId'],
          properties: {
            network: { type: 'string' },
            walletAddress: { type: 'string' },
            quoteId: { type: 'string' },
            gasPrice: { type: 'string' },
          },
        },
      },
    },
    async (request, reply) => {
      const connector = LXDex.getInstance(request.body.network || 'mainnet');
      const result = await connector.executeQuote(request.body);
      return reply.send(result);
    }
  );
}

export default routes;
