"""
LX AMM Connector

Hummingbot connector for LX DEX AMM (Automated Market Maker) operations.
Supports liquidity provision, swaps, and pool management.
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from hummingbot.connector.gateway.amm.gateway_amm_base import GatewayAmmBase
from hummingbot.connector.gateway.gateway_price_shim import GatewayPriceShim
from hummingbot.core.data_type.cancellation_result import CancellationResult
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.core.data_type.trade_fee import AddedToCostTradeFee, TokenAmount
from hummingbot.core.event.events import (
    BuyOrderCompletedEvent,
    BuyOrderCreatedEvent,
    MarketEvent,
    OrderFilledEvent,
    SellOrderCompletedEvent,
    SellOrderCreatedEvent,
)
from hummingbot.core.gateway.gateway_http_client import GatewayHttpClient
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.utils.async_utils import safe_ensure_future
from hummingbot.core.utils.tracking_nonce import get_tracking_nonce

logger = logging.getLogger(__name__)


class LxAmm(GatewayAmmBase):
    """
    LX AMM connector for liquidity pool operations.

    This connector enables:
    - Swaps via AMM pools
    - Liquidity provision (add/remove)
    - Pool information retrieval
    - Real-time price quotes
    """

    # Gateway connector name
    API_CALL_TIMEOUT = 30.0
    POLL_INTERVAL = 15.0
    UPDATE_BALANCE_INTERVAL = 30.0

    def __init__(
        self,
        client_config_map: "ClientConfigAdapter",
        connector_name: str,
        chain: str,
        network: str,
        wallet_address: str,
        trading_pairs: Optional[List[str]] = None,
        additional_spenders: Optional[List[str]] = None,
        trading_required: bool = True,
    ):
        """
        Initialize the LX AMM connector.

        Args:
            client_config_map: Hummingbot client configuration
            connector_name: Name of the connector (lx_amm)
            chain: Blockchain network (lux)
            network: Network name (mainnet/testnet)
            wallet_address: User's wallet address
            trading_pairs: List of trading pairs to track
            additional_spenders: Additional contract addresses for approvals
            trading_required: Whether trading is required
        """
        super().__init__(
            client_config_map=client_config_map,
            connector_name=connector_name,
            chain=chain,
            network=network,
            wallet_address=wallet_address,
            trading_pairs=trading_pairs or [],
            additional_spenders=additional_spenders or [],
            trading_required=trading_required,
        )

        self._connector_name = connector_name
        self._chain = chain
        self._network = network
        self._wallet_address = wallet_address
        self._trading_pairs = trading_pairs or []
        self._trading_required = trading_required

        self._last_poll_timestamp = 0
        self._last_balance_poll_timestamp = 0
        self._poll_notifier = asyncio.Event()
        self._status_polling_task: Optional[asyncio.Task] = None
        self._get_gas_estimate_task: Optional[asyncio.Task] = None

        # Caches
        self._allowances: Dict[str, Decimal] = {}
        self._pool_info_cache: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "lx_amm"

    @property
    def connector_name(self) -> str:
        return self._connector_name

    @property
    def chain(self) -> str:
        return self._chain

    @property
    def network(self) -> str:
        return self._network

    async def start_network(self):
        """Start network operations."""
        if self._trading_required:
            self._status_polling_task = safe_ensure_future(self._status_polling_loop())
        await self.update_balances()

    async def stop_network(self):
        """Stop network operations."""
        if self._status_polling_task is not None:
            self._status_polling_task.cancel()
            self._status_polling_task = None

    async def check_network(self) -> NetworkStatus:
        """Check network connectivity."""
        try:
            response = await self._get_gateway_instance().get_network_status(
                chain=self._chain,
                network=self._network,
            )
            if response.get("currentBlockNumber", 0) > 0:
                return NetworkStatus.CONNECTED
        except Exception as e:
            logger.warning(f"Network check failed: {e}")
        return NetworkStatus.NOT_CONNECTED

    async def update_balances(self):
        """Update wallet balances."""
        try:
            response = await self._get_gateway_instance().get_balances(
                chain=self._chain,
                network=self._network,
                address=self._wallet_address,
                token_symbols=list(self._get_relevant_tokens()),
            )

            self._account_available_balances.clear()
            self._account_balances.clear()

            for token, balance_str in response.get("balances", {}).items():
                balance = Decimal(str(balance_str))
                self._account_available_balances[token] = balance
                self._account_balances[token] = balance

        except Exception as e:
            logger.warning(f"Failed to update balances: {e}")

    def _get_relevant_tokens(self) -> set:
        """Get set of relevant token symbols for balance tracking."""
        tokens = set()
        for trading_pair in self._trading_pairs:
            base, quote = trading_pair.split("-")
            tokens.add(base)
            tokens.add(quote)
        return tokens

    async def _status_polling_loop(self):
        """Background loop for status polling."""
        while True:
            try:
                self._poll_notifier.clear()

                if self._last_poll_timestamp > 0:
                    await self._update_order_status()

                current_time = time.time()
                if current_time - self._last_balance_poll_timestamp > self.UPDATE_BALANCE_INTERVAL:
                    await self.update_balances()
                    self._last_balance_poll_timestamp = current_time

                self._last_poll_timestamp = current_time

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Status polling error: {e}")

            try:
                await asyncio.wait_for(
                    self._poll_notifier.wait(),
                    timeout=self.POLL_INTERVAL,
                )
            except asyncio.TimeoutError:
                pass

    async def get_quote_price(
        self,
        trading_pair: str,
        is_buy: bool,
        amount: Decimal,
    ) -> Optional[Decimal]:
        """
        Get a price quote for a swap.

        Args:
            trading_pair: Trading pair (e.g., "LUX-USDC")
            is_buy: True for buy, False for sell
            amount: Amount to trade

        Returns:
            Price quote or None if unavailable
        """
        try:
            base, quote = trading_pair.split("-")

            response = await self._get_gateway_instance().amm_quote(
                chain=self._chain,
                network=self._network,
                connector=self._connector_name,
                base_token=base,
                quote_token=quote,
                amount=str(amount),
                side="BUY" if is_buy else "SELL",
            )

            if "price" in response:
                return Decimal(str(response["price"]))

        except Exception as e:
            logger.warning(f"Failed to get quote for {trading_pair}: {e}")

        return None

    async def get_order_price(
        self,
        trading_pair: str,
        is_buy: bool,
        amount: Decimal,
    ) -> Decimal:
        """Get execution price for an order."""
        price = await self.get_quote_price(trading_pair, is_buy, amount)
        if price is None:
            raise ValueError(f"Unable to get price quote for {trading_pair}")
        return price

    def buy(
        self,
        trading_pair: str,
        amount: Decimal,
        order_type: OrderType,
        price: Decimal,
        **kwargs,
    ) -> str:
        """
        Place a buy order (swap).

        Args:
            trading_pair: Trading pair
            amount: Amount to buy
            order_type: Order type (only LIMIT supported for AMM)
            price: Maximum price (slippage protection)

        Returns:
            Order ID
        """
        order_id = self._create_order_id()
        safe_ensure_future(
            self._create_order(
                trade_type=TradeType.BUY,
                order_id=order_id,
                trading_pair=trading_pair,
                amount=amount,
                price=price,
            )
        )
        return order_id

    def sell(
        self,
        trading_pair: str,
        amount: Decimal,
        order_type: OrderType,
        price: Decimal,
        **kwargs,
    ) -> str:
        """
        Place a sell order (swap).

        Args:
            trading_pair: Trading pair
            amount: Amount to sell
            order_type: Order type (only LIMIT supported for AMM)
            price: Minimum price (slippage protection)

        Returns:
            Order ID
        """
        order_id = self._create_order_id()
        safe_ensure_future(
            self._create_order(
                trade_type=TradeType.SELL,
                order_id=order_id,
                trading_pair=trading_pair,
                amount=amount,
                price=price,
            )
        )
        return order_id

    def _create_order_id(self) -> str:
        """Generate a unique order ID."""
        return f"lx_amm_{get_tracking_nonce()}"

    async def _create_order(
        self,
        trade_type: TradeType,
        order_id: str,
        trading_pair: str,
        amount: Decimal,
        price: Decimal,
    ):
        """Execute an AMM swap order."""
        try:
            base, quote = trading_pair.split("-")
            is_buy = trade_type == TradeType.BUY

            # Emit order created event
            if is_buy:
                self.trigger_event(
                    MarketEvent.BuyOrderCreated,
                    BuyOrderCreatedEvent(
                        timestamp=time.time(),
                        type=OrderType.LIMIT,
                        trading_pair=trading_pair,
                        amount=amount,
                        price=price,
                        order_id=order_id,
                    ),
                )
            else:
                self.trigger_event(
                    MarketEvent.SellOrderCreated,
                    SellOrderCreatedEvent(
                        timestamp=time.time(),
                        type=OrderType.LIMIT,
                        trading_pair=trading_pair,
                        amount=amount,
                        price=price,
                        order_id=order_id,
                    ),
                )

            # Execute the swap via Gateway
            response = await self._get_gateway_instance().amm_trade(
                chain=self._chain,
                network=self._network,
                connector=self._connector_name,
                address=self._wallet_address,
                base_token=base,
                quote_token=quote,
                side="BUY" if is_buy else "SELL",
                amount=str(amount),
                limit_price=str(price),
            )

            tx_hash = response.get("txHash", "")
            executed_price = Decimal(str(response.get("price", price)))
            executed_amount = Decimal(str(response.get("amount", amount)))
            fee_amount = Decimal(str(response.get("fee", "0")))

            # Emit order filled event
            self.trigger_event(
                MarketEvent.OrderFilled,
                OrderFilledEvent(
                    timestamp=time.time(),
                    order_id=order_id,
                    trading_pair=trading_pair,
                    trade_type=trade_type,
                    order_type=OrderType.LIMIT,
                    price=executed_price,
                    amount=executed_amount,
                    trade_fee=AddedToCostTradeFee(
                        flat_fees=[TokenAmount(quote, fee_amount)]
                    ),
                    exchange_trade_id=tx_hash,
                ),
            )

            # Emit order completed event
            if is_buy:
                self.trigger_event(
                    MarketEvent.BuyOrderCompleted,
                    BuyOrderCompletedEvent(
                        timestamp=time.time(),
                        order_id=order_id,
                        base_asset=base,
                        quote_asset=quote,
                        base_asset_amount=executed_amount,
                        quote_asset_amount=executed_amount * executed_price,
                        order_type=OrderType.LIMIT,
                        exchange_order_id=tx_hash,
                    ),
                )
            else:
                self.trigger_event(
                    MarketEvent.SellOrderCompleted,
                    SellOrderCompletedEvent(
                        timestamp=time.time(),
                        order_id=order_id,
                        base_asset=base,
                        quote_asset=quote,
                        base_asset_amount=executed_amount,
                        quote_asset_amount=executed_amount * executed_price,
                        order_type=OrderType.LIMIT,
                        exchange_order_id=tx_hash,
                    ),
                )

            logger.info(f"Swap completed: {order_id} - {tx_hash}")

        except Exception as e:
            logger.error(f"Swap failed for {order_id}: {e}")
            self.trigger_event(
                MarketEvent.OrderFailure,
                {"order_id": order_id, "error": str(e)},
            )

    async def get_pool_info(
        self,
        base_token: str,
        quote_token: str,
    ) -> Dict[str, Any]:
        """
        Get information about a liquidity pool.

        Args:
            base_token: Base token symbol
            quote_token: Quote token symbol

        Returns:
            Pool information including reserves, TVL, APY
        """
        cache_key = f"{base_token}-{quote_token}"

        try:
            response = await self._get_gateway_instance().amm_pool_info(
                chain=self._chain,
                network=self._network,
                connector=self._connector_name,
                base_token=base_token,
                quote_token=quote_token,
            )

            pool_info = {
                "address": response.get("address", ""),
                "base_reserve": Decimal(str(response.get("reserveA", "0"))),
                "quote_reserve": Decimal(str(response.get("reserveB", "0"))),
                "total_liquidity": Decimal(str(response.get("totalLiquidity", "0"))),
                "fee": Decimal(str(response.get("fee", "0.003"))),
                "apy": response.get("apy"),
                "tvl_usd": response.get("tvlUSD"),
            }

            self._pool_info_cache[cache_key] = pool_info
            return pool_info

        except Exception as e:
            logger.warning(f"Failed to get pool info: {e}")
            return self._pool_info_cache.get(cache_key, {})

    async def add_liquidity(
        self,
        base_token: str,
        quote_token: str,
        base_amount: Decimal,
        quote_amount: Decimal,
        slippage_pct: Decimal = Decimal("0.5"),
    ) -> Dict[str, Any]:
        """
        Add liquidity to a pool.

        Args:
            base_token: Base token symbol
            quote_token: Quote token symbol
            base_amount: Amount of base token
            quote_amount: Amount of quote token
            slippage_pct: Slippage tolerance percentage

        Returns:
            Transaction result
        """
        try:
            response = await self._get_gateway_instance().amm_add_liquidity(
                chain=self._chain,
                network=self._network,
                connector=self._connector_name,
                address=self._wallet_address,
                base_token=base_token,
                quote_token=quote_token,
                amount_a=str(base_amount),
                amount_b=str(quote_amount),
                slippage_pct=float(slippage_pct),
            )

            logger.info(f"Added liquidity: {response.get('txHash', '')}")
            return response

        except Exception as e:
            logger.error(f"Failed to add liquidity: {e}")
            raise

    async def remove_liquidity(
        self,
        pool_address: str,
        liquidity_amount: Decimal,
        slippage_pct: Decimal = Decimal("0.5"),
    ) -> Dict[str, Any]:
        """
        Remove liquidity from a pool.

        Args:
            pool_address: Pool contract address
            liquidity_amount: Amount of LP tokens to burn
            slippage_pct: Slippage tolerance percentage

        Returns:
            Transaction result
        """
        try:
            response = await self._get_gateway_instance().amm_remove_liquidity(
                chain=self._chain,
                network=self._network,
                connector=self._connector_name,
                address=self._wallet_address,
                pool_address=pool_address,
                liquidity=str(liquidity_amount),
                slippage_pct=float(slippage_pct),
            )

            logger.info(f"Removed liquidity: {response.get('txHash', '')}")
            return response

        except Exception as e:
            logger.error(f"Failed to remove liquidity: {e}")
            raise

    async def get_lp_positions(self) -> List[Dict[str, Any]]:
        """
        Get current LP positions for the wallet.

        Returns:
            List of LP positions with details
        """
        try:
            response = await self._get_gateway_instance().amm_position_info(
                chain=self._chain,
                network=self._network,
                connector=self._connector_name,
                address=self._wallet_address,
            )

            return response.get("positions", [])

        except Exception as e:
            logger.warning(f"Failed to get LP positions: {e}")
            return []

    def _get_gateway_instance(self) -> GatewayHttpClient:
        """Get the Gateway HTTP client instance."""
        return GatewayHttpClient.get_instance(self._client_config)

    async def _update_order_status(self):
        """Update status of pending orders."""
        # AMM swaps are atomic, so no pending orders to track
        pass

    def cancel(self, trading_pair: str, order_id: str) -> bool:
        """
        Cancel an order (not supported for AMM swaps).

        AMM swaps are atomic and cannot be cancelled.
        """
        logger.warning("AMM swaps cannot be cancelled - they are atomic transactions")
        return False

    async def cancel_all(self, timeout_seconds: float) -> List[CancellationResult]:
        """Cancel all orders (not supported for AMM)."""
        return []

    @property
    def status_dict(self) -> Dict[str, bool]:
        """Get connector status."""
        return {
            "account_balance": len(self._account_balances) > 0,
            "trading_rule_initialized": True,
            "user_stream_initialized": True,
        }

    @property
    def ready(self) -> bool:
        """Check if connector is ready for trading."""
        return all(self.status_dict.values())
