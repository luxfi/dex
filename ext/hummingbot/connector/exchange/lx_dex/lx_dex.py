"""
LX DEX Spot Exchange Connector

Hummingbot connector for LX DEX spot/OrderBook trading.
Supports limit orders, market orders, and real-time order book updates.
"""

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from hummingbot.connector.exchange.lx_dex import lx_dex_constants as CONSTANTS
from hummingbot.connector.exchange.lx_dex.lx_dex_utils import (
    build_order_id,
    convert_from_exchange_symbol,
    convert_to_exchange_symbol,
    get_base_url,
    get_order_status,
    get_ws_url,
    str_to_decimal,
)
from hummingbot.connector.exchange_py_base import ExchangePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.connector.utils import combine_to_hb_trading_pair
from hummingbot.core.data_type.cancellation_result import CancellationResult
from hummingbot.core.data_type.common import OrderType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState, OrderUpdate, TradeUpdate
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.data_type.trade_fee import TokenAmount, TradeFeeBase, TradeFeeSchema
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.network_iterator import NetworkStatus
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory

logger = logging.getLogger(__name__)


class LxDexExchange(ExchangePyBase):
    """
    LX DEX Exchange connector for spot/OrderBook trading.

    This connector enables:
    - Limit and market order placement
    - Order cancellation
    - Real-time order book streaming
    - Balance tracking
    - Trade history
    """

    web_utils = None  # Set during initialization

    def __init__(
        self,
        client_config_map: "ClientConfigAdapter",
        lx_dex_api_key: str,
        lx_dex_api_secret: str,
        lx_dex_wallet_address: str,
        lx_dex_network: str = "mainnet",
        trading_pairs: Optional[List[str]] = None,
        trading_required: bool = True,
    ):
        self._api_key = lx_dex_api_key
        self._api_secret = lx_dex_api_secret
        self._wallet_address = lx_dex_wallet_address
        self._network = lx_dex_network
        self._trading_pairs = trading_pairs or []
        self._trading_required = trading_required
        self._last_timestamp = 0
        self._order_book_tracker = None
        self._user_stream_tracker = None

        super().__init__(client_config_map)

    @staticmethod
    def lx_dex_order_type(order_type: OrderType) -> str:
        """Convert Hummingbot order type to exchange format."""
        return {
            OrderType.LIMIT: "LIMIT",
            OrderType.LIMIT_MAKER: "LIMIT_MAKER",
            OrderType.MARKET: "MARKET",
        }.get(order_type, "LIMIT")

    @staticmethod
    def to_hb_order_type(exchange_order_type: str) -> OrderType:
        """Convert exchange order type to Hummingbot format."""
        return {
            "LIMIT": OrderType.LIMIT,
            "LIMIT_MAKER": OrderType.LIMIT_MAKER,
            "MARKET": OrderType.MARKET,
        }.get(exchange_order_type, OrderType.LIMIT)

    @property
    def authenticator(self):
        """Return the authenticator for API requests."""
        return LxDexAuth(
            api_key=self._api_key,
            api_secret=self._api_secret,
        )

    @property
    def name(self) -> str:
        return CONSTANTS.EXCHANGE_NAME

    @property
    def rate_limits_rules(self):
        return CONSTANTS.RATE_LIMITS

    @property
    def domain(self) -> str:
        return self._network

    @property
    def client_order_id_max_length(self) -> int:
        return CONSTANTS.MAX_ORDER_ID_LEN

    @property
    def client_order_id_prefix(self) -> str:
        return "hb"

    @property
    def trading_rules_request_path(self) -> str:
        return CONSTANTS.REST_URLS["symbols"]

    @property
    def trading_pairs_request_path(self) -> str:
        return CONSTANTS.REST_URLS["symbols"]

    @property
    def check_network_request_path(self) -> str:
        return CONSTANTS.REST_URLS["health"]

    @property
    def trading_pairs(self) -> List[str]:
        return self._trading_pairs

    @property
    def is_cancel_request_in_exchange_synchronous(self) -> bool:
        return True

    @property
    def is_trading_required(self) -> bool:
        return self._trading_required

    def supported_order_types(self) -> List[OrderType]:
        return [OrderType.LIMIT, OrderType.LIMIT_MAKER, OrderType.MARKET]

    def _is_request_exception_related_to_time_synchronizer(self, request_exception: Exception) -> bool:
        return False

    def _is_order_not_found_during_status_update_error(self, status_update_exception: Exception) -> bool:
        return "order not found" in str(status_update_exception).lower()

    def _is_order_not_found_during_cancelation_error(self, cancelation_exception: Exception) -> bool:
        return "order not found" in str(cancelation_exception).lower()

    async def _place_order(
        self,
        order_id: str,
        trading_pair: str,
        amount: Decimal,
        trade_type: TradeType,
        order_type: OrderType,
        price: Decimal,
        **kwargs,
    ) -> Tuple[str, float]:
        """Place an order on the exchange."""
        exchange_symbol = convert_to_exchange_symbol(trading_pair)
        side = "BUY" if trade_type == TradeType.BUY else "SELL"
        order_type_str = self.lx_dex_order_type(order_type)

        data = {
            "symbol": exchange_symbol,
            "side": side,
            "type": order_type_str,
            "quantity": str(amount),
            "clientOrderId": build_order_id(order_id),
            "walletAddress": self._wallet_address,
        }

        if order_type != OrderType.MARKET:
            data["price"] = str(price)
            data["timeInForce"] = "GTC"

        response = await self._api_request(
            method=RESTMethod.POST,
            path_url=CONSTANTS.REST_URLS["order"],
            data=data,
            is_auth_required=True,
        )

        exchange_order_id = response.get("orderId", "")
        transact_time = response.get("transactTime", time.time() * 1000) / 1000

        return exchange_order_id, transact_time

    async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder) -> bool:
        """Cancel an order on the exchange."""
        exchange_symbol = convert_to_exchange_symbol(tracked_order.trading_pair)

        data = {
            "symbol": exchange_symbol,
            "orderId": tracked_order.exchange_order_id,
            "walletAddress": self._wallet_address,
        }

        response = await self._api_request(
            method=RESTMethod.DELETE,
            path_url=CONSTANTS.REST_URLS["order"],
            data=data,
            is_auth_required=True,
        )

        return response.get("status") == "cancelled"

    async def _format_trading_rules(self, exchange_info: Dict[str, Any]) -> List[TradingRule]:
        """Parse trading rules from exchange info."""
        trading_rules = []

        for symbol_info in exchange_info.get("symbols", []):
            try:
                trading_pair = convert_from_exchange_symbol(symbol_info["symbol"])

                trading_rules.append(
                    TradingRule(
                        trading_pair=trading_pair,
                        min_order_size=Decimal(str(symbol_info.get("minOrderSize", "0.0001"))),
                        max_order_size=Decimal(str(symbol_info.get("maxOrderSize", "1000000"))),
                        min_price_increment=Decimal(str(symbol_info.get("tickSize", "0.00000001"))),
                        min_base_amount_increment=Decimal(str(symbol_info.get("stepSize", "0.00000001"))),
                        min_quote_amount_increment=Decimal(str(symbol_info.get("tickSize", "0.00000001"))),
                        min_notional_size=Decimal(str(symbol_info.get("minNotional", "1"))),
                    )
                )
            except Exception as e:
                logger.warning(f"Error parsing trading rule for {symbol_info}: {e}")

        return trading_rules

    async def _update_trading_fees(self):
        """Update trading fees from the exchange."""
        pass  # Fees are included in order responses

    async def _user_stream_event_listener(self):
        """Process user stream events (orders, trades, balances)."""
        async for event in self._iter_user_event_queue():
            try:
                event_type = event.get("type", "")

                if event_type == "order":
                    await self._process_order_update(event.get("data", {}))
                elif event_type == "trade":
                    await self._process_trade_update(event.get("data", {}))
                elif event_type == "balance":
                    await self._process_balance_update(event.get("data", {}))

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error processing user stream event: {e}")

    async def _process_order_update(self, data: Dict[str, Any]):
        """Process order update from user stream."""
        client_order_id = data.get("clientOrderId", "")
        if client_order_id.startswith("hb-"):
            client_order_id = client_order_id[3:]

        tracked_order = self._order_tracker.fetch_order(client_order_id=client_order_id)
        if tracked_order is None:
            return

        new_state = {
            "open": OrderState.OPEN,
            "partial": OrderState.PARTIALLY_FILLED,
            "filled": OrderState.FILLED,
            "cancelled": OrderState.CANCELED,
            "rejected": OrderState.FAILED,
        }.get(data.get("status", "").lower(), OrderState.OPEN)

        order_update = OrderUpdate(
            client_order_id=client_order_id,
            exchange_order_id=data.get("orderId", ""),
            trading_pair=tracked_order.trading_pair,
            update_timestamp=data.get("updateTime", time.time() * 1000) / 1000,
            new_state=new_state,
        )

        self._order_tracker.process_order_update(order_update)

    async def _process_trade_update(self, data: Dict[str, Any]):
        """Process trade update from user stream."""
        client_order_id = data.get("clientOrderId", "")
        if client_order_id.startswith("hb-"):
            client_order_id = client_order_id[3:]

        tracked_order = self._order_tracker.fetch_order(client_order_id=client_order_id)
        if tracked_order is None:
            return

        fee = TradeFeeBase.new_spot_fee(
            fee_schema=self.trade_fee_schema(),
            trade_type=tracked_order.trade_type,
            percent_token=data.get("feeCurrency", tracked_order.quote_asset),
            flat_fees=[
                TokenAmount(
                    token=data.get("feeCurrency", tracked_order.quote_asset),
                    amount=Decimal(str(data.get("fee", "0"))),
                )
            ],
        )

        trade_update = TradeUpdate(
            trade_id=data.get("tradeId", ""),
            client_order_id=client_order_id,
            exchange_order_id=data.get("orderId", ""),
            trading_pair=tracked_order.trading_pair,
            fee=fee,
            fill_base_amount=Decimal(str(data.get("quantity", "0"))),
            fill_quote_amount=Decimal(str(data.get("quoteQuantity", "0"))),
            fill_price=Decimal(str(data.get("price", "0"))),
            fill_timestamp=data.get("time", time.time() * 1000) / 1000,
        )

        self._order_tracker.process_trade_update(trade_update)

    async def _process_balance_update(self, data: Dict[str, Any]):
        """Process balance update from user stream."""
        for balance in data.get("balances", []):
            asset = balance.get("asset", "")
            free = Decimal(str(balance.get("free", "0")))
            locked = Decimal(str(balance.get("locked", "0")))

            self._account_balances[asset] = free + locked
            self._account_available_balances[asset] = free

    async def _all_trade_updates_for_order(self, order: InFlightOrder) -> List[TradeUpdate]:
        """Fetch all trades for a specific order."""
        trades = []

        try:
            response = await self._api_request(
                method=RESTMethod.GET,
                path_url=CONSTANTS.REST_URLS["my_trades"],
                params={
                    "symbol": convert_to_exchange_symbol(order.trading_pair),
                    "orderId": order.exchange_order_id,
                },
                is_auth_required=True,
            )

            for trade_data in response.get("trades", []):
                fee = TradeFeeBase.new_spot_fee(
                    fee_schema=self.trade_fee_schema(),
                    trade_type=order.trade_type,
                    percent_token=trade_data.get("feeCurrency", order.quote_asset),
                    flat_fees=[
                        TokenAmount(
                            token=trade_data.get("feeCurrency", order.quote_asset),
                            amount=Decimal(str(trade_data.get("fee", "0"))),
                        )
                    ],
                )

                trades.append(
                    TradeUpdate(
                        trade_id=trade_data.get("tradeId", ""),
                        client_order_id=order.client_order_id,
                        exchange_order_id=order.exchange_order_id,
                        trading_pair=order.trading_pair,
                        fee=fee,
                        fill_base_amount=Decimal(str(trade_data.get("quantity", "0"))),
                        fill_quote_amount=Decimal(str(trade_data.get("quoteQuantity", "0"))),
                        fill_price=Decimal(str(trade_data.get("price", "0"))),
                        fill_timestamp=trade_data.get("time", time.time() * 1000) / 1000,
                    )
                )

        except Exception as e:
            logger.warning(f"Error fetching trades for order {order.client_order_id}: {e}")

        return trades

    async def _request_order_status(self, tracked_order: InFlightOrder) -> OrderUpdate:
        """Fetch the current status of an order."""
        response = await self._api_request(
            method=RESTMethod.GET,
            path_url=CONSTANTS.REST_URLS["order"],
            params={
                "symbol": convert_to_exchange_symbol(tracked_order.trading_pair),
                "orderId": tracked_order.exchange_order_id,
            },
            is_auth_required=True,
        )

        new_state = {
            "open": OrderState.OPEN,
            "partial": OrderState.PARTIALLY_FILLED,
            "filled": OrderState.FILLED,
            "cancelled": OrderState.CANCELED,
            "rejected": OrderState.FAILED,
        }.get(response.get("status", "").lower(), OrderState.OPEN)

        return OrderUpdate(
            client_order_id=tracked_order.client_order_id,
            exchange_order_id=response.get("orderId", ""),
            trading_pair=tracked_order.trading_pair,
            update_timestamp=response.get("updateTime", time.time() * 1000) / 1000,
            new_state=new_state,
        )

    async def _update_balances(self):
        """Fetch and update account balances."""
        response = await self._api_request(
            method=RESTMethod.GET,
            path_url=CONSTANTS.REST_URLS["balances"],
            params={"walletAddress": self._wallet_address},
            is_auth_required=True,
        )

        self._account_balances.clear()
        self._account_available_balances.clear()

        for balance in response.get("balances", []):
            asset = balance.get("asset", "")
            free = Decimal(str(balance.get("free", "0")))
            locked = Decimal(str(balance.get("locked", "0")))

            self._account_balances[asset] = free + locked
            self._account_available_balances[asset] = free

    async def _api_request(
        self,
        method: RESTMethod,
        path_url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        is_auth_required: bool = False,
    ) -> Dict[str, Any]:
        """Make an API request to the exchange."""
        base_url = get_base_url(self._network)
        url = f"{base_url}{path_url}"

        headers = {
            "Content-Type": "application/json",
            "X-Client": "hummingbot",
        }

        if is_auth_required:
            headers["X-API-Key"] = self._api_key
            # Add signature logic here

        async with aiohttp.ClientSession() as session:
            if method == RESTMethod.GET:
                async with session.get(url, params=params, headers=headers) as response:
                    return await response.json()
            elif method == RESTMethod.POST:
                async with session.post(url, json=data, headers=headers) as response:
                    return await response.json()
            elif method == RESTMethod.DELETE:
                async with session.delete(url, json=data, headers=headers) as response:
                    return await response.json()

        return {}

    def trade_fee_schema(self) -> TradeFeeSchema:
        """Return the fee schema for this exchange."""
        return TradeFeeSchema(
            maker_percent_fee_decimal=Decimal(str(CONSTANTS.DEFAULT_MAKER_FEE)),
            taker_percent_fee_decimal=Decimal(str(CONSTANTS.DEFAULT_TAKER_FEE)),
            buy_percent_fee_deducted_from_returns=True,
        )


class LxDexAuth:
    """Authentication handler for LX DEX API."""

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret

    def get_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        timestamp = str(int(time.time() * 1000))
        return {
            "X-API-Key": self._api_key,
            "X-Timestamp": timestamp,
        }

    def sign_request(self, method: str, path: str, params: Dict[str, Any]) -> str:
        """Sign a request with HMAC-SHA256."""
        import hashlib
        import hmac

        timestamp = str(int(time.time() * 1000))
        message = f"{timestamp}{method}{path}"
        if params:
            message += str(sorted(params.items()))

        signature = hmac.new(
            self._api_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()

        return signature
